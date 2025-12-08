import asyncio
import dataclasses
import json
import os
import re
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Set, Optional, Dict

from omegaconf import DictConfig

from ms_agent import LLMAgent
from ms_agent.agent import CodeAgent
from ms_agent.llm import Message
from ms_agent.memory.condenser.code_condenser import CodeCondenser
from ms_agent.tools.code_server import LSPCodeServer
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_TAG, DEFAULT_INDEX_DIR, DEFAULT_LOCK_DIR
from ms_agent.utils.utils import extract_code_blocks, file_lock
from utils import parse_imports, stop_words

logger = get_logger()


class Programmer(LLMAgent):

    def __init__(self,
                 config: DictConfig = DictConfig({}),
                 tag: str = DEFAULT_TAG,
                 trust_remote_code: bool = False,
                 code_file: str = None,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.code_file = code_file
        index_dir = getattr(config, 'index_cache_dir', DEFAULT_INDEX_DIR)
        self.index_dir = os.path.join(self.output_dir, index_dir)
        self.code_condenser = CodeCondenser(config)
        
        # LSP incremental checking - support multiple languages
        self.lsp_servers: Dict[str, LSPCodeServer] = {}  # language -> server instance
        self.project_languages: Set[str] = set()  # Detected languages in project
        self.accumulated_code = ''  # Accumulate code segments
        self.current_file_name = ''  # Track current file
        self.import_phase_done = False  # Track if import phase is done

    async def condense_memory(self, messages):
        return messages

    async def add_memory(self, messages, **kwargs):
        return

    async def on_task_begin(self, messages: List[Message]):
        if 'extra_body' not in self.llm.args:
            self.llm.args['extra_body'] = DictConfig({})
        self.llm.args['extra_body']['stop_sequences'] = stop_words
        self.code_files = [self.code_file]
        self.find_all_files()
        
        # Initialize LSP server
        await self._init_lsp_server()
    
    async def _init_lsp_server(self):
        """Initialize LSP servers based on project languages"""
        try:
            framework_file = os.path.join(self.output_dir, 'framework.txt')
            if not os.path.exists(framework_file):
                return
                
            with open(framework_file, 'r') as f:
                framework = f.read().lower()
            
            # Detect all languages in the project
            detected_languages = set()
            
            if any(kw in framework for kw in ['typescript', 'javascript', 'react', 'vue', 'node', 'npm']):
                detected_languages.add('typescript')
            
            if any(kw in framework for kw in ['python', 'django', 'flask', 'fastapi']):
                detected_languages.add('python')
            
            if any(kw in framework for kw in ['java', 'spring', 'maven', 'gradle']):
                detected_languages.add('java')
            
            if not detected_languages:
                logger.warning("No supported languages detected in framework.txt")
                return
            
            self.project_languages = detected_languages
            logger.info(f"Detected project languages: {', '.join(detected_languages)}")
            
            # Initialize LSP server for each detected language
            lsp_config = DictConfig({
                'workspace_dir': self.output_dir,
                'output_dir': self.output_dir
            })
            
            for lang in detected_languages:
                try:
                    lsp_server = LSPCodeServer(lsp_config)
                    await lsp_server.connect()
                    self.lsp_servers[lang] = lsp_server
                    logger.info(f"LSP Code Server initialized for {lang}")
                except Exception as e:
                    logger.warning(f"Failed to initialize LSP server for {lang}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to initialize LSP servers: {e}")
            self.lsp_servers = {}
    
    async def _incremental_lsp_check(self, code_file: str, partial_code: str) -> Optional[str]:
        """Incrementally check code quality using appropriate LSP server"""
        if not self.lsp_servers:
            return None
            
        try:
            # Determine language from file extension
            file_ext = os.path.splitext(code_file)[1].lower()
            
            # Map file extension to language
            lang = None
            if file_ext in ['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.vue']:
                lang = 'typescript'
            elif file_ext == '.py':
                lang = 'python'
            elif file_ext in ['.java', '.kt', '.kts']:
                lang = 'java'
            else:
                logger.debug(f"Unsupported file extension: {file_ext}")
                return None
            
            # Get LSP server for this language
            lsp_server = self.lsp_servers.get(lang)
            if not lsp_server:
                logger.debug(f"No LSP server initialized for {lang}")
                return None
                
            result = await lsp_server.call_tool(
                "lsp_code_server",
                tool_name="update_and_check",
                tool_args={
                    "file_path": code_file,
                    "content": partial_code,
                    "language": lang
                }
            )
            
            diagnostics = json.loads(result)
            
            if diagnostics.get('has_errors'):
                issues = diagnostics.get('diagnostics', [])
                # Filter critical errors only
                critical_errors = [
                    d for d in issues 
                    if d.get('severity') == 'Error' and 
                    'expected' not in d.get('message', '').lower()
                ]
                
                if critical_errors:
                    error_msg = f"\n⚠️ LSP detected {len(critical_errors)} critical issues:\n"
                    for i, diag in enumerate(critical_errors[:3], 1):
                        line = diag.get('line', 0)
                        msg = diag.get('message', '')
                        error_msg += f"{i}. Line {line}: {msg}\n"
                    error_msg += "Please fix these issues before continuing.\n"
                    return error_msg
                    
        except Exception as e:
            logger.debug(f"LSP check failed for {code_file}: {e}")
            
        return None
    
    async def on_task_end(self, messages: List[Message]):
        """Clean up all LSP servers"""
        if self.lsp_servers:
            for lang, lsp_server in self.lsp_servers.items():
                try:
                    await lsp_server.cleanup()
                    logger.info(f"LSP server for {lang} cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up LSP server for {lang}: {e}")

    def filter_code_files(self):
        code_files = []
        for code_file in self.code_files:
            if not os.path.exists(os.path.join(self.output_dir, code_file)):
                code_files.append(code_file)
        self.code_files = code_files

    def find_all_files(self):
        self.all_code_files = []
        with open(os.path.join(self.output_dir, 'file_order.txt'), 'r') as f:
            for group in json.load(f):
                self.all_code_files.extend(group['files'])

    def find_all_read_files(self, messages):
        files = []
        for message in messages:
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if 'read_file' in tool_call['tool_name']:
                        arguments = tool_call['arguments']
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                                files.extend(arguments['paths'])
                            except json.decoder.JSONDecodeError:
                                pass
        return set(files)

    def read_index_file(self, path):
        with open(os.path.join(self.index_dir, path), 'r') as f:
            return f.read()

    async def after_tool_call(self, messages: List[Message]):
        deps_not_exist = False
        pattern = r'<result>[a-zA-Z]*:([^\n\r`]+)\n(.*?)'
        matches = re.findall(pattern, messages[-1].content, re.DOTALL)
        try:
            code_file = next(iter(matches))[0].strip()
        except StopIteration:
            code_file = ''
        is_config = code_file.endswith('.json') or code_file.endswith(
            '.yaml') or code_file.endswith('.md')
        if 'abbr' in code_file:
            print()
        coding_finish = '<result>' in messages[
            -1].content and '</result>' in messages[-1].content
        import_finish = '<result>' in messages[-1].content and self.llm.args[
            'extra_body'][
                'stop_sequences'] == stop_words and '</result>' not in messages[
                    -1].content and not is_config
        # Detect if stop_words triggered during code body generation
        code_continue = (not coding_finish and not import_finish and 
                        self.import_phase_done and 
                        '<result>' in messages[-1].content and
                        '</result>' not in messages[-1].content and
                        not is_config)
        
        # Detect fix and continue response
        has_fix = '<fix_line>' in messages[-1].content if messages[-1].role == 'assistant' else False
        has_continue = '<continue>' in messages[-1].content if messages[-1].role == 'assistant' else False
        
        has_tool_call = len(messages[-1].tool_calls
                            or []) > 0 or messages[-1].role != 'assistant'
        
        if (not has_tool_call) and (has_fix or has_continue):
            # Handle model's fix and continue response
            content = messages[-1].content
            
            # Process fixes if any
            if has_fix:
                fix_pattern = r'<fix_line>\s*([\s\S]*?)\s*</fix_line>'
                fix_matches = re.findall(fix_pattern, content)
                for fix_content in fix_matches:
                    # Parse line fixes: "line_number: code"
                    for line in fix_content.strip().split('\n'):
                        if ':' in line:
                            try:
                                line_num_str, fixed_code = line.split(':', 1)
                                line_num = int(line_num_str.strip())
                                # Apply fix to accumulated code
                                code_lines = self.accumulated_code.split('\n')
                                if 0 < line_num <= len(code_lines):
                                    code_lines[line_num - 1] = fixed_code.strip()
                                    self.accumulated_code = '\n'.join(code_lines)
                            except (ValueError, IndexError) as e:
                                logger.debug(f"Failed to parse fix: {line}, error: {e}")
            
            # Process continue segment
            if has_continue:
                continue_pattern = r'<continue>\s*([\s\S]*?)\s*</continue>'
                continue_matches = re.findall(continue_pattern, content)
                if continue_matches:
                    # Append continuation
                    continuation = continue_matches[0].strip()
                    self.accumulated_code += '\n' + continuation
            
            # Remove the assistant message and trigger next check
            messages.pop(-1)
            # Simulate stop_words trigger to continue checking
            code_continue = True
        
        if (not has_tool_call) and import_finish:
            # First time after imports - load dependencies
            if os.path.isfile(os.path.join(self.output_dir, code_file)):
                index_content = self.read_index_file(code_file)
                messages.append(
                    Message(
                        role='user',
                        content=
                        f'We break your generation to import more relative information. '
                        f'The file: {code_file} you are generating has existed already, here is the abbreviate content:\n'
                        f'{index_content}\n'
                        f'Read the content and generating other code files.'
                    ))
            else:
                contents = messages[-1].content.split('\n')
                comments = ['*', "#", '-', '%', '/']
                contents = [c for c in contents if not any(c.strip().startswith(cm) for cm in comments)]
                content = [c for c in contents if '<result>' in c and ':' in c][0]
                code_file = content.split('<result>')[1].split(':')[1].split(
                    '\n')[0].strip()
                
                # Initialize for this file
                self.current_file_name = code_file
                self.accumulated_code = ''
                
                all_files = parse_imports(code_file, '\n'.join(contents),
                                          self.output_dir) or []
                all_read_files = self.find_all_read_files(messages)
                deps = []
                definitions = []
                folders = []
                wrong_imports = []
                for file in all_files:
                    if file.source_file == code_file:
                        wrong_imports.append(f'You should not import the file itself: {code_file}')
                        continue
                    filename = os.path.join(self.output_dir, file.source_file)
                    if not os.path.exists(filename):
                        if file.source_file in self.all_code_files:
                            deps_not_exist = True
                            self.code_files.append(file.source_file)
                        else:
                            wrong_imports.append(file.source_file)
                    elif os.path.isfile(filename):
                        if file.source_file not in all_read_files:
                            deps.append(file.source_file)
                            definitions.extend(file.imported_items)
                    else:
                        folders.append(
                            f'You are importing {file.imported_items} from {file.source_file} folder'
                        )

                if not deps_not_exist:
                    dep_content = ''
                    for dep in deps:
                        content = self.read_index_file(dep)
                        need_detail = False
                        for definition in definitions:
                            if definition not in content:
                                need_detail = True
                                break
                        if need_detail:
                            detail_file = os.path.join(self.output_dir, dep)
                            with open(detail_file, 'r') as f:
                                content = f.read()
                        dep_content += f'File content {dep}:\n{content}\n\n'
                    if folders:
                        folders = '\n'.join(folders)
                        dep_content += (
                            f'Some definitions come from folders:\n{folders}\nYou need to check the definition '
                            f'file with `read_file` tool if they are not in your context.\n'
                        )
                    if wrong_imports:
                        wrong_imports = '\n'.join(wrong_imports)
                        dep_content += (
                            f'Some import files are not in the project plans: {wrong_imports}, '
                            f'check the error now.\n')
                    
                    content = messages.pop(-1).content.split('<result>')[1]
                    self.accumulated_code = content  # Start accumulating
                    
                    messages.append(
                        Message(
                            role='user',
                            content=
                            f'We break your generation to import more relative information. '
                            f'According to your imports, some extra contents manually given here:\n'
                            f'\n{dep_content or "No extra dependencies needed"}\n'
                            f'Here is the start of your code: {content}\n\n'
                            f'**IMPORTANT**: Continue generating {code_file}.\n'
                            f'Separate functions/methods/classes with double newlines (\\n\\n).\n'
                            f'The generation will pause at these boundaries for quality checks.\n'
                            f'When complete, use </result> to finish.\n'
                        ))
                    if not wrong_imports:
                        # Set stop sequences for code body: \n\n + language-specific keywords
                        code_stop_sequences = [
                            '\n\n',              # Primary: double newline (function/method boundary)
                            '\nclass ',          # JS/TS/Java/Python: class definition
                            '\nfunction ',       # JS/TS: function definition  
                            '\nexport class ',   # TS: exported class
                            '\nexport function ', # TS: exported function
                            '\ndef ',            # Python: function definition
                            '\nclass ',          # Python: class definition
                            '\npublic class ',   # Java: public class
                            '\nprivate class ',  # Java: private class
                            '\npublic ',         # Java: public method/field
                            '\nprotected ',      # Java: protected method/field
                        ]
                        self.llm.args['extra_body']['stop_sequences'] = code_stop_sequences
                        self.import_phase_done = True
                        
        elif (not has_tool_call) and code_continue:
            # Code body generation continues - accumulate and check
            content = messages[-1].content
            
            # Extract code segment
            if '<result>' in content:
                content_segment = content.split('<result>')[1]
            else:
                content_segment = content
            
            # Remove the message and accumulate
            messages.pop(-1)
            self.accumulated_code += content_segment
            
            # Run LSP check on accumulated code
            lsp_feedback = await self._incremental_lsp_check(self.current_file_name, self.accumulated_code)
            
            # Prepare feedback message
            feedback_msg = f'Progress: {len(self.accumulated_code.splitlines())} lines generated.\n'
            
            if lsp_feedback:
                feedback_msg += lsp_feedback
                feedback_msg += '\n**Fix the issues** using:\n'
                feedback_msg += '<fix_line>\n'
                feedback_msg += 'line_number: corrected code\n'
                feedback_msg += '</fix_line>\n'
                feedback_msg += 'Then continue with <continue>code</continue>\n'
            else:
                feedback_msg += '✓ No issues. Continue generating.\n'
                feedback_msg += 'Remember to use double newlines (\\n\\n) between functions/methods/classes.\n'
            
            messages.append(Message(role='user', content=feedback_msg))
            # Stop sequences remain active
        elif (not has_tool_call) and coding_finish:
            result, remaining_text = extract_code_blocks(messages[-1].content)
            if result:
                _response = remaining_text
                saving_result = ''
                for r in result:
                    path = r['filename']
                    # Use accumulated code if available, otherwise use extracted code
                    if r['filename'] == self.current_file_name and self.accumulated_code:
                        code = self.accumulated_code
                        # Extract final segment and append
                        final_segment = messages[-1].content.split('<result>')[1].split('</result>')[0] if '</result>' in messages[-1].content else ''
                        if final_segment and final_segment not in code:
                            code += final_segment
                    else:
                        code = r['code']
                    
                    path = os.path.join(self.output_dir, path)

                    lock_dir = os.path.join(self.output_dir, DEFAULT_LOCK_DIR)

                    # Check and write file with lock
                    with file_lock(lock_dir, r['filename']):
                        file_exists = os.path.exists(path)
                        if not file_exists:
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            with open(path, 'w') as f:
                                f.write(code)
                        else:
                            with open(path, 'r') as f:
                                code = f.read()
                            _response += f'\n```{path.split(".")[-1]}: {r["filename"]}\n{code}\n```\n'

                    saving_result += f'Save file <{r["filename"]}> successfully\n'
                
                # Reset for next file
                self.accumulated_code = ''
                self.current_file_name = ''
                self.import_phase_done = False

                messages.append(Message(role='user', content=saving_result))
                self.llm.args['extra_body']['stop_sequences'] = stop_words
            self.filter_code_files()
            if not self.code_files:
                self.runtime.should_stop = True

        new_task = coding_finish and self.code_files
        if not has_tool_call and (deps_not_exist or new_task):
            last_file = self.code_files[-1]
            messages.append(
                Message(
                    role='user',
                    content=
                    f'\nA code file in your imports not found, you should write it first: {last_file}\n'
                ))
            self.llm.args['extra_body']['stop_sequences'] = stop_words

        await self.code_condenser.run(messages)


@dataclasses.dataclass
class FileRelation:

    name: str
    description: str
    done: bool = False
    deps: Set[str] = dataclasses.field(default_factory=set)


class CodingAgent(CodeAgent):

    async def write_code(self, topic, user_story, framework, protocol, name,
                         description, fast_fail):
        logger.info(f'Writing {name}')
        _config = deepcopy(self.config)
        messages = [
            Message(role='system', content=self.config.prompt.system),
            Message(
                role='user',
                content=f'原始需求(topic.txt): {topic}\n'
                f'LLM规划的用户故事(user_story.txt): {user_story}\n'
                f'技术栈(framework.txt): {framework}\n'
                f'通讯协议(protocol.txt): {protocol}\n'
                f'你需要编写的文件: {name}\n文件描述: {description}\n'),
        ]

        _config = deepcopy(self.config)
        _config.save_history = True
        _config.load_cache = False
        programmer = Programmer(
            _config,
            tag=f'programmer-{name.replace(os.sep, "-")}',
            trust_remote_code=True,
            code_file=name)
        await programmer.run(messages)

    async def execute_code(self, inputs, **kwargs):
        with open(os.path.join(self.output_dir, 'topic.txt')) as f:
            topic = f.read()
        with open(os.path.join(self.output_dir, 'user_story.txt')) as f:
            user_story = f.read()
        with open(os.path.join(self.output_dir, 'framework.txt')) as f:
            framework = f.read()
        with open(os.path.join(self.output_dir, 'protocol.txt')) as f:
            protocol = f.read()

        file_orders = self.construct_file_orders()
        file_relation = OrderedDict()
        self.refresh_file_status(file_relation)
        lock_dir = os.path.join(self.output_dir, 'locks')
        shutil.rmtree(lock_dir, ignore_errors=True)

        max_workers = 5

        for files in file_orders:
            while True:
                files = self.filter_done_files(files)
                files = self.find_description(files)
                self.construct_file_information(file_relation)
                if not files:
                    break

                # Convert async tasks to sync wrapper for thread pool
                def write_code_sync(name, description):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            self.write_code(
                                topic,
                                user_story,
                                framework,
                                protocol,
                                name,
                                description,
                                fast_fail=False))
                    finally:
                        loop.close()

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(write_code_sync, name, description)
                        for name, description in files.items()
                    ]
                    # Wait for all tasks to complete
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f'Error writing code: {e}')

            self.refresh_file_status(file_relation)

        self.construct_file_information(file_relation)
        return inputs

    def construct_file_orders(self):
        with open(os.path.join(self.output_dir, 'file_order.txt')) as f:
            file_order = json.load(f)

        file_orders = []
        for files in file_order:
            file_orders.append(files['files'])
        return file_orders

    def find_description(self, files):
        file_desc = {file: '' for file in files}
        with open(os.path.join(self.output_dir, 'file_design.txt')) as f:
            file_design = json.load(f)

        for module in file_design:
            files = module['files']
            for file in files:
                name = file['name']
                description = file['description']
                if name in file_desc:
                    file_desc[name] = description
        return file_desc

    def filter_done_files(self, file_group):
        output = []
        with open(os.path.join(self.output_dir, 'file_design.txt')) as f:
            file_designs = json.load(f)

        for file_design in file_designs:
            files = file_design['files']
            for file in files:
                file_name = file['name']
                file_path = os.path.join(self.output_dir, file_name)
                if file_name in file_group and not os.path.exists(file_path):
                    output.append(file_name)
        return output

    def refresh_file_status(self, file_relation):
        with open(os.path.join(self.output_dir, 'file_design.txt')) as f:
            file_designs = json.load(f)

        for file_design in file_designs:
            files = file_design['files']
            for file in files:
                file_name = file['name']
                description = file['description']
                file_path = os.path.join(self.output_dir, file_name)
                if file_name not in file_relation:
                    file_relation[file_name] = FileRelation(
                        name=file_name, description=description)
                file_relation[file_name].done = os.path.exists(file_path)

    def construct_file_information(self, file_relation, add_output_dir=False):
        file_info = '以下文件按架构设计编写顺序排序：\n'
        for file, relation in file_relation.items():
            if add_output_dir:
                file = os.path.join(self.output_dir, file)
            if relation.done:
                file_info += f'{file}: ✅已构建\n'
            else:
                file_info += f'{file}: ❌未构建\n'
        with open(os.path.join(self.output_dir, 'tasks.txt'), 'w') as f:
            f.write(file_info)
