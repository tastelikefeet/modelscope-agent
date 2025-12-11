import asyncio
import dataclasses
import json
import os
import shutil
from collections import OrderedDict
from copy import deepcopy
from typing import List, Set, Optional

from omegaconf import DictConfig

from ms_agent import LLMAgent
from ms_agent.agent import CodeAgent
from ms_agent.llm import Message
from ms_agent.memory.condenser.code_condenser import CodeCondenser
from ms_agent.tools.code_server import LSPCodeServer
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_TAG, DEFAULT_INDEX_DIR, DEFAULT_LOCK_DIR
from ms_agent.utils.utils import extract_code_blocks, file_lock
from .utils import parse_imports, ImportInfo

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
        self.lock_dir = os.path.join(self.output_dir, DEFAULT_LOCK_DIR)
        self.code_condenser = CodeCondenser(config)
        # self.refine_condenser = RefineCondenser(config)
        self.code_files = []
        # LSP incremental checking - lazy creation, shared across Programmers
        self.shared_lsp_context = kwargs.get('shared_lsp_context', {})
        self.unchecked_files = {}
        self.unchecked_issues = {}

    async def condense_memory(self, messages):
        return messages

    async def add_memory(self, messages, **kwargs):
        return

    async def on_task_begin(self, messages: List[Message]):
        self.code_files = [self.code_file]

    async def _incremental_check(self, code_file: str, partial_code: str):
        lsp_result = await self._incremental_lsp_check(code_file, partial_code)
        import_result = await self._import_check(code_file, partial_code)
        return (lsp_result or '') + (import_result or '')


    async def _import_check(self, code_file: str, partial_code: str) -> Optional[str]:
        errors = []
        all_imports: List[ImportInfo] = parse_imports(code_file, partial_code, self.output_dir)
        
        for info in all_imports:
            source_file = info.source_file
            if not source_file:
                continue

            if not os.path.isabs(source_file):
                full_path = os.path.join(self.output_dir, source_file)
            else:
                full_path = source_file
            
            # 1. Check file existence
            if not os.path.exists(full_path):
                if os.path.isdir(full_path):
                    index_found = False
                    init_path = os.path.join(full_path, '__init__.py')
                    if os.path.exists(init_path):
                        full_path = init_path
                        index_found = True
                    else:
                        for index_file in ['index.ts', 'index.tsx', 'index.js', 'index.jsx', 'index.vue']:
                            index_path = os.path.join(full_path, index_file)
                            if os.path.exists(index_path):
                                full_path = index_path
                                index_found = True
                                break
                    
                    if not index_found:
                        errors.append(
                            f"Import error in {code_file}:\n"
                            f"  Directory '{source_file}' exists but has no index file (__init__.py, index.ts, etc.)\n"
                            f"  Statement: {info.raw_statement}\n"
                        )
                        continue
                else:
                    errors.append(
                        f"Import error in {code_file}:\n"
                        f"  File '{source_file}' does not exist\n"
                        f"  Statement: {info.raw_statement}\n"
                    )
                    continue
            
            # 2. Check if imported symbols exist in the file
            # Skip side-effect imports (import './style.css') and wildcard imports (import * as X)
            if info.import_type == 'side-effect' or info.imported_items == ['*']:
                continue
            
            if not info.imported_items:
                continue

            with open(full_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            missing_items = []
            for item in info.imported_items:
                if item not in file_content:
                    missing_items.append(item)

            if missing_items:
                errors.append(
                    f"Import error in {code_file}:\n"
                    f"  Items {missing_items} not found in '{source_file}'\n"
                    f"  Statement: {info.raw_statement}\n"
                )
        
        return '\n'.join(errors) if errors else None

    
    async def _incremental_lsp_check(self, code_file: str, partial_code: str) -> Optional[str]:
        lsp_servers = self.shared_lsp_context.get('lsp_servers', {})
        if not lsp_servers:
            return None

        file_basename = os.path.basename(code_file)
        if file_basename in LSPCodeServer.skip_files:
            logger.debug(f"Skipping LSP check for config file: {code_file}")
            return None

        if code_file.endswith('.vue'):
            return None

        lsp_lock = self.shared_lsp_context.get('lsp_lock')
        if lsp_lock is None:
            lsp_lock = asyncio.Lock()
            self.shared_lsp_context['lsp_lock'] = lsp_lock
            
        async with lsp_lock:
            file_ext = os.path.splitext(code_file)[1].lower()
            lang = None
            for key, value in LSPCodeServer.language_mapping.items():
                if file_ext in value:
                    lang = key
                    break

            if lang is None:
                return None

            lsp_server = lsp_servers.get(lang)
            if not lsp_server:
                logger.debug(f"No LSP server initialized for {lang}")
                return None

            return await lsp_server.call_tool(
                "lsp_code_server",
                tool_name="update_and_check",
                tool_args={
                    "file_path": code_file,
                    "content": partial_code,
                    "language": lang
                }
            )

    def filter_code_files(self):
        code_files = []
        for code_file in self.code_files:
            if not os.path.exists(os.path.join(self.output_dir, code_file)):
                code_files.append(code_file)
        self.code_files = code_files

    @staticmethod
    def find_all_read_files(messages):
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

    def add_unchecked_file(self, untrack_file):
        self.unchecked_files[untrack_file] = 0

    def increment_unchecked_file(self):
        for key in list(self.unchecked_files.keys()):
            self.unchecked_files[key] = self.unchecked_files[key] + 1
            if self.unchecked_files[key] > 3:
                self.unchecked_files.pop(key)
                logger.error(f"Unchecked file {key} still have problem:\n{self.unchecked_issues.get('key')}\n"
                             f"But the checking limit has reached.")

    async def after_tool_call(self, messages: List[Message]):
        coding_finish = '<result>' in messages[
            -1].content and '</result>' in messages[-1].content
        
        has_tool_call = len(messages[-1].tool_calls
                            or []) > 0 or messages[-1].role != 'assistant'
        
        if (not has_tool_call) and coding_finish:
            message = messages[-1]
            result, remaining_text = extract_code_blocks(message.content)
            if result:
                _response = remaining_text
                saving_result = ''
                for r in result:
                    path = r['filename']
                    code = r['code']

                    path = os.path.join(self.output_dir, path)

                    lock_dir = os.path.join(self.output_dir, DEFAULT_LOCK_DIR)

                    with file_lock(lock_dir, r['filename']):
                        new_file = not os.path.exists(path)
                        if new_file:
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            with open(path, 'w') as f:
                                f.write(code)
                            self.add_unchecked_file(r['filename'])
                        else:
                            with open(path, 'r') as f:
                                code = f.read()
                        _response += f'\n<result>{path.split(".")[-1]}: {r["filename"]}\n{code}\n</result>\n'
                    saving_result += f'Save file <{r["filename"]}> successfully\n'
                message.content = _response
                messages.append(Message(role='user', content=saving_result))
        if not has_tool_call:
            if not messages[-1].content:
                messages[-1].content = 'I should continue to solve the problem.'
            all_issues = []

            for uncheck_file in list(self.unchecked_files.keys()):
                with open(os.path.join(self.output_dir, uncheck_file), 'r') as f:
                    _code = f.read()
                lsp_feedback = await self._incremental_check(uncheck_file, _code)
                if lsp_feedback:
                    all_issues.append(f'❎Issues in {uncheck_file}:' + lsp_feedback)
                    self.unchecked_issues[uncheck_file] = lsp_feedback
                else:
                    logger.info(f'✅No issues found in {uncheck_file}.')
                    self.unchecked_files.pop(uncheck_file)
            self.increment_unchecked_file()

            if all_issues:
                all_issues = '\n'.join(all_issues)
                logger.warning(f'Compile error in {self.tag}:')
                logger.warning(all_issues)
                all_issues = (f'We check the code with LSP server, here are the issues found:\n'
                              f'{all_issues}\n'
                              f'You can read related file to find the root cause if needed\n'
                              f'Then fix the file with `replace_file_contents`\n'
                              f'Some tips:\n'
                              f'1. Check any code file not in your dependencies and not in the `file_design.txt`\n'
                              f'2. Consider the relative path mistakes to your current writing file location\n'
                              f'3. Do not rewrite the code with <result></result> after fixing with `replace_file_contents`\n'
                              )
                messages.append(Message(role='user', content=all_issues))
                messages[0].content = self.config.prompt.system

        self.filter_code_files()
        if not self.code_files and not self.unchecked_files:
            self.runtime.should_stop = True

        new_task = (not has_tool_call) and self.code_files and (not self.unchecked_files)
        if new_task:
            last_file = self.code_files[-1]
            messages.append(
                Message(
                    role='user',
                    content=
                    f'\nA code file in your imports not found, you should write it first: {last_file}\n'
                ))

        await self.code_condenser.run(messages)


@dataclasses.dataclass
class FileRelation:

    name: str
    description: str
    done: bool = False
    deps: Set[str] = dataclasses.field(default_factory=set)


class CodingAgent(CodeAgent):
    
    def __init__(self, config, tag, trust_remote_code, **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        # Shared LSP context across all Programmers
        self.shared_lsp_context = {}
    
    async def _init_lsp_servers(self):
        framework_file = os.path.join(self.output_dir, 'framework.txt')
        if not os.path.exists(framework_file):
            logger.info("framework.txt not found, skipping LSP initialization")
            return

        with open(framework_file, 'r') as f:
            framework = f.read().lower()

        # Detect all languages in the project
        detected_languages = set()

        if any(kw in framework for kw in ['typescript', 'javascript', 'react', 'node', 'npm', 'html']):
            detected_languages.add('typescript')

        if any(kw in framework for kw in ['python', 'django', 'flask', 'fastapi']):
            detected_languages.add('python')

        if any(kw in framework for kw in ['java ', 'java\n', 'spring', 'maven', 'gradle']):
            detected_languages.add('java')

        if not detected_languages:
            logger.info("No supported languages detected in framework.txt")
            return

        logger.info(f"Initializing LSP servers for languages: {', '.join(detected_languages)}")

        # Initialize LSP server for each detected language
        lsp_config = DictConfig({
            'workspace_dir': self.output_dir,
            'output_dir': self.output_dir
        })

        lsp_servers = {}
        for lang in detected_languages:
            lsp_server = LSPCodeServer(lsp_config)
            await lsp_server.connect()
            lsp_servers[lang] = lsp_server
            logger.info(f"LSP Code Server created for {lang}")

        for lang, lsp_server in lsp_servers.items():
            logger.info(f"Building LSP index for {lang}...")
            await lsp_server.call_tool(
                "lsp_code_server",
                tool_name="check_directory",
                tool_args={
                    "directory": '',
                    "language": lang
                }
            )
            logger.info(f"LSP index built for {lang}")

        self.shared_lsp_context['lsp_servers'] = lsp_servers
        self.shared_lsp_context['project_languages'] = detected_languages
        logger.info(f"LSP servers ready for use")
    
    async def _cleanup_lsp_servers(self):
        lsp_servers = self.shared_lsp_context.get('lsp_servers', {})
        if lsp_servers:
            for lang, lsp_server in lsp_servers.items():
                try:
                    await lsp_server.cleanup()
                    lsp_server.cleanup_lsp_index_dirs()
                except Exception: # noqa
                    pass

    async def write_code(self, topic, user_story, framework, protocol, name,
                         description, index, last_batch, siblings, next_batch):
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
                        f'你需要编写的文件: {name}\n'
                        f'文件编写index: {index}\n'
                        f'文件描述: {description}\n'
                        f'上一批编写的代码:{last_batch}\n'
                        f'其他workers在并行编写:{siblings}\n'
                        f'下一批编写的代码:{next_batch}\n'),
        ]

        _config = deepcopy(self.config)
        _config.save_history = True
        _config.load_cache = False
        programmer = Programmer(
            _config,
            tag=f'programmer-{name.replace(os.sep, "-")}',
            trust_remote_code=True,
            code_file=name,
            shared_lsp_context=self.shared_lsp_context)  # Pass shared context
        await programmer.run(messages)

    async def execute_code(self, inputs, **kwargs):
        await self._init_lsp_servers()
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
        shutil.rmtree(os.path.join(self.output_dir, 'locks'), ignore_errors=True)

        for idx, files in enumerate(file_orders):
            while True:
                files = self.filter_done_files(files)
                files = self.find_description(files)
                self.construct_file_information(file_relation)
                if not files:
                    break

                if idx == 0:
                    last_batch = 'You are the first batch.'
                    next_batch = '\n'.join(file_orders[idx + 1])
                if idx == len(file_orders) - 1:
                    last_batch = '\n'.join(file_orders[idx - 1])
                    next_batch = 'You are the last batch.'
                else:
                    last_batch = '\n'.join(file_orders[idx - 1])
                    next_batch = '\n'.join(file_orders[idx + 1])

                tasks = [
                    self.write_code(
                        topic,
                        user_story,
                        framework,
                        protocol,
                        name,
                        description,
                        index=idx,
                        last_batch=last_batch,
                        siblings='\n'.join(set(files)-{name}),
                        next_batch=next_batch)
                    for name, description in files.items()
                ]

                # for task in tasks:
                #     await task
                await asyncio.gather(*tasks, return_exceptions=True)

            self.refresh_file_status(file_relation)

        self.construct_file_information(file_relation)
        await self._cleanup_lsp_servers()
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
