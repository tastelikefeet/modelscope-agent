import asyncio
import dataclasses
import json
import os
import re
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Set

from omegaconf import DictConfig

from ms_agent import LLMAgent
from ms_agent.agent import CodeAgent
from ms_agent.llm import Message
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_TAG
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

    async def on_task_begin(self, messages: List[Message]):
        if 'extra_body' not in self.llm.args:
            self.llm.args['extra_body'] = DictConfig({})
        self.llm.args['extra_body']['stop_sequences'] = stop_words
        self.code_files = [self.code_file]
        self.find_all_files()

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
                            arguments = json.loads(arguments)
                        files.extend(arguments['paths'])
        return set(files)

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
        coding_finish = '<result>' in messages[
            -1].content and '</result>' in messages[-1].content
        import_finish = '<result>' in messages[-1].content and self.llm.args[
            'extra_body'][
                'stop_sequences'] == stop_words and '</result>' not in messages[
                    -1].content and not is_config
        has_tool_call = len(messages[-1].tool_calls
                            or []) > 0 or messages[-1].role != 'assistant'
        if (not has_tool_call) and import_finish:
            contents = messages[-1].content.split('\n')
            content = [c for c in contents if '<result>' in c and ':' in c][0]
            code_file = content.split('<result>')[1].split(':')[1].split(
                '\n')[0].strip()
            all_files = parse_imports(code_file, messages[-1].content,
                                      self.output_dir) or []
            all_read_files = self.find_all_read_files(messages)
            deps = []
            definitions = []
            folders = []
            wrong_imports = []
            for file in all_files:
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
                    content = self.generate_abbr_file(dep)
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
                messages.append(
                    Message(
                        role='user',
                        content=
                        f'We break your generation to import more relative information. '
                        f'According to your imports, some extra contents manually given here:\n'
                        f'\n{dep_content or "No extra dependencies needed"}\n'
                        f'Here is the a few start lines of your code: {content}\n\n'
                        f'Now review your imports in it, correct any error according to the dependencies, '
                        f'if any data structure undefined/not found, you can go on reading any code files you need, '
                        f'then rewrite the full code of {code_file} based on the start lines:\n'
                    ))
                if not wrong_imports:
                    self.llm.args['extra_body']['stop_sequences'] = []
        elif (not has_tool_call) and coding_finish:
            result, remaining_text = extract_code_blocks(messages[-1].content)
            if result:
                saving_result = ''
                for r in result:
                    path = r['filename']
                    code = r['code']
                    path = os.path.join(self.output_dir, path)

                    lock_dir = os.path.join(self.output_dir, 'locks')

                    # Check and write file with lock
                    with file_lock(lock_dir, r['filename']):
                        file_exists = os.path.exists(path)
                        if not file_exists:
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            with open(path, 'w') as f:
                                f.write(code)

                    # Generate abbreviation outside the lock to avoid nested locking
                    abbr_content = self.generate_abbr_file(r['filename'])

                    if file_exists:
                        saving_result += (
                            f'The target file exists, cannot override. here is the file abbreviation '
                            f'content: \n{abbr_content}\n')
                    else:
                        saving_result += (
                            f'Save file <{r["filename"]}> successfully\n. here is the file abbreviation '
                            f'content: \n{abbr_content}\n')
                messages[-1].content = remaining_text + (
                    'Code generated here. Content removed to condense messages, '
                    'check save result for details.')
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

    async def condense_memory(self, messages):
        return messages

    async def add_memory(self, messages, **kwargs):
        return


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
