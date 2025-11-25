import asyncio
import dataclasses
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Set, List

from omegaconf import DictConfig

from ms_agent import LLMAgent
from ms_agent.agent import CodeAgent, Agent
from ms_agent.llm import Message
from ms_agent.llm.openai_llm import OpenAI
from ms_agent.utils import get_logger, async_retry
from ms_agent.utils.constants import DEFAULT_TAG
from .utils import stop_words, parse_imports

logger = get_logger()


class Programmer(LLMAgent):

    def __init__(self,
                 config: DictConfig = DictConfig({}),
                 tag: str = DEFAULT_TAG,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.stop = stop_words

    async def on_generate_response(self, messages: List[Message]):
        if self.llm.args.get('stop', None) != self.stop:
            self.llm.args['stop'] = self.stop
        else:
            self.llm.args['stop'] = '```'

    def generate_abbr_file(self, file):
        system = """你是一个帮我简化代码并返回缩略的机器人。你缩略的文件会给与另一个LLM用来编写代码，因此你缩略的代码文件需要具有充足的其他文件依赖的信息。

需要保留的信息：
1. 类名、方法名、方法参数类型，返回值类型
2. imports依赖
3. exports导出及导出类型
4. 如果是结构定义代码，保留结构和字段等充分信息
5. 如果是css样式代码，保留样式名称
6. 如果是json，保留结构即可
7. 
"""

    async def on_tool_call(self, messages: List[Message]):
        await super().on_tool_call(messages)
        if '```' not in messages[-1].content:
            return

        code_file = messages[-1].content.split('```')[1].split(':')[1].split('\n')[0].strip()
        all_files = parse_imports(code_file, messages[-1].content) or []
        deps_not_exist = False
        missing_deps = []
        deps = []
        for file in all_files:
            if not os.path.exists(file):
                deps_not_exist = True
                missing_deps.append(file)
            else:
                deps.append(file)

        if deps_not_exist:
            messages = messages[:-1]
            messages.append(Message(role='user', content=f'Some dependencies are missing: {missing_deps}, create them first:'))
        else:



    @async_retry(max_attempts=Agent.retry_count, delay=1.0)
    async def step(
            self, messages: List[Message]
    ) -> AsyncGenerator[List[Message], Any]:  # type: ignore
        messages = deepcopy(messages)
        if messages[-1].role != 'assistant':
            messages = await self.condense_memory(messages)
            await self.on_generate_response(messages)
            tools = await self.tool_manager.get_tools()

            _response_message = self.llm.generate(messages, tools=tools)
            if _response_message.content:
                self.log_output('[assistant]:')
                self.log_output(_response_message.content)

            # Response generated
            self.handle_new_response(messages, _response_message)
            await self.on_tool_call(messages)
        else:
            _response_message = messages[-1]
        self.save_history(messages)

        if _response_message.tool_calls:
            messages = await self.parallel_tool_call(messages)
        else:
            self.runtime.should_stop = True

        await self.after_tool_call(messages)
        self.log_output(
            f'[usage] prompt_tokens: {_response_message.prompt_tokens}, '
            f'completion_tokens: {_response_message.completion_tokens}')
        yield messages

    async def condense_memory(self, messages):
        return messages
        if not getattr(self, '_memory_fetched', False):
            for memory_tool in self.memory_tools:
                messages = await memory_tool.run(messages)
            self._memory_fetched = True
        return messages

    async def add_memory(self, messages, **kwargs):
        return
        if not self.runtime.should_stop:
            return
        all_written_files = []
        for idx, message in enumerate(messages):
            if message.role == 'assistant' and message.tool_calls:
                if message.tool_calls[0]['tool_name'] == 'file_system---write_file':
                    arguments = message.tool_calls[0]['arguments']
                    arguments = json.loads(arguments)
                    if not arguments.get('abbreviation', False) and not arguments['path'].startswith('abbr'):
                        all_written_files.append(arguments['content'])

        if all_written_files:
            for file_content in all_written_files:
                file_len = len(file_content)
                chunk_size = 2048
                overlap = 256
                chunks = []

                if file_len <= chunk_size:
                    chunks.append(file_content)
                else:
                    start = 0
                    while start < file_len:
                        end = min(start + chunk_size, file_len)
                        chunks.append(file_content[start:end])
                        if end >= file_len:
                            break
                        start += (chunk_size - overlap)

                # Add each chunk to memory
                for chunk in chunks:
                    _messages = [Message(role='assistant', content=chunk)]
                    await super().add_memory(_messages, **kwargs)
        

@dataclasses.dataclass
class FileRelation:

    name: str
    description: str
    done: bool = False
    deps: Set[str] = dataclasses.field(default_factory=set)


class CodingAgent(CodeAgent):

    async def write_code(self, topic, user_story, framework, protocol,
                         name, description, fast_fail):
        logger.info(f'Writing {name}')
        _config = deepcopy(self.config)
        messages = [
            Message(role='system', content=self.config.system),
            Message(role='user', content=f'原始需求(topic.txt): {topic}\n'
                                         f'LLM规划的用户故事(user_story.txt): {user_story}\n'
                                         f'技术栈(framework.txt): {framework}\n'
                                         f'通讯协议(protocol.txt): {protocol}\n'
                                         f'你需要编写的文件: {name}\n文件描述: {description}\n'),
        ]

        _config = deepcopy(self.config)
        _config.save_history = True
        _config.load_cache = False
        programmer = Programmer(_config, tag=f'programmer-{name.replace(os.sep, "-")}', trust_remote_code=True)
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

        # Use ThreadPoolExecutor for IO-intensive LLM API calls
        max_workers = 4  # Optimal for IO-intensive tasks
        
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
                            self.write_code(topic, user_story, framework, protocol, name, description, fast_fail=False)
                        )
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

    @staticmethod
    def get_next_file(file_relation):

        def get_parent(parent, loops):
            for _dep in file_relation[parent].deps:
                if file_relation[_dep].done:
                    continue
                if _dep in loops:
                    return loops[-1], False
                loops.append(_dep)
                return get_parent(_dep, loops)

            return parent, True

        for file in file_relation.values():
            file: FileRelation
            if file.done:
                continue

            loops = [file.name]
            return get_parent(file.name, loops)
        return None, True

    def get_missing_files(self):
        if os.path.exists(os.path.join(self.output_dir, 'missing.txt')):
            with open(os.path.join(self.output_dir, 'missing.txt')) as f:
                missing_files = f.readlines()
                missing_files = [file.strip() for file in missing_files if file.strip()]
            os.remove(os.path.join(self.output_dir, 'missing.txt'))
            assert not os.path.exists(os.path.join(self.output_dir, 'missing.txt'))
            return missing_files
        else:
            return []

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
                    file_relation[file_name] = FileRelation(name=file_name, description=description)
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