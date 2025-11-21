import asyncio
import dataclasses
import json
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Set

from ms_agent import LLMAgent
from ms_agent.agent import CodeAgent
from ms_agent.llm import Message
from ms_agent.utils import get_logger

logger = get_logger()


class Programmer(LLMAgent):

    async def condense_memory(self, messages):
        if not getattr(self, '_memory_fetched', False):
            for memory_tool in self.memory_tools:
                messages = await memory_tool.run(messages)
            self._memory_fetched = True
        return messages

    async def add_memory(self, messages, **kwargs):
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

    _fast_fail = """
6. 如果你发现依赖的任一底层代码文件不存在，你不应当继续编写代码文件，而应当调用**missing_dependency工具**汇报缺失文件
"""
    _continue = """
6. 如果你发现依赖的任一底层代码文件不存在，你应当创建这个代码文件和对应的缩略文件
"""
    async def write_code(self, topic, user_story, framework, protocol,
                         name, description, fast_fail):
        logger.info(f'Writing {name}')
        _config = deepcopy(self.config)
        if fast_fail:
            system = self.config.prompt.system + self._fast_fail
        else:
            _config.tools.plugins.pop(-1)
            system = self.config.prompt.system + self._continue
        messages = [
            Message(role='system', content=system),
            Message(role='user', content=f'原始需求(topic.txt): {topic}\n'
                                         f'LLM规划的用户故事(user_story.txt): {user_story}\n'
                                         f'技术栈(framework.txt): {framework}\n'
                                         f'通讯协议(protocol.txt): {protocol}\n'
                                         f'你需要编写的文件: {name}\n文件描述: {description}\n'),
        ]

        _config = deepcopy(self.config)
        _config.save_history = False
        _config.load_cache = False
        programmer = Programmer(_config, tag=f'programmer-{name}', trust_remote_code=True)
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

        for files in file_orders:
            while True:
                files = self.filter_done_files(files)
                files = self.find_description(files)
                self.construct_file_information(file_relation)
                if not files:
                    break

                tasks = [
                    self.write_code(topic, user_story, framework, protocol, name, description, fast_fail=False)
                    for name, description in files.items()
                ]
                await asyncio.gather(*tasks)

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