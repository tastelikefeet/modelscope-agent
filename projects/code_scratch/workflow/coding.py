import asyncio
import dataclasses
import json
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Set

from omegaconf import DictConfig

from ms_agent import LLMAgent
from ms_agent.agent import CodeAgent
from ms_agent.llm import Message
from ms_agent.utils import get_logger

logger = get_logger()


class Programmer(LLMAgent):

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

    _fast_fail = """
6. 如果你发现依赖的任一底层代码文件不存在，你不应当继续编写代码文件，而应当调用**missing_dependency工具**汇报缺失文件
"""
    _continue = """
6. 如果你发现依赖的任一底层代码文件不存在，你应当创建这个代码文件和对应的缩略文件
"""

    _find_deps = """你是一个优秀的软件项目架构师。你的职责是根据原始需求和模块划分以及具体的文件，找到其上游需要依赖的其他代码文件。你的工作流程如下：
    
1. 用户原始需求和用户故事已经放入上下文，你无需再读取。这些知识包括：
  * topic.txt：原始需求
  * user_story.txt：用户故事
  * protocol.txt：通讯协议
  * framework.txt：技术选型
  * tasks.txt: 项目文件列表
2. 列出在本项目中需要依赖的其他文件，并和tasks.txt的内容进行比对，
  * **确认依赖文件在文件列表内，不要使用未在文件列表内定义的代码文件**
  * 你只能依赖file_order.txt中index小于你的文件，等于你的文件会和本文件一起编写，大于你的会在后续编写
  你的输出例子：
  为完成xxx代码，根据通讯协议和文件列表分析，我需要和 ... 进行http通讯，为完成user_story的设计，我需要使用 ... 的底层服务，综上所述我需要依赖：
  <result>
  xxx
  yyy
  ...
  </result>
  文件依赖放入<result></result>中，以列(\n)分隔。如果没有依赖文件，返回空的<result></result>
  注意：你不需要编写任何具体的代码文件
"""

    async def find_deps(self, topic, user_story, framework, protocol,
                         name, description):
        _config = deepcopy(self.config)
        _config.tools = DictConfig({})
        _config.prompt = None
        _config.memory = None
        with open(os.path.join(self.output_dir, 'file_order.txt')) as f:
            file_order = f.read()
        messages = [
            Message(role='system', content=self._find_deps),
            Message(role='user', content=f'原始需求(topic.txt): {topic}\n'
                                         f'LLM规划的用户故事(user_story.txt): {user_story}\n'
                                         f'技术栈(framework.txt): {framework}\n'
                                         f'通讯协议(protocol.txt): {protocol}\n'
                                         f'文件编写列表(file_order.txt): {file_order}\n'
                                         f'The file in one index will be written parallelly.\n'
                                         f'你负责查找依赖的文件: {name}\n文件描述: {description}\n'),
        ]
        _config.save_history = False
        _config.load_cache = False
        deps = LLMAgent(_config, tag=f'deps-{name}', trust_remote_code=True)
        messages = await deps.run(messages)
        all_deps = []
        pattern = r'<result>(.*?)</result>'
        for deps in re.findall(pattern, messages[-1].content, re.DOTALL):
            all_deps.extend(deps.split('\n'))
        all_deps = [dep.strip() for dep in all_deps if dep.strip()]

        all_file_deps = ''
        for dep in all_deps:
            abbr_dep = os.path.join(self.output_dir, 'abbr', dep)
            if os.path.exists(abbr_dep):
                with open(abbr_dep, 'r') as f:
                    all_file_deps += f'The abbreviation content of {dep}: {f.read()}\n'
            elif os.path.exists(os.path.join(self.output_dir, dep)):
                with open(os.path.join(self.output_dir, dep), 'r') as f:
                    all_file_deps += f'The content of {dep}: {f.read()}\n'
            else:
                all_file_deps += f'A file named: {dep} you need may not exists.\n'
        if all_file_deps:
            return f'一些文件内容: {all_file_deps}\n'
        else:
            return ''


    async def write_code(self, topic, user_story, framework, protocol,
                         name, description, fast_fail):
        logger.info(f'Writing {name}')
        _config = deepcopy(self.config)
        if fast_fail:
            system = self.config.prompt.system + self._fast_fail
        else:
            _config.tools.plugins.pop(-1)
            system = self.config.prompt.system + self._continue
        all_file_deps = await self.find_deps(topic, user_story, framework, protocol,name,description)
        messages = [
            Message(role='system', content=system),
            Message(role='user', content=f'原始需求(topic.txt): {topic}\n'
                                         f'LLM规划的用户故事(user_story.txt): {user_story}\n'
                                         f'技术栈(framework.txt): {framework}\n'
                                         f'通讯协议(protocol.txt): {protocol}\n'
                                         f'{all_file_deps}'
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