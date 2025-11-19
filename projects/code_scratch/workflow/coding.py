import json
import os
from copy import deepcopy

from ms_agent import LLMAgent
from ms_agent.agent import CodeAgent
from ms_agent.llm import Message
from ms_agent.memory.mem0ai import Mem0Memory
from ms_agent.utils import get_logger

logger = get_logger()


class Programmer(LLMAgent):

    async def condense_memory(self, messages):
        if not getattr(self, '_memory_fetched', False):
            for memory_tool in self.memory_tools:
                messages = await memory_tool.run(messages)
            self._memory_fetched = True
        return messages

    def save_memory(self, messages):
        new_messages = []
        for idx, message in enumerate(messages):
            if message.role == 'assistant' and message.tool_calls:
                if message.tool_calls[0]['tool_name'] == 'file_system---write_file':
                    arguments = message.tool_calls[0]['arguments']
                    arguments = json.loads(arguments)
                    if not arguments.get('abbreviation', False) and not arguments['path'].startswith('abbr'):
                        new_messages.append(message)
                        new_messages.append(messages[idx+1])

        if new_messages:
            for memory_tool in self.memory_tools:
                if isinstance(memory_tool, Mem0Memory):
                    memory_tool.add_memories_from_procedural(
                        new_messages, self.get_user_id(), self.tag,
                        'procedural_memory')


class CodingAgent(CodeAgent):

    async def execute_code(self, inputs, **kwargs):
        with open(os.path.join(self.output_dir, 'file_design.txt')) as f:
            file_designs = json.load(f)

        file_status = self.refresh_file_status()
        _config = deepcopy(self.config)
        _config.save_history = False
        _config.load_cache = False

        with open(os.path.join(self.output_dir, 'topic.txt')) as f:
            topic = f.read()
        with open(os.path.join(self.output_dir, 'user_story.txt')) as f:
            user_story = f.read()
        with open(os.path.join(self.output_dir, 'framework.txt')) as f:
            framework = f.read()
        with open(os.path.join(self.output_dir, 'protocol.txt')) as f:
            protocol = f.read()
        index = 0
        for file_design in file_designs:
            files = file_design['files']
            for file in files:
                name = file['name']
                if file_status[name]:
                    continue

                file = None
                for file_design in file_designs:
                    for file in file_design['files']:
                        if file['name'] == name:
                            break
                    if file['name'] == name:
                        break

                logger.info(f'Writing {name}')
                description = file['description']
                messages = [
                    Message(role='system', content=self.config.prompt.system),
                    Message(role='user', content=f'原始需求(topic.txt): {topic}\n'
                                                 f'LLM规划的用户故事(user_story.txt): {user_story}\n'
                                                 f'技术栈(framework.txt): {framework}\n'
                                                 f'通讯协议(protocol.txt): {protocol}\n'
                                                 f'文件列表:{self.construct_file_information(file_status)}\n'
                                                 f'你需要编写的文件: {name}\n文件描述: {description}\n'),
                ]
                programmer = Programmer(_config, tag=f'programmer-{index+1}', trust_remote_code=True)
                await programmer.run(messages)
                index += 1
                file_status = self.refresh_file_status()

    def refresh_file_status(self):
        with open(os.path.join(self.output_dir, 'file_design.txt')) as f:
            file_designs = json.load(f)
        
        file_status = {}
        for file_design in file_designs:
            files = file_design['files']
            for file in files:
                file_name = file['name']
                file_path = os.path.join(self.output_dir, file_name)
                file_status[file_name] = os.path.exists(file_path)
        
        return file_status

    def construct_file_information(self, file_status):
        file_info = ''
        for file, status in file_status.items():
            if status:
                file += f'{file}: ✅已构建\n'
            else:
                file += f'{file}: ❌未构建\n'
        return file_info