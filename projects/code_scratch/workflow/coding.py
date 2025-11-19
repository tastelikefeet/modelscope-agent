import json
import os
from copy import deepcopy
from typing import List

from ms_agent import LLMAgent
from ms_agent.agent import CodeAgent
from ms_agent.llm import Message
from ms_agent.tools import SplitTask
from ms_agent.utils import get_logger

logger = get_logger()


class Programmer(LLMAgent):

    def on_generate_response(self, messages: List[Message]):
        for message in messages:
            if message.role == 'assistant' and message.tool_calls:
                assert len(message.tool_calls) == 1
                if message.tool_calls[0]['tool_name'] == 'file_system---write_file':

                elif message.tool_calls[0]['tool_name'] == 'file_system---read_file':



class CodingAgent(CodeAgent):

    async def execute_code(self, inputs, **kwargs):
        with open(os.path.join(self.output_dir, 'file_design.txt')) as f:
            file_designs = json.load(f)

        file_status = {}
        for file_design in file_designs:
            files = file_design['files']
            for file in files:
                file_status[file['name']] = False

        _config = deepcopy(self.config)
        _config.save_history = False
        _config.load_cache = False
        split_task = SplitTask(_config)

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
                args = {
                    'tasks': [{
                        'system': self.config.prompt.system,
                        'query': f'文件列表:{self.construct_file_information(file_status)}, '
                                 f'你需要编写的文件: {name}, 描述: {description}'
                    }]
                }
                await split_task.call_tool('', tool_name='', tool_args=args)
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