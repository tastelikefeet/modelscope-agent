import json
import os
from copy import deepcopy

from ms_agent.agent import CodeAgent
from ms_agent.tools import SplitTask
from ms_agent.utils import get_logger

logger = get_logger()


class CodingAgent(CodeAgent):

    async def execute_code(self, inputs, **kwargs):
        with open(os.path.join(self.output_dir, 'file_design.txt')) as f:
            file_designs = json.load(f)

        file_status = {}
        all_files = []
        for file_design in file_designs:
            files = file_design['files']
            for file in files:
                file_status[file['name']] = False
                all_files.append(file['name'])
        all_files = '\n'.join(all_files)

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
                _config = deepcopy(self.config)
                _config.save_history = False
                _config.load_cache = False
                split_task = SplitTask(_config)

                args = {
                    'tasks': [{
                        'system': self.config.prompt.system,
                        'query': f'文件列表:{all_files}, 你需要编写的文件: {name}, 描述: {description}'
                    }]
                }
                await split_task.call_tool('', tool_name='', tool_args=args)
                file_status[name] = True

    async def refresh_file_status(self):
        with open(os.path.join(self.output_dir, 'file_design.txt')) as f:
            file_designs = json.load(f)

