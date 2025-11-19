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
        deps_file = os.path.join(self.output_dir, 'file_deps.txt')
        for file_design in file_designs:
            files = file_design['files']
            for file in files:
                file_status[file['name']] = False
                all_files.append(file['name'])
        all_files = '\n'.join(all_files)

        async def write_file(filename):
            file = None
            for file_design in file_designs:
                for file in file_design['files']:
                    if file['name'] == filename:
                        break
                if file['name'] == filename:
                    break

            logger.info(f'Writing {filename}')
            description = file['description']
            _config = deepcopy(self.config)
            _config.save_history = False
            _config.load_cache = False
            split_task = SplitTask(_config)

            args = {
                'tasks': [{
                    'system': self.config.prompt.system,
                    'query': f'文件列表:{all_files}, 你需要编写的文件: {filename}, 描述: {description}'
                }]
            }
            try:
                await split_task.call_tool('', tool_name='', tool_args=args)
                file_status[name] = True
                return True
            except FileNotFoundError as e:
                missing = str(e)
                with open(os.path.join(self.output_dir, 'file_deps.txt'), 'a') as f:
                    f.write(missing + ',' + filename + '\n')
                logger.info(f'Missing dep file: {missing}')
                return False

        while True:
            if os.path.exists(deps_file):
                with open(deps_file) as f:
                    file_deps = f.readlines()
                    file_deps = [dep.strip() for dep in file_deps if dep.strip()]
                if file_deps:
                    missing, _ = file_deps[-1].split(',')
                    await write_file(missing)
                    continue

            for file_design in file_designs:
                files = file_design['files']
                for file in files:
                    name = file['name']
                    if file_status[name]:
                        continue

                    success = await write_file(name)
                    if not success:
                        break


