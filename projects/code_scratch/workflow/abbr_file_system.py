import json
import os
from typing import List

from ms_agent.tools import FileSystemTool


class AbbrFileSystemTool(FileSystemTool):

    async def _get_tools_inner(self):
        tools = await super()._get_tools_inner()
        tool_list = tools['file_system']
        for tool in tool_list:
            if tool.tool_name in ('write_file', 'read_file'):
                tool.parameters['properties']['abbreviation'] = {
                    'type': 'string',
                    'description': 'Read the abbreviation content of file'
                }
        return tools

    def missing_file(self, missing):
        with open(os.path.join(self.output_dir, 'file_design.txt'), 'w') as f:
            file_designs = json.load(f)

        all_files = []
        for file_design in file_designs:
            for file in file_design['files']:
                all_files.append(file['name'])

        if missing not in all_files:
            return False

        with open(os.path.join(self.output_dir, 'file_deps.txt'), 'r') as f:
            deps = f.readlines()

        for dep in deps:
            _, file = dep.split(',')
            if missing == file:
                return False
        return True

    async def read_file(self, paths: List[str], abbreviation='0'):
        if abbreviation == '1':
            abbr_paths = [os.path.join('abbr', path) for path in paths]
        else:
            abbr_paths = [None] * len(paths)

        file_contents = {}
        for abbr_path, path in zip(abbr_paths, paths):
            content = await super().read_file([abbr_path])
            if 'FileNotFound' in content:
                content = await super().read_file([path])
            if 'FileNotFound' in content:
                if self.missing_file(path):
                    raise FileNotFoundError(path)
            content = json.loads(content)
            file_contents.update(content)

        return json.dumps(file_contents, indent=2, ensure_ascii=False)

    async def write_file(self, path: str, content: str, abbreviation='0'):
        if abbreviation == '1':
            path = os.path.join('abbr', path)

        return super().write_file(path, content)


