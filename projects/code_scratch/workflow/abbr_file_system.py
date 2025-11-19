import json
import os
from typing import List

from ms_agent.tools import FileSystemTool


class AbbrFileSystemTool(FileSystemTool):

    async def _get_tools_inner(self):
        tools = await super()._get_tools_inner()
        tool_list = tools['file_system']
        for tool in tool_list:
            if tool['tool_name'] in ('write_file', 'read_file'):
                tool['parameters']['properties']['abbreviation'] = {
                    'type': 'integer',
                    'description': 'Read the abbreviation content of file, can be 0(default, read the original file) or 1(read the abbreviation)'
                }
        return tools

    async def read_file(self, paths: List[str], abbreviation=0):
        if abbreviation:
            abbr_paths = [os.path.join('abbr', path) for path in paths]
        else:
            abbr_paths = [None] * len(paths)

        file_contents = {}
        for abbr_path, path in zip(abbr_paths, paths):
            content = ''
            if abbr_path:
                content = await super().read_file([abbr_path])
            if not content or 'FileNotFound' in content:
                content = await super().read_file([path])
            content = json.loads(content)
            file_contents.update(content)

        return json.dumps(file_contents, indent=2, ensure_ascii=False)

    async def write_file(self, path: str, content: str, abbreviation=0):
        if abbreviation:
            path = os.path.join('abbr', path)

        return super().write_file(path, content)


