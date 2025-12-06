import json
import os

from ms_agent.tools import FileSystemTool
from ms_agent.utils.constants import DEFAULT_INDEX_DIR


class ApiSearch(FileSystemTool):

    def __init__(self, config):
        super().__init__(config)
        mem_config = self.config.memory.code_condenser
        index_dir = getattr(mem_config, 'index_cache_dir', DEFAULT_INDEX_DIR)
        self.index_dir = os.path.join(self.output_dir, index_dir)

    async def _get_tools_inner(self):
        tools = await super()._get_tools_inner()
        file_system = tools['file_system']
        for tool in file_system:
            if 'read_file' == tool['tool_name']:
                tool['description'] += ('\nThis tool will read the index file(containing the abbreviation of the original code file) by default.\n'
                                        'If you want to read the original file, pass `start_line` argument.')
        return tools

    async def read_file(self,
                        paths: list[str],
                        start_line: int = None,
                        end_line: int = None):
        if start_line is None and end_line is None:
            paths = [os.path.join(self.index_dir, path) for path in paths]
        index_results = await super().read_file(paths, start_line, end_line)
        file_contents = json.loads(index_results)
        outputs = {}
        for file, content in file_contents.items():
            file = file[len(self.index_dir) + len(os.sep):]
            if 'FileNotFound' in content:
                origin_content = await super().read_file([file], start_line, end_line)
                outputs.update(json.loads(origin_content))
            else:
                outputs[file] = content
        return json.dumps(outputs, indent=2, ensure_ascii=False)