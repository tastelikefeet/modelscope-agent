import json
import os
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils.constants import DEFAULT_INDEX_DIR


class ApiSearch(ToolBase):

    def __init__(self, config):
        super().__init__(config)
        mem_config = self.config.memory.code_condenser
        index_dir = getattr(mem_config, 'index_cache_dir', DEFAULT_INDEX_DIR)
        self.index_dir = os.path.join(self.output_dir, index_dir)

    async def connect(self) -> None:
        pass

    async def _get_tools_inner(self) -> Dict[str, Any]:
        tools = {
            'shell': [
                Tool(
                    tool_name='url_search',
                    server_name='api_search',
                    description='Search api definitions with any keywords. These apis are summarized from the code you have written. You need to use this tool when:\n'
                                '1. You are writing a frontend api interface, which needs the exact http definitions\n'
                                '2. You want to check any api problem\n'
                                '3. You want to know if your api definition will duplicate with others\n'
                                'Instructions & Examples:\n'
                                '1. Search user api with `user`\n'
                                '2. Search create music api with music/create\n'
                                '3. Split keywords with `,`\n'
                                '4. If you want all api definitions, pass empty string into `keywords` argument\n'
                                '5. FOLLOW the definitions of this tool results, you should not use any undefined api. Instead, you need to write missing api to a target file.\n',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'keywords': {
                                'type': 'string',
                                'description': 'The keywords in the url to search api of.',
                            }
                        },
                        'required': [],
                        'additionalProperties': False
                    }),
            ]
        }
        return tools

    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict) -> str:
        return await self.url_search(**tool_args)

    async def url_search(self, keywords: str = None):
        if keywords:
            keywords = keywords.split(',')
        def search_in_file(file_path):
            matches = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if 'protocols' not in content:
                        return []
                    for protocol in content['protocols']:
                        if not keywords or any([keyword in protocol['url'] for keyword in keywords]):
                            matches.append(json.dumps(protocol, ensure_ascii=False))
            except Exception: # noqa
                return []
            if matches:
                matches.insert(0, f'API{" with keywords: " + str(keywords) if keywords else ""} defined in {file_path}:')
                matches.append('\n')
            return matches

        files_to_search = []
        for path, _, files in os.walk(self.index_dir):
            for file in files:
                files_to_search.append(os.path.join(path, file))

        # Use thread pool to search files in parallel
        all_matches = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_file = {
                executor.submit(search_in_file, f): f
                for f in files_to_search
            }
            for future in as_completed(future_to_file):
                matches = future.result()
                all_matches.extend(matches)
        return '\n'.join(all_matches)
