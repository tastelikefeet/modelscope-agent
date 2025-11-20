import os

from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR


class MissingDependencyTool(ToolBase):

    def __init__(self, config, **kwargs):
        super(MissingDependencyTool, self).__init__(config)
        self.exclude_func(getattr(config.tools, 'file_system', None))
        self.output_dir = getattr(config, 'output_dir', DEFAULT_OUTPUT_DIR)

    async def connect(self) -> None:
        pass

    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict):
        missing_files = tool_args["missing_files"]
        with open(os.path.join(self.output_dir, 'missing.txt'), "r") as f:
            f.writelines(missing_files)
        return f'Missing dependencies {missing_files} reported, Now you can quit the task.'


    async def _get_tools_inner(self):
        return {
            'missing_dependency': [
                Tool(
                    tool_name='report',
                    server_name='missing_dependency',
                    description=
                    'Report issue that the file cannot be created because the dependency files are missing',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'missing_files': {
                                'type':
                                    'array',
                                'description':
                                    'The missing dependency file name list',
                            }
                        },
                        'required': ['missing_files'],
                        'additionalProperties': False
                    })
            ]
        }
