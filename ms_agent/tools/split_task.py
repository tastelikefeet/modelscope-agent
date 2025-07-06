# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio

from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils.utils import escape_yaml_string
from omegaconf import DictConfig

from projects.code_scratch.callbacks.file_parser import extract_code_blocks


class SplitTask(ToolBase):
    """A tool special for task splitting"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        if hasattr(config, 'tools') and hasattr(config.tools, 'split_task'):
            self.tag_prefix = getattr(config.tools.split_task, 'tag_prefix',
                                      'worker-')
        else:
            self.tag_prefix = 'worker-'
        self.round = 0

    async def connect(self):
        pass

    async def cleanup(self):
        pass

    async def get_tools(self):
        return {
            'split_task': [
                Tool(
                    tool_name='split_to_sub_task',
                    server_name='split_task',
                    description=
                    'Split complex task into sub tasks and start them, for example, '
                    'split a website generation task into sub tasks, '
                    'you plan the framework, include code files and classes and functions, and give the detail '
                    'information to the system and query field of the subtask, then '
                    'let each subtask to write a single file',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'tasks': {
                                'type':
                                'array',
                                'description':
                                'MANDATORY: Each element is a dict, which must contains two fields: '
                                '`system`(str) and `query`(str) to start one sub task.'
                            }
                        },
                        'required': ['tasks'],
                        'additionalProperties': False
                    })
            ]
        }

    async def call_tool(self, server_name: str, *, tool_name: str,
                        tool_args: dict):
        """
        1. LLMAgent will be used to start subtask
        2. config will be inherited from the parent task
        """
        from ms_agent.agent import LLMAgent
        tasks = tool_args.get('tasks')
        results = []
        all_contents = []
        for i, task in enumerate(tasks):
            system = task['system']
            query = task['query']
            config = DictConfig(self.config)
            config.prompt.system = escape_yaml_string(system)
            # config.prompt.system += '\n\n'.join(all_contents)
            trust_remote_code = getattr(config, 'trust_remote_code', False)
            agent = LLMAgent(
                config=config,
                trust_remote_code=trust_remote_code,
                tag=f'{self.config.tag}-r{self.round}-{self.tag_prefix}{i}',
                task='subtask')
            messages = await agent.run(query)
            results.append(messages[-1].content)
            # all_files, _ = extract_code_blocks(messages[-1].content, target_filename='summary.txt')
            # all_contents.extend([file['code'] for file in all_files])

        self.round += 1
        result = ''
        for i in range(len(results)):
            result += f'SplitTask{i}:{results[i]}\n'
        return result
