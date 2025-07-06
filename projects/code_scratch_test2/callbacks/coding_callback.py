# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List

import json
from ms_agent.agent.runtime import Runtime
from ms_agent.callbacks import Callback
from ms_agent.llm.utils import Message
from ms_agent.tools.filesystem_tool import FileSystemTool
from ms_agent.utils import get_logger
from omegaconf import DictConfig

from projects.code_scratch.callbacks.file_parser import extract_code_blocks

logger = get_logger()


class CodingCallback(Callback):
    """Add more prompts when coding
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()
        query = f"""The original query is {messages[1].content}

The PRD  is {messages[2].content}

Now please implement the files one by one:
"""
        system = messages[0]
        message = Message(role='user', content=query)
        messages.clear()
        messages.extend([system, message])

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        files_exist = await self.file_system.list_files()
        files_all = await self.file_system.read_file('files.json')
        try:
            files_all = json.loads(files_all)
        except:
            print()
        left_files = set(files_all) - set(files_exist.split('\n'))
        _messages = messages[:3] + [message for message in messages[3:] if message.role != 'user']
        messages.clear()
        messages.extend(_messages)
        if len(left_files) > 0:
            messages.append(Message(role='user', content=f'Some code files still need to be generated: {left_files}, generate them from lower modules to higher modules:'))
            runtime.should_stop = False
        else:
            runtime.should_stop = True