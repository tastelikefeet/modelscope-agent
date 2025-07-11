# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List
from ms_agent.agent.runtime import Runtime
from ms_agent.callbacks import Callback
from ms_agent.llm.utils import Message
from ms_agent.tools.filesystem_tool import FileSystemTool
from ms_agent.utils import get_logger
from omegaconf import DictConfig

from ms_agent.utils.utils import extract_blocks

logger = get_logger()


class ArtifactCallback(Callback):
    """Save the artifacts to disk.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    async def on_task_end(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.cleanup()

    async def after_generate_response(self, runtime: Runtime,
                                      messages: List[Message]):
        if messages[-1].role != 'assistant':
            return
        await self.file_system.create_directory()
        content =messages[-1].content
        all_files, _ = extract_blocks(content)
        results = []
        for f in all_files:
            result = await self.file_system.write_file(
                f['filename'], f['code'])
            results.append(result)

        r = '\n'.join(results)
        if len(r) > 0:
            messages.append(Message(role='user', content=r))
