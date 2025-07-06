# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
from typing import List

from file_parser import extract_code_blocks
from ms_agent.agent.runtime import Runtime
from ms_agent.callbacks import Callback
from ms_agent.llm.utils import Message
from ms_agent.tools.filesystem_tool import FileSystemTool
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


class ArtifactCallback(Callback):
    """Save the output code to local disk.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    async def summarize(self, llm, file_content: str):

        system = """You are a assistant help me to summarize a code file.
You must summarize the code file to a short and brief summary.

If it's a code file:

1. All imports
2. The class names
3. The function names
4. The args/types/returns of the function
5. All exports

You do not need to give explanations for functions or arguments.

If the file is a project file, like package.json:

1. output the coding language, dependencies and requirements information

If the file is a json file:

1. only output the data structure

Later, the information will be used to guide other coding tasks.
**Do not output any other unrelated information, ONLY OUTPUT the summary itself, keep it short and brief.**

Now begin:
"""
        messages = [
            Message(role='system', content=system),
            Message(role='user', content=f'The code file to be summarized: {file_content}'),
        ]
        _content = ''
        for _response_message in llm.generate(messages, stream=True):
            new_content = _response_message.content[len(_content):]
            sys.stdout.write(new_content)
            sys.stdout.flush()
            _content = _response_message.content
        return _response_message.content

    async def after_generate_response(self, runtime: Runtime,
                                      messages: List[Message]):
        if messages[-1].tool_calls or messages[-1].role == 'tool':
            return
        await self.file_system.create_directory()
        content = messages[-1].content
        all_files, left_content = extract_code_blocks(content)
        results = []
        summaries = []
        for f in all_files:
            if not f['filename'].startswith(
                    'frontend') and not f['filename'].startswith('backend') and f['filename'] not in ('files.json'):
                results.append(
                    f'Error: You should generate files in frontend or backend, '
                    f'but now is: {f["filename"]}')
            else:
                result = await self.file_system.write_file(f['filename'], f['code'])
                results.append(result)

            if f['filename'] != 'files.json':
                summary = await self.summarize(runtime.llm, f['code'])
                summaries.append(f'You have implemented the code file: {f["filename"]}, '
                                 f'The code file has been moved to file because the content size,'
                                 f'Its interface is:\n{summary}')

        r = '\n\n'.join(summaries)
        if len(r):
            messages[-1].content = left_content + r
