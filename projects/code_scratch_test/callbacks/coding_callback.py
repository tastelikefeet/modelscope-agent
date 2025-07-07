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

    async def on_tool_call(self, runtime: Runtime, messages: List[Message]):
        if not messages[-1].tool_calls or messages[-1].tool_calls[0][
                'tool_name'] != 'split_to_sub_task':
            return
        assert messages[0].role == 'system'
        arguments = messages[-1].tool_calls[0]['arguments']
        arguments = json.loads(arguments)
        tasks = arguments['tasks']
        if isinstance(tasks, str):
            tasks = json.loads(tasks)
        for task in tasks:
            task['_system'] = task['system']
            task['system'] = f"""{task["system"]}

The PRD of this project:

{messages[2].content}

Strictly follow the steps:

1. Before writing each file, list the imports you think of and read the implementations first, to compatible with other code files.

```
The A file depends on the B and C file, and D on the css format, I should read them:
```

If any dependencies do not exist, create them.

2. Output your code with this format:

```js:js/index.js
... code ...
```
The `js/index.js` will be used to saving.

Now Begin:
""" # noqa
        messages[-1].tool_calls[0]['arguments'] = json.dumps({'tasks': tasks})

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        if not messages[-2].tool_calls or messages[-2].tool_calls[0][
                'tool_name'] != 'split_to_sub_task':
            return
        assert messages[0].role == 'system'
        arguments = messages[-2].tool_calls[0]['arguments']
        arguments = json.loads(arguments)
        tasks = arguments['tasks']
        if isinstance(tasks, str):
            tasks = json.loads(tasks)
        for task in tasks:
            task['system'] = task['_system']
            task.pop('_system')
        messages[-2].tool_calls[0]['arguments'] = json.dumps({'tasks': tasks})

        all_files, _ = extract_code_blocks(messages[-1].content, target_filename='summary.txt')
        content = '\n\n'.join([file['code'] for file in all_files])
        messages[2].content += content
