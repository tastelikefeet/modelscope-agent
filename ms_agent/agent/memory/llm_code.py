# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List

from omegaconf import DictConfig

from ms_agent.agent import Runtime
from ms_agent.agent.memory import Memory
from ms_agent.llm import Message
from ms_agent.utils.utils import extract_blocks


class LLMCodeSummary(Memory):

    _system = f"""You are an assistant helps me summarize code files. You must follow these instructions:

1. A few code files are given to you with the format:

```js: a.js
... code ...
```

```json: package.json
... json ...
```

2. You need to give summary according to the types of the code files.
    * If it's a normal code file, output the most brief information which can represent the interfaces of the class/functions:
    
    An example:
    ```js: a.js
    imports:
        - xx from b.js
        - yy from c.js
    definitions:
        class A
            - function foo
                args:
                    - x: type int
                    - y: type str
                returns:
                    z: type str
    exports:
        A
    ```
    Do not give descriptions, we need a most condensed summary.
    
    * If it's a json file, or a project file, output the brief information of versions and dependencies:
    
    An example:
    ```json: package.json
    main dependencies:
        - package a
        - package b
        - package c
    ... other information here ...
    ```
    The output format of blocks should be the same with the input format of code blocks:
    ```json: package.json
    ... summaries ...
    ```
    
3. You must give the same number of summaries with the input code blocks, no more no less.

4. Do not output unrelated information, like `I thought`, `It seems`, `Let me finish`, etc. Only output the summaries.

Now begin:
"""

    def __init__(self, config: DictConfig):
        super().__init__(config)

    async def run(self, runtime: Runtime, messages: List[Message]) -> List[Message]:
        code_blocks, remaining = extract_blocks(messages[-1].content)
        code_blocks_len = len(code_blocks)
        file_names = [f['filename'] for f in code_blocks]
        query = (f'Here are {code_blocks_len} code blocks:\n'
                 f'{messages[-1].content}\n'
                 f'You must output {code_blocks_len} summaries:\n')

        llm = runtime.llm
        messages = [
            Message(role='system', content=self._system),
            Message(role='user', content=query),
        ]
        response = llm.generate(messages)
        code_blocks, _ = extract_blocks(response.content)
        summaries = ''
        for block in code_blocks:
            summaries += (f'# {block["filename"]}\n'
                          f'```\n{block["code"]}```\n\n')

        new_content = f"""You generated {code_blocks_len} code files:

{file_names}

These actual code has been saved to the local disk. Here are the summaries:

{summaries}

"""
        messages[-1].content = remaining + new_content
