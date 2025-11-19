import json
from typing import List

from ms_agent import LLMAgent
from ms_agent.llm import Message


class CodingAgent(LLMAgent):

    async def on_tool_call(self, messages):
        # tool name is not 'split_to_sub_task', ut is 'SplitTask---split_to_sub_task'
        if not messages[-1].tool_calls or 'split_to_sub_task' not in messages[
            -1].tool_calls[0]['tool_name']:
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



    Now Begin:
    """  # noqa
        messages[-1].tool_calls[0]['arguments'] = json.dumps({'tasks': tasks})
