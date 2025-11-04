from copy import deepcopy

from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.llm.openai_llm import OpenAI


class GeneratePrompt(CodeAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm: OpenAI = LLM.from_config(self.config)

    async def run(self, inputs, **kwargs):
        _response_message = self.llm.generate(deepcopy(inputs))
        response = _response_message.content
        if '```python' in response:
            manim_code = response.split('```python')[1].split('```')[0]
        elif '```' in response:
            manim_code = response.split('```')[1].split('```')[0]
        else:
            manim_code = response

        inputs.append(Message(role='assistant', content=manim_code))
        return inputs

