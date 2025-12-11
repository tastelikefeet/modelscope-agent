from ms_agent import LLMAgent
from ms_agent.llm import Message


class InstallAgent(LLMAgent):

    async def run(self, messages, **kwargs):
        query = '请编写依赖文件并安装依赖:'

        messages = [
            Message(role='system', content=self.config.prompt.system),
            Message(role='user', content=query),
        ]
        return await super().run(messages, **kwargs)
