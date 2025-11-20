import os
from typing import List, OrderedDict

from ms_agent import LLMAgent
from ms_agent.llm import Message
from coding import CodingAgent


class RefineAgent(LLMAgent):

    async def run(self, messages, **kwargs):
        with open(os.path.join(self.output_dir, 'topic.txt')) as f:
            topic = f.read()
        with open(os.path.join(self.output_dir, 'user_story.txt')) as f:
            user_story = f.read()
        with open(os.path.join(self.output_dir, 'framework.txt')) as f:
            framework = f.read()
        with open(os.path.join(self.output_dir, 'protocol.txt')) as f:
            protocol = f.read()

        file_relation = OrderedDict()
        CodingAgent.refresh_file_status(self, file_relation)
        file_info = CodingAgent.construct_file_information(self, file_relation, True)
        messages = [
            Message(role='system', content=self.config.prompt.system),
            Message(role='user', content=f'原始需求(topic.txt): {topic}\n'
                                         f'LLM规划的用户故事(user_story.txt): {user_story}\n'
                                         f'技术栈(framework.txt): {framework}\n'
                                         f'通讯协议(protocol.txt): {protocol}\n'
                                         f'文件列表:{file_info}\n'
                                         f'你的shell工具的work_dir是{self.output_dir}\n'
                                         f'请针对项目进行refine:'),
        ]
        return await super().run(messages, **kwargs)

    async def on_task_end(self, messages: List[Message]):
        assert os.path.isfile(os.path.join(self.output_dir, 'framework.txt'))
        assert os.path.isfile(os.path.join(self.output_dir, 'protocol.txt'))
        assert os.path.isfile(os.path.join(self.output_dir, 'modules.txt'))
