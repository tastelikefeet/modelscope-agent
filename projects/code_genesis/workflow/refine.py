import os
from typing import List, OrderedDict
import sys
import json

from omegaconf import DictConfig

from coding import CodingAgent
from ms_agent import LLMAgent
from ms_agent.llm import Message
from ms_agent.memory.condenser.refine_condenser import RefineCondenser
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_TAG

logger = get_logger()


class RefineAgent(LLMAgent):

    def __init__(self,
                 config: DictConfig = DictConfig({}),
                 tag: str = DEFAULT_TAG,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.refine_condenser = RefineCondenser(config)

    async def condense_memory(self, messages):
        return await self.refine_condenser.run([m for m in messages])

    async def run(self, messages, **kwargs):
        with open(os.path.join(self.output_dir, 'topic.txt')) as f:
            topic = f.read()
        with open(os.path.join(self.output_dir, 'framework.txt')) as f:
            framework = f.read()
        with open(os.path.join(self.output_dir, 'protocol.txt')) as f:
            protocol = f.read()
        with open(os.path.join(self.output_dir, 'tasks.txt')) as f:
            file_info = f.read()

        file_relation = OrderedDict()
        CodingAgent.refresh_file_status(self, file_relation)
        CodingAgent.construct_file_information(self, file_relation, True)
        messages = [
            Message(role='system', content=self.config.prompt.system),
            Message(
                role='user',
                content=f'原始需求(topic.txt): {topic}\n'
                        f'技术栈(framework.txt): {framework}\n'
                        f'通讯协议(protocol.txt): {protocol}\n'
                        f'文件列表:{file_info}\n'
                        f'你的shell工具的work_dir（项目输出文件）是{self.output_dir}\n'
                        f'python环境是: {sys.executable}\n'
                        f'请针对项目进行refine:'),
        ]
        return await super().run(messages, **kwargs)

    async def after_tool_call(self, messages: List[Message]):
        has_tool_call = len(messages[-1].tool_calls) > 0
        if not has_tool_call:
            query = input('>>>')
            messages.append(Message(role='user', content=query))
