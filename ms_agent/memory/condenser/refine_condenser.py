import json
from typing import List

from ms_agent.llm import Message, LLM
from ms_agent.memory import Memory


class RefineCondenser(Memory):
    system = """你是一个帮助总结、压缩模型执行历史的机器人。你会被给与模型的历史messages，你需要总结给你的多轮消息，并压缩它们。压缩比例需要达到1:6（30000token压缩到5000token）

你的工作场景是代码编写完成后的修复场景。大模型会不断调用shell等工具，并尝试解决一个大的代码项目中出现的问题。你的工作流程：

1. 你会被给与项目原始需求，技术栈以及文件列表，你需要仔细阅读它们
2. 你会被给与修复历史，其中可能修复了不同问题，也可能在同一个问题上死锁。
    * 对于已经解决的问题，可以保留较少的token或完全移除
    * 保留模型的思路和主要轨迹，保留已经完成了哪些任务的提示，例如创建了文件等
    * 未解决问题可以保留较多的token
    * 多次未解决的死锁问题应增加多次未解决的额外标注
    * 保留最后一个未解决问题的历史记录，并提示模型继续解决该问题
    * 你的优化目标：1. 最少的保留token数量 2. 尽量还原未解决问题概况 3. 尽量保留并总结模型的错误轨迹以备后用
3. 返回你总结好的消息历史，不要增加额外内容（例如“让我来总结...”或“下面是对...的总结...”）

你的优化目标：
1. 【优先】保留充足的信息供后续使用
2. 【其次】保留尽量少的token数量
"""

    def __init__(self, config):
        super().__init__(config)
        self.llm: LLM = LLM.from_config(self.config)
        mem_config = self.config.memory.refine_condenser
        if getattr(mem_config, 'system', None):
            self.system = mem_config.system
        self.threshold = getattr(mem_config, 'threshold', 60000)

    async def condense_memory(self, messages):
        if len(str(messages)) > self.threshold and messages[-1].role in ('user',
                                                                'tool'):
            keep_messages = messages[:2]
            keep_messages_tail = []
            i = 0
            for i, message in enumerate(reversed(messages)):
                keep_messages_tail.append(message)
                if message.role == 'assistant':
                    break

            keep_messages_tail = reversed(keep_messages_tail)
            compress_messages = json.dumps(
                [message.to_dict_clean() for message in messages[2:-i - 1]],
                ensure_ascii=False,
                indent=2)
            keep_messages_json = json.dumps(
                [message.to_dict_clean() for message in keep_messages],
                ensure_ascii=False,
                indent=2)
            keep_messages_tail_json = json.dumps(
                [message.to_dict_clean() for message in keep_messages_tail],
                ensure_ascii=False,
                indent=2)

            query = (f'# 会被保留的消息\n'
                     f'## system和user: {keep_messages_json}\n'
                     f'## 最后的assistant回复: {keep_messages_tail_json}\n'
                     f'# 需要被压缩的消息'
                     f'## 这些消息位于system/user和最后的assistant回复之间:{compress_messages}')

            _messages = [
                Message(role='system', content=self.system),
                Message(role='user', content=query),
            ]
            _response_message = self.llm.generate(_messages, stream=False)
            content = _response_message.content
            keep_messages.append(
                Message(
                    role='user',
                    content=
                    f'Intermediate messages are compressed, here is the compressed message:\n{content}\n'
                ))
            messages = keep_messages + list(keep_messages_tail) + [
                Message(
                    role='user', content='历史消息已经压缩，现在根据历史消息和最后的tool调用继续解决问题：')
            ]
            return messages
        else:
            return messages

    async def run(self, messages: List[Message]):
        return await self.condense_memory(messages)



