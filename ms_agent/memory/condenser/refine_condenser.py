import json
from typing import List

from ms_agent.llm import Message, LLM
from ms_agent.memory import Memory


class RefineCondenser(Memory):
    system = """You are a bot that helps summarize and compress model execution history. You will be given model messages, and you need to summarize and compress them. The compression ratio should reach 1:6 (compressing 30,000 tokens to 5,000 tokens).

Your working scenario is code writing and subsequent debugging. The large language model will continuously call tools like shell to write or solve problems in complex code projects. Your workflow:

1. The model's conversation history may have fixed different issues, or may be deadlocked on the same issue
    * Retain the model's thought process and main trajectory for completing tasks, such as creating files, fixing issues, viewing key information from documentation, etc.
    * For issues that have been resolved, you can retain fewer tokens or remove them entirely
    * Unresolved issues can retain more tokens
    * Deadlocked issues that remain unresolved after multiple attempts should be marked with additional annotations indicating multiple failed attempts
    * Retain records of unresolved issues or code under development, and prompt the model to continue solving those issues
2. Return your summarized message history without adding extra content (such as "Let me summarize..." or "Below is a summary of...")

Your optimization objectives:
1. [Priority] Restore an overview of unresolved issues, retain and summarize the model's trajectory for future reference
2. [Secondary] Retain as few tokens as possible""" # noqa

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
            keep_messages = messages[:2] # keep system and user
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

            query = (f'# Messages to be retained\n'
                     f'## system and user: {keep_messages_json}\n'
                     f'## Last assistant response: {keep_messages_tail_json}\n'
                     f'# Messages to be compressed'
                     f'## These messages are located between system/user '
                     f'and the last assistant response: {compress_messages}')

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
                    role='user', content='History messages are compressed due to a long sequence, now '
                                         'continue solve your problem according to '
                                         'the messages and the tool calling:\n')
            ]
            return messages
        else:
            return messages

    async def run(self, messages: List[Message]):
        return await self.condense_memory(messages)



