import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import List

from omegaconf import DictConfig

from ms_agent.agent import CodeAgent
from ms_agent.llm import Message, LLM
from ms_agent.llm.openai_llm import OpenAI
from ms_agent.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Pattern:

    name: str
    pattern: str
    tags: List[str] = field(default_factory=list)


class Segment(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.patterns = self.create_patterns()
        self.llm: OpenAI = LLM.from_config(self.config)

    @staticmethod
    def create_patterns():
        patterns = [Pattern(name='formula', pattern=r'<formula>(.*?)</formula>', tags=['<formula>', '</formula>']),
                    Pattern(name='code', pattern=r'<code>(.*?)</code>', tags=['<code>', '</code>']),
                    Pattern(name='chart', pattern=r'<chart>(.*?)</chart>', tags=['<chart>', '</chart>']),
                    Pattern(name='definition', pattern=r'<definition>(.*?)</definition>',
                            tags=['<definition>', '</definition>']),
                    Pattern(name='theorem', pattern=r'<theorem>(.*?)</theorem>', tags=['<theorem>', '</theorem>']),
                    Pattern(name='example', pattern=r'<example>(.*?)</example>', tags=['<example>', '</example>']),
                    Pattern(name='emphasis', pattern=r'<emphasis>(.*?)</emphasis>', tags=['<emphasis>', '</emphasis>'])]
        return patterns

    async def run(self, inputs, **kwargs):
        messages, context = inputs
        script = None
        with open(os.path.join(self.work_dir, 'script.txt'), 'r') as f:
            script = f.read()
        assert script is not None
        segments = await self.generate_segments(script)
        context['segments'] = segments
        return messages, context

    async def generate_segments(self, script) -> list:
        segments = self.parse_structured_content(script)
        final_segments = []
        async_tasks = []
        task_indices = []

        for segment in segments:
            if segment['type'] == 'text' and len(segment['content']) > 100:
                task = self.split_text_by_punctuation(segment['content'])
                async_tasks.append(task)
                task_indices.append((len(final_segments), segment))
                final_segments.append(None)
            else:
                final_segments.append(segment)

        if async_tasks:
            results = await asyncio.gather(*async_tasks)
            for (index, parent_segment), subsegments in zip(task_indices, results):
                processed_subsegments = []
                for subseg_dict in subsegments:
                    if subseg_dict['content'].strip():
                        processed_subsegments.append({
                            'content': subseg_dict['content'].strip(),
                            'type': 'text',
                            'parent_segment': parent_segment
                        })
                final_segments[index] = processed_subsegments

        flattened_segments = []
        for item in final_segments:
            if isinstance(item, list):
                flattened_segments.extend(item)
            else:
                flattened_segments.append(item)

        return flattened_segments

    async def split_text_by_punctuation(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        prompt = f"""Please intelligently segment the text into sentences, ensuring:
1. Each sentence is semantically complete without breaking the logic
2. Punctuation marks remain at the end of sentences and are not separated
3. Each sentence has moderate length: at least 10-15 characters, maximum 35-40 characters
4. Prioritize splitting at natural semantic boundaries (such as before/after conjunctions like: therefore, so, but, moreover, etc.)
5. Preserve the original meaning

Return a list of sentences, separated by lines, for example:
Sentence 1
Sentence 2
Sentence 3
...

MANDATORY: Only return split sentences, DO NOT contain any thinking logics or prefixes like `Here is the list...`.

Here is the original text:"""
        messages = [
            Message(role='system', content=prompt),
            Message(role='user', content=text),
        ]

        _response_message = self.llm.generate(messages)
        segments = _response_message.content.split('\n')
        segments = [s.strip() for s in segments if s.strip()]
        return segments

    def parse_structured_content(self, script):
        segments = []
        current_pos = 0

        all_matches = []
        for item in self.patterns:
            for match in re.finditer(item.pattern, script, re.DOTALL):
                all_matches.append({
                    'start': match.start(),
                    'end': match.end(),
                    'type': item.name,
                    'content': match.group(1).strip(),
                    'full_match': match.group(0)
                })

        all_matches.sort(key=lambda x: x['start'])

        for i, match in enumerate(all_matches):
            if match['start'] > current_pos:
                normal_text = script[current_pos:match['start']].strip()
                if normal_text:
                    segments.append({'type': 'text', 'content': normal_text})

            context_start = max(0, match['start'] - 100)
            context_end = min(len(script), match['end'] + 100)
            surrounding_text = script[context_start:context_end]

            # TODO
            context_info = None

            segments.append({
                'type': match['type'],
                'content': match['content'],
                'surrounding_text': surrounding_text,
                'context_info': context_info,
                'position_in_script': match['start'] / len(script)
            })

            current_pos = match['end']

        if current_pos < len(script):
            remaining_text = script[current_pos:].strip()
            if remaining_text:
                segments.append({'type': 'text', 'content': remaining_text})

        return segments