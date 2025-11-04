import re
import uuid
from dataclasses import dataclass, field
import json
import os
from typing import List, Union

from omegaconf import DictConfig, OmegaConf

from ms_agent.agent import CodeAgent
from ms_agent.llm import Message, LLM
from ms_agent.llm.openai_llm import OpenAI
from projects.video_generate.core import workflow as video_workflow
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
        self.animation_mode = os.environ.get('MS_ANIMATION_MODE',
                                              'auto').strip().lower() or 'auto'
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
                    Pattern(name='emphasis', pattern=r'<emphasis>(.*?)</emphasis>', tags=['<emphasis>', '</emphasis>']),
                    Pattern(name='comparison', pattern=r'<comparison>(.*?)</comparison>',
                            tags=['<comparison>', '</comparison>']),
                    Pattern(name='step', pattern=r'<step>(.*?)</step>', tags=['<step>', '</step>']),
                    Pattern(name='metaphor', pattern=r'<metaphor>([^<]*?)(?=<|$)', tags=['<metaphor>']),
                    Pattern(name='analogy', pattern=r'<analogy>([^<]*?)(?=<|$)', tags=['<analogy>']),
                    Pattern(name='note', pattern=r'<note>([^<]*?)(?=<|$)', tags=['<note>']),
                    Pattern(name='tip', pattern=r'<tip>([^<]*?)(?=<|$)', tags=['<tip>']),
                    Pattern(name='key', pattern=r'<key>([^<]*?)(?=<|$)', tags=['<key>'])]
        return patterns

    async def run(self, inputs: Union[str, List[Message]],
                  **kwargs) -> List[Message]:
        return await self._generate_segments(inputs)

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

    async def _generate_segments(self, messages: List[Message]) -> str:
        logger.info('Starting asset generation from script')
        with open('script.txt', 'r', encoding='utf-8') as f:
            script = f.read()

        for message in messages:
            if message.role == 'user':
                topic = messages[1]
                break

        segments = self.parse_structured_content(script)

        # Further split long text segments
        final_segments = []
        for segment in segments:
            if segment['type'] == 'text' and len(segment['content']) > 100:
                subsegments = video_workflow.split_text_by_punctuation(
                    segment['content'])
                for subseg_dict in subsegments:
                    if subseg_dict['content'].strip():
                        final_segments.append({
                            'content':
                            subseg_dict['content'].strip(),
                            'type':
                            'text',
                            'parent_segment':
                            segment
                        })
            else:
                final_segments.append(segment)
        segments = final_segments
        logger.info(f'[video_agent] Script parsed into {len(segments)} segments.')
        return segments
