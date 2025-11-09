import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Any, Tuple

from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.llm.openai_llm import OpenAI
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


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
        patterns = [
            Pattern(
                name='formula',
                pattern=r'<formula>(.*?)</formula>',
                tags=['<formula>', '</formula>']),
            Pattern(
                name='code',
                pattern=r'<code>(.*?)</code>',
                tags=['<code>', '</code>']),
            Pattern(
                name='chart',
                pattern=r'<chart>(.*?)</chart>',
                tags=['<chart>', '</chart>']),
            Pattern(
                name='definition',
                pattern=r'<definition>(.*?)</definition>',
                tags=['<definition>', '</definition>']),
            Pattern(
                name='theorem',
                pattern=r'<theorem>(.*?)</theorem>',
                tags=['<theorem>', '</theorem>']),
            Pattern(
                name='example',
                pattern=r'<example>(.*?)</example>',
                tags=['<example>', '</example>']),
            Pattern(
                name='emphasis',
                pattern=r'<emphasis>(.*?)</emphasis>',
                tags=['<emphasis>', '</emphasis>'])
        ]
        return patterns

    async def execute_code(self, inputs, **kwargs):
        messages, context = inputs
        script = None
        with open(os.path.join(self.work_dir, 'script.txt'), 'r') as f:
            script = f.read()
        assert script is not None
        logger.info(f'Segmenting script to sentences.')
        topic = context['topic']
        segments = await self.generate_segments(topic, script)
        context['segments'] = segments
        for segment in segments:
            assert 'content' in segment
            assert 'background' in segment
        return messages, context

    async def generate_segments(self, topic, script) -> list:
        segments = self.split_scene(topic, script)
        return segments

    def split_scene(self, topic, script):
        system = """你是一个动画分镜设计师，现在有一个短视频场景需要设计分镜。分镜需要满足条件：

1. 每个分镜会携带一段旁白、（最多）一个manim技术动画、一个生成的图片背景和一个字幕
    * 你可以自由决定manim动画是否存在，如果manim动画不需要，在返回值中可以没有manim这个key
2. 你的每个分镜如果按正常语速朗读约20秒左右，太短会造成切换的频繁感，太长会显得过于定格
3. 你需要给每个分镜写出具体的旁白、技术动画的要求，以及背景图片的**细节**要求
4. 你会被给与一段源台本，你的分镜设计需要根据台本进行，你也可以额外增加一些你觉得有用的信息
5. 你的返回格式是json格式
6. 你需要注意，不要使用中文引号，使用[]来代替它，例如[attention]

一个例子：
```json
[
    {
        "content": "下面我们来讲解...",
        "background": "背景图片需要满足色调... 人物... 背景... 视角...",
        "manim": "图表需要满足... 位置在... ",
    },
    ...
]
```

现在开始：
"""
        query = f'原始需求：\n\n{topic}\n\n，原始脚本：\n\n{script}\n\n请根据脚本完成你的设计:'
        inputs = [
            Message(role='system', content=system),
            Message(role='user', content=query),
        ]
        _response_message = self.llm.generate(inputs)
        response = _response_message.content
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        return json.loads(response)

    def save_history(self, messages, **kwargs):
        messages, context = messages
        self.config.context = context
        return super().save_history(messages, **kwargs)

    def read_history(self, messages, **kwargs):
        _config, _messages = super().read_history(messages, **kwargs)
        if _config is not None:
            context = _config['context']
            return _config, (_messages, context)
        else:
            return _config, _messages
