import json
import os

from omegaconf import DictConfig

from ms_agent.agent import LLMAgent
from ms_agent.llm import Message
from ms_agent.utils import get_logger

logger = get_logger()


class HumanFeedback(LLMAgent):

    system = """你是一个负责协助解决短视频生成中的人工反馈问题的助手。你的职责是根据人工反馈的问题，定位问题出现在哪个工作流程中，并适当删除前置任务的配置文件，以触发任务的重新执行。

前置工作流：
首先有一个根目录文件夹用于存储所有文件，以下描述的所有文件，或你的所有shell命令都基于这个根目录进行，你无需关心根目录位置，仅需要关注相对目录即可
1. 根据用户需求生成基本台本，存储到script.txt中，短视频名字，存储到title.txt中
2. 根据台本切分分镜设计，分镜设计存储在内存中，稍后该设计会给你
3. 生成分镜的音频讲解，文件存储在audio文件夹中，以segment_N.mp3命名，其中N从1开始，代表分镜索引
4. 生成manim动画代码，这部分的代码
5. 修复manim
"""

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        config.prompt.system = self.system
        config.tools = DictConfig({
            "file_system":{
                "mcp": False,
                "exclude": ["create_directory", "write_file"]
            }
        })
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')

    async def create_messages(self, messages):
        assert isinstance(messages, str)
        return [
            Message(role='system', content=self.system),
            Message(role='user', content=messages),
        ]

    async def run(self, inputs, **kwargs):
        logger.info(f'Segmenting script to sentences.')
        messages, context = inputs
        script = None
        with open(os.path.join(self.work_dir, 'script.txt'), 'r') as f:
            script = f.read()
        title = None
        with open(os.path.join(self.work_dir, 'title.txt'), 'r') as f:
            title = f.read()
        assert title.strip() is not None
        context['title'] = title.strip()
        topic = context['topic']
        query = f'Original topic: \n\n{topic}\n\n，original script：\n\n{script}\n\nPlease finish your animation storyboard design:\n'
        messages = await super().run(query, **kwargs)
        response = messages[-1].content
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        segments = json.loads(response)
        for i, segment in enumerate(segments):
            assert 'content' in segment
            assert 'background' in segment
            logger.info(f'\nScene {i}\n'
                        f'Content: {segment["content"]}\n'
                        f'Image requirement: {segment["background"]}\n'
                        f'Manim requirement: {segment.get("manim", "No manim")}')
        context['segments'] = segments
        return messages, context

