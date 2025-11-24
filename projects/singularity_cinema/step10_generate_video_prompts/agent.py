# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union

from omegaconf import DictConfig

from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.utils import get_logger

logger = get_logger(__name__)


class GenerateVideoPrompts(CodeAgent):

    system = """你是一个为生成视频创作场景描述的专家。根据给定的知识点或分镜脚本，生成详细的英文描述，用于创建符合指定主题和风格的文生视频。要求：

生成的视频必须只描绘一个场景，不是多个场景
只有在确实需要表达知识点或场景含义时，才在图像中添加清晰、可读的文本。不要强制在每个场景中使用任何特定单词。如果不需要文本，就不要包含任何文本。
图像中的所有文本必须清晰、可读，不能扭曲、乱码或随机。
所有元素都应与主题和当前字幕片段的含义相关
视频面板尺寸为1920*1080
视频需要能准确反映文本的要求
用英文输出200个单词左右，只输出场景描述，不包含风格关键词，只有在场景确实需要时才在图像中使用英文文本。
只返回提示词本身，不要添加任何其他解释或标记。"""  # noqa

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.num_parallel = getattr(self.config, 'llm_num_parallel', 10)
        self.video_prompts_dir = os.path.join(self.work_dir,
                                                     'video_prompts')
        os.makedirs(self.video_prompts_dir, exist_ok=True)

    async def execute_code(self, messages: Union[str, List[Message]],
                           **kwargs) -> List[Message]:
        if self.config.use_text2video:
            return messages
        with open(os.path.join(self.work_dir, 'segments.txt'), 'r') as f:
            segments = json.load(f)
        with open(os.path.join(self.work_dir, 'topic.txt'), 'r') as f:
            topic = f.read()
        logger.info('Generating video prompts.')

        tasks = [(i, segment) for i, segment in enumerate(segments)]

        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = {
                executor.submit(self._generate_video_prompts_static, i,
                                segment, self.config, topic, self.system,
                                self.video_prompts_dir): i
                for i, segment in tasks if 'video' in segment
            }
            for future in as_completed(futures):
                future.result()
        return messages

    @staticmethod
    def _generate_video_prompts_static(i, segment, config, topic,
                                              system,
                                              video_prompts_dir):
        llm = LLM.from_config(config)
        GenerateVideoPrompts._generate_video_prompt_impl(
            llm, i, segment, topic, system, video_prompts_dir)

    @staticmethod
    def _generate_video_prompt_impl(llm, i, segment, topic, system,
                                    video_prompts_dir):
        if os.path.exists(
                os.path.join(video_prompts_dir, f'segment_{i+1}.txt')):
            return
        video = segment['video']
        query = (f'The user original request is: {topic}, '
                 f'illustration based on: {segment["content"]}, '
                 f'Requirements from the storyboard designer: {video}')
        logger.info(
            f'Generating video prompt for : {segment["content"]}.')
        inputs = [
            Message(role='system', content=system),
            Message(role='user', content=query),
        ]
        _response_message = llm.generate(inputs)
        response = _response_message.content
        prompt = response.strip()
        with open(
                os.path.join(video_prompts_dir, f'segment_{i + 1}.txt'),
                'w') as f:
            f.write(prompt)

