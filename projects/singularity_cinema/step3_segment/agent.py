# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import json
from copy import deepcopy

from ms_agent.agent import LLMAgent
from ms_agent.llm import Message
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


class Segment(LLMAgent):

    system = """You are an animation storyboard designer. Now there is a short video scene that needs storyboard design. The storyboard needs to meet the following conditions:

- Each storyboard panel will carry a piece of narration, (at most) one manim technical animation, one generated image background, and one subtitle
    * You can freely decide whether the manim animation exists. If the manim animation is not needed, the manim key can be omitted from the return value
    * For tech-related short videos, they should have a technical and professional feel. For product-related short videos, they should be gentle and authentic, avoiding exaggerated expressions such as "shocking", "solved by xxx", "game-changing," "rule-breaking," or "truly achieved xx", etc. Describe things objectively and accurately.
    * Consider the color style across all storyboards comprehensively, for example, all purple style, all deep blue style, etc.
        - Consider more colors like white, black, dark blue, dark purple, dark orange, etc, which will make your design elegant, avoid using light yellow/blue, which will make your animation look superficial, DO NOT use grey color, it's not easy to read
    * Use less stick man unless the user wants to, to prevent the animation from being too naive, try to make your effects more dazzling/gorgeous/spectacular/blingbling

- Each of your storyboard panels should take about 5 seconds to 10 seconds to read at normal speaking speed. Avoid the feeling of frequent switching and static
    * If a storyboard panel has no manim animation, it should not exceed 5s
    * Pay attention to the coordination between the background image and the manim animation.
        - If a manim animation exists, the background image should not be too flashy. Else the background image will become the main focus, and the image details should be richer
        - The foreground and the background should not have the same objects. For example, draw birds at the foreground, sky and clouds at the background, other examples like charts and scientist, cloth and girls
    * If a storyboard panel has manim animation, the image should be more concise, with a stronger supporting role

- Write specific narration for each storyboard panel, technical animation requirements, and **detailed** background image requirements
    * Specify your expected manim animation content, presentation details, position and size, etc., and remind the large model generating manim of technical requirements, and **absolutely prevent size overflow and animation position overlap**
    * Estimate the reading duration of this storyboard panel to estimate the duration of the manim animation. The actual duration will be completely determined in the next step of voice generation
    * The video resolution is around 1920*1080, 200-pixel margin on all four sides for title and subtitle, so manim can use center (1500, 700).
    * Use thicker lines to emphasis elements
    * Use smaller font size and smaller elements in Manim animations to prevent from going beyond the canvas
    * LLMs excel at animation complexity, not layout complexity.
        - Use multiple storyboard scenes rather than adding more elements to one animation to avoid layout problems
        - For animations with many elements, consider layout carefully. For instance, arrange elements horizontally given the canvas's wider width
        - With four or more horizontal elements, put summary text or similar content at the canvas bottom, this will effectively reduce the cutting off and overlap problems
    * Consider the synchronization between animations and content. When read at a normal speaking pace, the content should align with the animation's progression.
    * Specify the language of the manim texts, it should be the same with the script and the storyboard content(Chinese/English for example)

- You will be given a script. Your storyboard design needs to be based on the script. You can also add some additional information you think is useful

- Review the requirements and any provided documents. Integrate their content, formulas, charts, and visuals into the script to refine the video's screenplay and animations.
    [CRITICAL]: The manim and image generation steps will not receive the original requirements and files. Supply very detail information for them, especially any data/points/formulas to prevent any mismatch with the original query and/or documentation
    
- Your return format is JSON format, no need to save file, later the json will be parsed out of the response body

- You need to pay attention not to use Chinese quotation marks. Use [] to replace them, for example [attention]

An example:
```json
[
    {
        "index": 1, # index of the segment, start from 1
        "content": "Now let's explain...",
        "background": "An image describe... color ... (your detailed requirements here)",
        "manim": "The animation should ... draw component ...",
    },
    ...
]
```
```

Now begin:""" # noqa

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        _config = deepcopy(config)
        _config.prompt.system = self.system
        _config.tools = DictConfig({})
        super().__init__(_config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.images_dir = os.path.join(self.work_dir, 'images')

    async def create_messages(self, messages):
        assert isinstance(messages, str)
        return [
            Message(role='system', content=self.system),
            Message(role='user', content=messages),
        ]

    async def run(self, messages, **kwargs):
        logger.info('Segmenting script to sentences.')
        script = None
        if os.path.exists(os.path.join(self.work_dir, 'segments.txt')):
            return messages
        with open(os.path.join(self.work_dir, 'script.txt'), 'r') as f:
            script = f.read()
        with open(os.path.join(self.work_dir, 'topic.txt'), 'r') as f:
            topic = f.read()

        query = (
            f'Original topic: \n\n{topic}\n\n'
            f'Original script：\n\n{script}\n\n'
            f'Please finish your animation storyboard design:\n')
        messages = await super().run(query, **kwargs)
        response = messages[-1].content
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        segments = json.loads(response)
        segments = await self.add_images(segments, topic, script, **kwargs)

        for i, segment in enumerate(segments):
            assert 'content' in segment
            assert 'background' in segment
            logger.info(
                f'\nScene {i}\n'
                f'Content: {segment["content"]}\n'
                f'Image requirement: {segment["background"]}\n'
                f'Manim requirement: {segment.get("manim", "No manim")}')
        with open(os.path.join(self.work_dir, 'segments.txt'), 'w') as f:
            f.write(json.dumps(segments, indent=4, ensure_ascii=False))
        return messages

    async def add_images(self, segments, topic, script, **kwargs):

        system = """你是一个动画短视频分镜辅助设计师。你的职责是协助分镜设计师为分镜添加前景图片。你会被一个给与一个分镜设计草案，和一些图片列表，该图片列表来自用户输入，可以自由挑选使用。
下面你需要选择两类图片：

Manim animation may contain one or more images, these images come from user's documentation, or a powerful text-to-image model (same way with the generated background images)
- If the user's documentation contains any images, the information will be given to you:
    * The image information will include content description, size(width*height) and filename
    * Select useful images in each segment and reference the filename in the `user_image` field

- User-provided images may be insufficient. Trust text-to-image models to generate additional images for more visually compelling videos
    * Output image generation requirements and the generated filenames(with .png format) in `foreground` field
    
1. 来自用户的图片，可以直接使用
2. 你认为的可以增加短视频效果的图片，你需要给出生成图片的具体要求,一个收信任的文生图模型会帮你生成它，生成的图片均为正方形
3. 修改对应分镜的manim字段，该字段用于后续的manim动画生成的指导意见，修改该字段使后续manim生成大模型清楚如何使用你的这些图片
4. 每个分镜使用的图片数量不必相同，也可以不使用图片
5. 仔细分析用户提供的图片信息，尽可能使用它们
6. 减少注意力分散，你需要仅关心图片信息、manim两个字段，并生成manim、user_image、foreground三个字段
7. 图片不宜过大，防止占满整个屏幕或大半个屏幕
8. 重要: 考虑图片的展示尺寸，防止一个动画中出现太多元素无法布局

一个例子：
```json
[
    {
        "index": 1, # index of the segment, start from 1
        "manim": "The animation should ..., use images to... ",
        "user_image": [
            "user_image1.jpg",
            "user_image2.jpg"
        ]
        "foreground": [
            "An image describe... color ... (your detailed requirements here)",
            ...
        ],
    },
    ...
]
```

An example of image structures given to the manim LLM:
```json
[
    {
        "file_path": "foreground_images/1.jpg",
        "size": "2000*2000",
        "description": "The image contains ..."
    },
    ...
]

注意:
* 你的返回值中不必要包括content和background，这些信息不需要你关心
* 你的返回长度应当和源分镜长度相同。如果不需要图片则返回空的user_image和foreground列表

现在开始：
"""
        new_image_info = 'No images offered.'
        name_mapping = {}
        if os.path.exists(os.path.join(self.work_dir, 'image_info.txt')):
            with open(os.path.join(self.work_dir, 'image_info.txt'), 'r') as f:
                image_info = f.readlines()

            image_info = [image.strip() for image in image_info if image.strip()]
            image_list = []
            for i, info in enumerate(image_info):
                info = json.loads(info)
                filename = info['filename']
                new_filename = f'user_image_{i}.png'
                name_mapping[new_filename] = filename
                info['filename'] = new_filename
                image_list.append(json.dumps(info, ensure_ascii=False))

            new_image_info = json.dumps(image_list, ensure_ascii=False)

        query = (
            f'Original topic: \n\n{topic}\n\n'
            f'Original script：\n\n{script}\n\n'
            f'Original segments：\n\n{json.dumps(segments, ensure_ascii=False, indent=4)}\n\n'
            f'User offered images: \n\n{new_image_info}\n\n'
            f'Please finish your images design:\n')
        messages = [
            Message(role='system', content=system),
            Message(role='user', content=query),
        ]
        message = self.llm.generate(messages, **kwargs)
        response = message.content
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        _segments = json.loads(response)

        for i, segment in enumerate(_segments):
            user_images = segment.get('user_image', [])
            new_user_images = []
            for image in user_images:
                if image in name_mapping:
                    new_user_images.append(name_mapping[image])
            segment['user_image'] = new_user_images

        assert len(_segments) == len(segments)
        for segment, _segment in zip(segments, _segments):
            segment.update(_segment)

        return segments

