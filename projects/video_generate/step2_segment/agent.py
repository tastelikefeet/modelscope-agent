import json
import os

from omegaconf import DictConfig

from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.llm.openai_llm import OpenAI
from ms_agent.utils import get_logger

logger = get_logger()


class Segment(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.llm: OpenAI = LLM.from_config(self.config)

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
        for i, segment in enumerate(segments):
            assert 'content' in segment
            assert 'background' in segment
            logger.info(f'\nScene {i}\n'
                        f'Content: {segment["content"]}\n'
                        f'Image requirement: {segment["background"]}\n'
                        f'Manim requirement: {segment.get("background")}')
        return messages, context

    async def generate_segments(self, topic, script) -> list:
        segments = self.split_scene(topic, script)
        return segments

    def split_scene(self, topic, script):
        system = """You are an animation storyboard designer. Now there is a short video scene that needs storyboard design. The storyboard needs to meet the following conditions:

1. Each storyboard panel will carry a piece of narration, (at most) one manim technical animation, one generated image background, and one subtitle
    * You can freely decide whether the manim animation exists. If the manim animation is not needed, the manim key can be omitted from the return value
2. Each of your storyboard panels should take about 10 seconds to 20 seconds to read at normal speaking speed. Too short will cause a sense of frequent switching, and too long will appear too static
    * If a storyboard panel has no manim animation, it should not exceed 5s to 10s at most
    * Pay attention to the coordination between the background image and the manim animation. If a manim animation exists, the background image should not be too flashy. Else the background image will become the main focus, and the image details should be richer
    * If a storyboard panel has manim animation, the image should be more concise, with a stronger supporting role
3. You need to write specific narration for each storyboard panel, technical animation requirements, and **detailed** background image requirements
    * You need to specify your expected manim animation content, presentation details, position and size, etc., and remind the large model generating manim of technical requirements, and **absolutely prevent size overflow and animation position overlap**
    * You must specify the color scheme for the manim animation, and this color scheme must be coordinated with the background color scheme. For example, if the background color scheme is light-colored, then the text, boxes, arrows, etc. in the manim animation should generally use dark colors. If the background is dark-colored, then the elements of the manim animation should use light colors.
    * You can estimate the reading duration of this storyboard panel to estimate the duration of the manim animation. The actual duration will be completely determined in the next step of voice generation
    * The video resolution is around 1920*1080. Lines that are too thin are easily difficult to see clearly. You need to explicitly specify the line thickness of the manim animation, emphasis elements should use thicker lines
4. You will be given a script. Your storyboard design needs to be based on the script. You can also add some additional information you think is useful
5. Your return format is JSON format
6. You need to pay attention not to use Chinese quotation marks. Use [] to replace them, for example [attention]

An example:
```json
[
    {
        "content": "Now let's explain...",
        "background": "An image describe... color ... (your detailed requirements here)",
        "manim": "The animation should ... line thick... element color ... position ... (your detailed requirements here)",
    },
    ...
]
```

Now begin:"""
        query = f'Original topic: \n\n{topic}\n\n，original script：\n\n{script}\n\nPlease finish your animation storyboard design:\n'
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
