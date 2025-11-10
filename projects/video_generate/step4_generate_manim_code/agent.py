import asyncio

from omegaconf import DictConfig

from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.llm.openai_llm import OpenAI
from ms_agent.utils import get_logger

logger = get_logger()


class GenerateManimCode(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.llm: OpenAI = LLM.from_config(self.config)
        self.animation_mode = getattr(self.config, 'animation_code', 'auto')

    async def execute_code(self, inputs, **kwargs):
        messages, context = inputs
        segments = context['segments']
        context['manim_code'] = []
        logger.info(f'Generating manim code.')
        tasks = []
        for i, segment in enumerate(segments):
            manim_requirement = segment.get('manim')
            if manim_requirement is not None:
                tasks.append(self.generate_manim_code(segment, i))
            else:
                tasks.append(asyncio.sleep(0, result=None))
        context['manim_code'] = await asyncio.gather(*tasks)
        return messages, context

    async def generate_manim_code(self, segment, i):
        audio_duration = segment['audio_duration']
        class_name = f'Scene{i + 1}'
        content = segment['content']
        manim_requirement = segment.get('manim')

        prompt = f"""You are a professional Manim animation expert, creating clear and beautiful educational animations.

**Task**: Create animation
- Class name: {class_name}
- Content: {content}
- Extra requirement: {manim_requirement}
- Duration: {audio_duration} seconds
- Code language: **Python**

**Color Requirements (CRITICAL)**:
• ALL text must use BLACK color: Text(..., color=BLACK)
• ALL math formulas must use BLACK color: MathTex(..., color=BLACK)
• Do NOT use random colors or color gradients
• Keep consistent BLACK color throughout the animation
• Use color=BLACK explicitly for every Text, MathTex, Tex object

**Line Thickness & Stroke Requirements (CRITICAL)**:
• ALL shapes (Rectangle, Circle, Arrow, Line, etc.) must use THICK strokes: stroke_width=4
• For emphasis or important diagrams: stroke_width=5 or 6
• For subtle elements: minimum stroke_width=3
• Arrows must be prominent: Arrow(..., stroke_width=5, buff=0.15)
• Always explicitly specify stroke_width for every shape
• Never use default thin strokes - they are too thin for visibility

**Spatial Constraints (Important)**:
• Safe area: x∈(-6.5, 6.5), y∈(-3.5, 3.5) (0.5 units from edge)
• Element spacing: Use buff=0.3 or larger (avoid overlap)
• Relative positioning: Prioritize next_to(), align_to(), shift()
• Avoid multiple elements using the same reference point

**Box/Rectangle Size Standards (CRITICAL)**:
• For diagram boxes: Use consistent dimensions, e.g., Rectangle(width=2.5, height=1.5, stroke_width=4)
• For labels/text boxes: width=1.5~3.0, height=0.8~1.2, stroke_width=4
• For emphasis boxes: width=3.0~4.0, height=1.5~2.0, stroke_width=5
• Always specify both width AND height explicitly: Rectangle(width=2.5, height=1.5, stroke_width=4)
• Avoid using default sizes - always set explicit dimensions
• Maintain consistent box sizes within the same diagram level/category
• All boxes must have thick strokes for clear visibility

**Visual Quality Enhancement**:
• Use thick, clear strokes for all shapes (stroke_width=4 minimum)
• Make arrows bold and prominent (stroke_width=5, tip_length=0.25)
• Add rounded corners for modern aesthetics: RoundedRectangle(corner_radius=0.15)
• Use subtle fill colors with transparency when appropriate: fill_opacity=0.1
• Ensure high contrast between elements for clarity
• Apply consistent spacing and alignment throughout

**Layout Suggestions**:
• Content clearly layered
• Key information highlighted
• Reasonable use of space
• Maintain visual balance

- If rendering a definition:
    • Title centered and slightly up (UP*2~3)
    • Definition content in center area
    • Examples or supplementary notes below
    • Use clear visual hierarchy

- If rendering an example:
    • Example title at top
    • Core example in center
    • Step-by-step display from top to bottom
    • Comparison content arranged left and right

- If rendering an emphasis:
    • Core information centered and prominent
    • Supporting content displayed around it
    • Use color and size to emphasize key points
    • Animation effects enhance expression

**Animation Requirements**:
• Concise and smooth animation effects
• Progressive display, avoid information overload
• Appropriate pauses and rhythm
• Professional visual presentation with thick, clear lines
• Use GrowArrow for arrows instead of Create for better effect
• Consider using Circumscribe or Indicate to highlight important elements

**Code Style**:
• Implement directly in Scene class
• Use VGroup appropriately to organize related elements
• Clear comments and explanations
• Avoid overly complex structures

Please create Manim animation code that meets the above requirements."""

        logger.info(f'Generating manim code for: {content}')
        _response_message = self.llm.generate(
            [Message(role='user', content=prompt)], temperature=0.3)
        response = _response_message.content
        if '```python' in response:
            manim_code = response.split('```python')[1].split('```')[0]
        elif '```' in response:
            manim_code = response.split('```')[1].split('```')[0]
        else:
            manim_code = response
        return manim_code

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