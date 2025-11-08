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
        for i, segment in enumerate(segments):
            if segment['type'] == 'text' and self.animation_mode == 'human':
                context['manim_code'].append(None)
                continue
            content_type = segment['type']
            audio_duration = segment['audio_duration']
            class_name = f'Scene{i+1}'
            content = segment['content']

            base_prompt = f"""You are a professional Manim animation expert, creating clear and beautiful educational animations.

**Task**: Create {content_type} type animation
- Class name: {class_name}
- Content: {content}
- Duration: {audio_duration} seconds
- Code language: **Python**

**Color Requirements (CRITICAL)**:
• ALL text must use BLACK color: Text(..., color=BLACK)
• ALL math formulas must use BLACK color: MathTex(..., color=BLACK)
• Do NOT use random colors or color gradients
• Keep consistent BLACK color throughout the animation
• Use color=BLACK explicitly for every Text, MathTex, Tex object

**Spatial Constraints (Important)**:
• Safe area: x∈(-6.5, 6.5), y∈(-3.5, 3.5) (0.5 units from edge)
• Element spacing: Use buff=0.3 or larger (avoid overlap)
• Relative positioning: Prioritize next_to(), align_to(), shift()
• Avoid multiple elements using the same reference point

**Layout Suggestions**:
"""

            # Add specific layout strategy based on type
            if content_type == 'definition':
                layout_strategy = """• Title centered and slightly up (UP*2~3)
                    • Definition content in center area
                    • Examples or supplementary notes below
                    • Use clear visual hierarchy"""

            elif content_type == 'example':
                layout_strategy = """• Example title at top
                    • Core example in center
                    • Step-by-step display from top to bottom
                    • Comparison content arranged left and right"""

            elif content_type == 'emphasis':
                layout_strategy = """• Core information centered and prominent
                    • Supporting content displayed around it
                    • Use color and size to emphasize key points
                    • Animation effects enhance expression"""

            else:
                layout_strategy = """• Content clearly layered
                    • Key information highlighted
                    • Reasonable use of space
                    • Maintain visual balance"""

            prompt = base_prompt + layout_strategy + """

**Animation Requirements**:
• Concise and smooth animation effects
• Progressive display, avoid information overload
• Appropriate pauses and rhythm
• Professional visual presentation

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
            context['manim_code'].append(manim_code)
        return messages, context

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