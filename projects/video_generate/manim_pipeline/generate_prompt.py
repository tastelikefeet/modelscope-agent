from ms_agent.agent import CodeAgent
from ms_agent.llm import Message


class GeneratePrompt(CodeAgent):

    async def run(self, inputs, **kwargs):
        content_type = kwargs['content_type']
        content = inputs
        audio_duration = kwargs['audio_duration']
        class_name = kwargs['class_name']

        base_prompt = f"""You are a professional Manim animation expert, creating clear and beautiful educational animations.

**Task**: Create {content_type} type animation
- Class name: {class_name}
- Content: {content}
- Duration: {audio_duration} seconds
- Code language: **Python**

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

        return [Message(role='user', content=prompt)]