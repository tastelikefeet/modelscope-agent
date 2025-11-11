import json
import os
import re
import shutil
from typing import Dict, List
import asyncio
from omegaconf import DictConfig

from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.llm.openai_llm import OpenAI
from ms_agent.utils import get_logger

logger = get_logger()


class FixManimCode(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.max_fix_rounds = getattr(self.config, 'max_fix_rounds', 3)
        self.llm: OpenAI = LLM.from_config(self.config)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.code_fix_dir = os.path.join(self.work_dir, 'code_fix')
        os.makedirs(self.code_fix_dir, exist_ok=True)

    async def execute_code(self, messages, **kwargs):
        logger.info(f'Fixing manim code.')
        
        async def process_single_code(i, pre_error, code):
            if not code:
                return i, ''

            for _ in range(self.max_fix_rounds):
                if pre_error is not None:
                    if pre_error:
                        code = await self.fix_code(pre_error, code)
                    break
                else:
                    analysis = self.analyze_and_score(code)
                    if not analysis['needs_fix'] or analysis['layout_score'] >= 90:
                        break

                    if analysis['issue_count'] == 0:
                        break

                    code = await self.fix_code(analysis['fix_prompt'], analysis['manim_code'])
            
            code = self.optimize_simple_code(code)
            return i, code

        with open(os.path.join(self.work_dir, 'segments.txt'), 'r') as f:
            segments = json.load(f)

        manim_code_dir = os.path.join(self.work_dir, 'manim_code')
        manim_code = []
        pre_errors = []
        pre_error_mode = False
        for i in range(len(segments)):
            with open(os.path.join(manim_code_dir, f'segment_{i+1}.py'), 'r') as f:
                manim_code.append(f.read())
            error_file = os.path.join(self.code_fix_dir, f'code_fix_{i + 1}.txt')
            if os.path.exists(error_file):
                pre_error_mode = True
                with open(error_file, 'r') as _f:
                    pre_error = _f.read()
                    pre_error = pre_error or ''
            else:
                pre_error = ''
            pre_errors.append(pre_error)
        assert len(manim_code) == len(segments)
        if pre_error_mode:
            assert len(pre_errors) == len(segments)
        tasks = [
            process_single_code(i, pre_error, code)
            for i, pre_error, code in enumerate(zip(manim_code, pre_errors))
        ]
        results = await asyncio.gather(*tasks)
        if pre_error_mode:
            shutil.rmtree(self.code_fix_dir)
        for i, code in enumerate(results):
            manim_file = os.path.join(manim_code_dir, f'segment_{i + 1}.py')
            with open(manim_file, 'w') as f:
                f.write(code)
        
        return messages

    @staticmethod
    def optimize_simple_code(code):
        """Fix spacing issues and normalize box sizes in code"""
        lines = code.split('\n')
        optimized_lines = []

        for line in lines:
            # Add buff to next_to to prevent overlap
            if 'next_to(' in line and 'buff=' not in line and ')' in line:
                line = line.replace(')', ', buff=0.3)')

            # Increase buff if too small
            line = re.sub(r'buff=0\.[012](?!\d)', 'buff=0.3', line)
            
            # Normalize Rectangle/Square sizes - ensure explicit width and height
            if 'Rectangle(' in line and 'width=' not in line:
                # Add default width and height if not specified
                line = line.replace('Rectangle(', 'Rectangle(width=2.5, height=1.5, ')
            
            if 'Square(' in line and 'side_length=' not in line:
                # Add default side_length if not specified
                line = line.replace('Square(', 'Square(side_length=1.5, ')
            
            # Ensure RoundedRectangle has size specifications
            if 'RoundedRectangle(' in line and 'width=' not in line:
                line = line.replace('RoundedRectangle(', 'RoundedRectangle(width=2.5, height=1.5, ')

            optimized_lines.append(line)

        return '\n'.join(optimized_lines)

    async def fix_code(self, fix_prompt, manim_code):
        fix_request = f"""
{fix_prompt}

**Original Code**:
```python
{manim_code}
```

- Please focus on solving the detected issues
- Keep the good parts, only fix problematic areas
- Ensure no new layout issues are introduced
- If some issues are difficult to solve, prioritize the most impactful ones

Please precisely fix the detected issues while maintaining the richness and creativity of the animation.
"""
        inputs = [Message(role='user', content=fix_request)]
        _response_message = self.llm.generate(inputs)
        response = _response_message.content
        if '```python' in response:
            manim_code = response.split('```python')[1].split('```')[0]
        elif '```' in response:
            manim_code = response.split('```')[1].split('```')[0]
        else:
            manim_code = response

        return manim_code

    @staticmethod
    def analyze_and_score(code):
        lines = code.split('\n')
        issues = FixManimCode.detect_layout_issues(code)

        # Basic statistics
        element_count = len([
            line for line in lines
            if any(kw in line
                   for kw in ['Text(', 'Circle(', 'Rectangle(', 'VGroup('])
        ])

        # Calculate score
        layout_score = 85  # Higher base score

        # Deduct points based on issue types
        for issue_group in issues:
            if 'Boundary Violation Risk' in issue_group:
                layout_score -= 15
            elif 'Overlap Risk' in issue_group:
                layout_score -= 10
            elif 'Layout Crowding' in issue_group:
                layout_score -= 8
            elif 'Complexity Issues' in issue_group:
                layout_score -= 12
            elif 'Animation Timing Issues' in issue_group:
                layout_score -= 5

        # Add points for content richness
        if element_count > 0:
            layout_score += min(element_count * 1, 15)

        layout_score = max(0, min(100, layout_score))

        # Count spacing issues
        spacing_issues = 0
        for issue_group in issues:
            if 'Overlap Risk' in issue_group or 'Layout Crowding' in issue_group:
                spacing_issues += 1

        # Check for over-engineering
        is_over_engineered = False
        class_count = len([
            line for line in lines
            if 'class' in line and line.strip().startswith('class')
        ])
        if element_count > 8 or class_count > 1:
            is_over_engineered = True

        return {
            'layout_score':
            layout_score,
            'element_count':
            element_count,
            'spacing_issues':
            spacing_issues,
            'is_over_engineered':
            is_over_engineered,
            'issues':
            issues,
            'issue_count':
            len(issues),
            'needs_fix':
            len(issues) > 0,
            'manim_code':
            code,
            'fix_prompt':
            FixManimCode.generate_fix_prompt(code, issues) if issues else ''
        }

    @staticmethod
    def generate_fix_prompt(code, issues):
        """Generate fix suggestions based on detected issues"""

        if not issues:
            return ''

        # Analyze code complexity and content richness
        lines = code.split('\n')
        text_count = len([line for line in lines if 'Text(' in line])
        animation_count = len([
            line for line in lines
            if any(anim in line
                   for anim in ['Write(', 'Create(', 'FadeIn(', 'Transform('])
        ])
        color_count = len([
            line for line in lines
            if any(color in line
                   for color in ['color=', 'fill_color=', 'stroke_color='])
        ])

        richness_level = 'Rich' if (text_count > 5 and animation_count > 5
                                    and color_count > 3) else 'Moderate' if (
                                        text_count > 3) else 'Simple'

        fix_prompt = f"""**Layout Issue Fix Task**

The following issues have been detected and need fixing:
{''.join(issues)}

**Fix Objectives**:
• Resolve all detected layout issues
• Maintain richness and diversity of animation effects (current content richness: {richness_level})
• Keep code concise and avoid over-engineering
• Ensure the final result doesn't lose original creativity and expressiveness
• Apply thick strokes to all shapes for better visibility (stroke_width=4 minimum)
• Make arrows bold and prominent (stroke_width=5, tip_length=0.25)

**Fix Guidelines**:

**Visual Quality Guidelines**:
• Stroke width: All shapes must use stroke_width=4 minimum (arrows: stroke_width=5)
• Arrow tips: Use tip_length=0.25 for better visibility
• Rounded corners: Apply corner_radius=0.15 to RoundedRectangle for modern look
• Line clarity: Never use default thin strokes - always specify thick strokes
• Consistency: Maintain uniform stroke widths throughout related elements
• [CRITICAL]Absolutely prevent **element spatial overlap** or **elements going out of bounds** or **elements not aligned**.
• [CRITICAL]Connection lines between boxes/text are of proper length, with **both endpoints attached to the objects**.

**Boundary Control**:
• Canvas size: (1280, 700) (width x height) which is the top 3/4 of screen, bottom is left for subtitles
• Safe area: x ∈ (-6.5, 6.5), y ∈ (-3.5, 3.5)
• Font size: Recommended 12-48, titles can be larger but not exceeding 60
• Use relative positioning: to_edge(), next_to(), align_to() instead of absolute coordinates

**Overlap Avoidance**:
• Element spacing: buff >= 0.3 (compact layouts can use 0.25)
• Avoid multiple center() calls: use arrange() or next_to() for distribution
• VGroup organization: manage positions of related elements with VGroup

**Layout Optimization**:
• Element layering: Title → Main content → Supplementary information visual hierarchy
• Space utilization: Distribute reasonably, avoid concentrated clustering
• Dynamic adjustment: Adaptive layout based on content volume
• Use more horizontal layouts to leverage the wider space and minimize positional conflicts between animation components.

**Maintain Animation Richness**:
• Diverse effects: Mix Write, Create, FadeIn, Transform, Indicate, etc.
• Rich colors: Maintain existing color schemes and emphasis effects
• Rhythm control: Moderate run_time and wait times (1-3 second range)
• Visual highlights: Keep special effects, highlights, dynamic changes, and creative elements

**Code Simplification**:
• Direct implementation: Complete directly in Scene class, avoid Helper classes
• Reasonable organization: Related functions can be properly encapsulated, but not over-decomposed
• Clear comments: Maintain code readability

**Fix Strategy Suggestions**:
- For boundary issues: Use to_edge() and shift() instead of move_to() with fixed coordinates
- For overlap issues: Increase buff parameter, use arrange() to distribute elements
- For crowding issues: Use grouped display or step-by-step presentation
- For complexity: Simplify structure while maintaining functional integrity

**Creativity Preservation Requirements**:
Please ensure during the fix process:
1. Keep all animation effects and visual creativity
2. Don't reduce use of colors and special effects
3. Maintain educational value and expressiveness of content
4. Keep animation rhythm smooth and engaging

Please return the complete fixed code, ensuring both layout issues are resolved and animation richness and creativity are maintained.""" # noqa
        return fix_prompt

    @staticmethod
    def detect_layout_issues(code):
        issues = []
        lines = code.split('\n')

        # Check boundary violations
        boundary_violations = []
        for i, line in enumerate(lines, 1):
            line_clean = line.strip()
            if not line_clean or line_clean.startswith('#'):
                continue

            # Check if move_to exceeds boundaries
            if re.search(r'\.move_to\(\[?\s*[+-]?([8-9]|[1-9]\d)', line):
                boundary_violations.append(
                    f'Line {i}: Absolute position out of bounds - {line_clean}'
                )

            # Large shift values can also cause out of bounds
            if re.search(r'\.shift\(\s*[A-Z_]*\s*\*\s*([6-9]|[1-9]\d)', line):
                boundary_violations.append(
                    f'Line {i}: Shift displacement too large - {line_clean}')

            # Font size check
            font_match = re.search(r'font_size\s*=\s*([0-9]+)', line)
            if font_match:
                size = int(font_match.group(1))
                if size > 64:  # Too large to display fully
                    boundary_violations.append(
                        f'Line {i}: Font too large ({size}) - {line_clean}')
                elif size < 10:  # Too small to read
                    boundary_violations.append(
                        f'Line {i}: Font too small ({size}) - {line_clean}')

        if boundary_violations:
            issues.append('Boundary Violation Risk:')
            issues.extend(f'   • {v}' for v in boundary_violations)

        # Generic overlap risk detection (based on spatial relationship analysis)
        overlap_risks = []

        # 1. Extract all spatial positioning operations
        spatial_operations = FixManimCode._extract_spatial_operations(lines)

        # 2. Analyze spatial relationship conflicts
        spatial_conflicts = FixManimCode._analyze_spatial_conflicts(
            spatial_operations)

        # 3. Convert to readable issue descriptions
        for conflict in spatial_conflicts:
            overlap_risks.append(
                FixManimCode._format_conflict_description(conflict))

        if overlap_risks:
            issues.append('Overlap Risk:')
            issues.extend(f'   • {r}' for r in overlap_risks)

        # Check crowding issues
        crowding_issues = []
        text_elements = len([
            line for line in lines
            if 'Text(' in line and not line.strip().startswith('#')
        ])
        circle_elements = len([
            line for line in lines
            if 'Circle(' in line and not line.strip().startswith('#')
        ])
        rect_elements = len([
            line for line in lines
            if any(shape in line for shape in ['Rectangle(', 'Square('])
            and not line.strip().startswith('#')
        ])

        total_elements = text_elements + circle_elements + rect_elements

        if text_elements > 12:  # Too many text elements
            crowding_issues.append(
                f'Too many text elements ({text_elements}), consider grouping or pagination'
            )

        if total_elements > 20:  # Too many total elements, display will be cluttered
            crowding_issues.append(
                f'Too many display elements ({total_elements}), may appear crowded'
            )
        
        # Check for boxes/rectangles without explicit size specifications
        size_issues = []
        for i, line in enumerate(lines, 1):
            if 'Rectangle(' in line and 'width=' not in line:
                size_issues.append(
                    f'Line {i}: Rectangle without explicit width/height - {line.strip()}'
                )
            if 'Square(' in line and 'side_length=' not in line:
                size_issues.append(
                    f'Line {i}: Square without explicit side_length - {line.strip()}'
                )
            if 'RoundedRectangle(' in line and 'width=' not in line:
                size_issues.append(
                    f'Line {i}: RoundedRectangle without explicit dimensions - {line.strip()}'
                )
        
        if size_issues:
            crowding_issues.extend(size_issues)

        if crowding_issues:
            issues.append('Layout Crowding:')
            issues.extend(f'   • {c}' for c in crowding_issues)

        # Check code complexity
        complexity_issues = []

        if 'Helper' in code or 'helper' in code.lower():
            helper_count = code.count('Helper') + code.count('helper')
            complexity_issues.append(
                f'Using Helper class pattern ({helper_count} occurrences), consider simplification'
            )

        create_methods = code.count('def create_')
        if create_methods > 4:  # Too many create methods
            complexity_issues.append(
                f'Too many create methods ({create_methods}), consider merging related functions'
            )

        # Check VGroup nesting depth
        vgroup_depth = 0
        max_depth = 0
        for line in lines:
            if 'VGroup(' in line:
                vgroup_depth += line.count('VGroup(')
                max_depth = max(max_depth, vgroup_depth)
            if ')' in line:
                vgroup_depth = max(0, vgroup_depth - line.count(')'))

        if max_depth > 3:
            complexity_issues.append(
                f'VGroup nesting too deep ({max_depth} levels), consider simplifying structure'
            )

        if complexity_issues:
            issues.append('Complexity Issues:')
            issues.extend(f'   • {c}' for c in complexity_issues)

        # Animation timing check
        animation_issues = []

        # Animation time too long will drag the rhythm
        runtime_values = re.findall(r'run_time\s*=\s*([0-9.]+)', code)
        for runtime in runtime_values:
            if float(runtime) > 6.0:
                animation_issues.append(
                    f'Animation duration too long ({runtime}s), may affect rhythm'
                )

        wait_values = re.findall(r'self\.wait\(([0-9.]+)\)', code)
        for wait_time in wait_values:
            if float(wait_time) > 4.0:
                animation_issues.append(
                    f'Wait time too long ({wait_time}s), may affect continuity'
                )

        if animation_issues:
            issues.append('Animation Timing Issues:')
            issues.extend(f'   • {a}' for a in animation_issues)

        return issues

    @staticmethod
    def _extract_spatial_operations(lines: List[str]) -> List[Dict]:
        operations = []

        for i, line in enumerate(lines, 1):
            line_clean = line.strip()
            if not line_clean or line_clean.startswith('#'):
                continue

            # Extract object name
            object_name = FixManimCode._extract_object_name(line_clean)
            if not object_name:
                continue

            operation = {
                'line': i,
                'code': line_clean,
                'object': object_name,
                'type': 'unknown',
                'reference': None,
                'has_spacing': False,
                'spacing_value': None
            }

            # Analyze positioning type
            if '.move_to(' in line:
                operation['type'] = 'move_to'
                operation['reference'] = FixManimCode._extract_move_to_target(
                    line_clean)
            elif '.next_to(' in line:
                operation['type'] = 'next_to'
                operation[
                    'reference'] = FixManimCode._extract_next_to_reference(
                        line_clean)
                operation['has_spacing'] = 'buff=' in line
                operation['spacing_value'] = FixManimCode._extract_buff_value(
                    line_clean)
            elif '.center()' in line:
                operation['type'] = 'center'
                operation['reference'] = 'ORIGIN'
            elif '.to_edge(' in line:
                operation['type'] = 'to_edge'
                operation['reference'] = FixManimCode._extract_edge_direction(
                    line_clean)
            elif '.shift(' in line:
                operation['type'] = 'shift'
                operation['reference'] = 'relative'

            if operation['type'] != 'unknown':
                operations.append(operation)

        return operations

    @staticmethod
    def _extract_edge_direction(line):
        match = re.search(r'\.to_edge\(([^)]+)\)', line)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_buff_value(line):
        match = re.search(r'buff\s*=\s*([0-9.]+)', line)
        return float(match.group(1)) if match else None

    @staticmethod
    def _extract_next_to_reference(line):
        match = re.search(r'\.next_to\(([^,)]+)', line)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_object_name(line):
        # Handle assignment statements
        if '=' in line and not line.strip().startswith('#'):
            # Extract variable name on the left side of equals sign
            var_match = re.search(r'(\w+)\s*=', line)
            if var_match and any(method in line for method in [
                    '.move_to(', '.next_to(', '.center()', '.to_edge(',
                    '.shift('
            ]):
                return var_match.group(1)

        # Handle direct object.method() pattern
        method_match = re.search(
            r'(\w+)\.(?:move_to|next_to|center|to_edge|shift)\(', line)
        if method_match:
            obj_name = method_match.group(1)
            # Ensure valid identifier (alphanumeric and underscore)
            if obj_name and not obj_name[0].isdigit() and obj_name.replace(
                    '_', '').isalnum():
                return obj_name

        return None

    @staticmethod
    def _extract_move_to_target(line):
        match = re.search(r'\.move_to\(([^)]+)\)', line)
        if match:
            target = match.group(1).strip()

            get_methods = [
                '.get_left()', '.get_right()', '.get_top()', '.get_bottom()',
                '.get_center()'
            ]

            # Handle method calls, extract base object name
            for method in get_methods:
                if method in target:
                    base_obj = target.replace(method, '').strip()
                    return base_obj if base_obj else target

            # Remove brackets and whitespace
            target = target.replace('[', '').replace(']', '').strip()
            return target if target else None
        return None

    @staticmethod
    def _analyze_spatial_conflicts(operations):
        conflicts = []

        # 1. Detect same reference point conflicts
        reference_groups = {}
        for op in operations:
            if op['reference'] and op['reference'] != 'relative':
                if op['reference'] not in reference_groups:
                    reference_groups[op['reference']] = []
                reference_groups[op['reference']].append(op)

        for ref, ops_list in reference_groups.items():
            if len(ops_list) > 1:
                conflicts.extend(
                    FixManimCode._detect_reference_conflicts(ref, ops_list))

        # 2. Detect missing spacing risk
        for op in operations:
            if op['type'] == 'next_to' and not op['has_spacing']:
                conflicts.append({
                    'type':
                    'missing_spacing',
                    'severity':
                    'medium',
                    'operation':
                    op,
                    'description':
                    'Missing buff parameter may cause overlap'
                })

        # 3. Detect insufficient spacing
        for op in operations:
            if op['spacing_value'] is not None and op['spacing_value'] < 0.2:
                conflicts.append({
                    'type':
                    'insufficient_spacing',
                    'severity':
                    'medium',
                    'operation':
                    op,
                    'description':
                    f"Spacing too small ({op['spacing_value']}) may cause overlap"
                })

        # 4. Detect object-geometry overlap risk
        conflicts.extend(FixManimCode._detect_geometry_conflicts(operations))

        return conflicts

    @staticmethod
    def _is_text_object(obj_name):
        if not obj_name:
            return False
        return any(keyword in obj_name.lower()
                   for keyword in ['text', 'label', 'title'])

    @staticmethod
    def _is_geometry_reference(ref):
        if not ref:
            return False
        ref_lower = ref.lower()
        geometry_keywords = [
            'line', 'circle', 'square', 'rectangle', 'triangle', 'polygon',
            'arc', 'ellipse'
        ]
        return any(keyword in ref_lower for keyword in geometry_keywords)

    @staticmethod
    def _is_label_object(obj_name):
        if not obj_name:
            return False
        return 'label' in obj_name.lower() or obj_name.endswith('_text')

    @staticmethod
    def _detect_geometry_conflicts(operations):
        conflicts = []

        for op in operations:
            if op['type'] == 'move_to' and op['reference']:
                ref = op['reference']

                # Detect text moving directly to geometry object
                if FixManimCode._is_text_object(
                        op['object']) and FixManimCode._is_geometry_reference(
                            ref):
                    conflicts.append({
                        'type':
                        'text_geometry_overlap',
                        'severity':
                        'high',
                        'operation':
                        op,
                        'description':
                        f"Text object ({op['object']}) moving directly to geometry object ({ref}) position"
                    })

                # Detect label moving directly to object
                elif FixManimCode._is_label_object(
                        op['object']
                ) and not FixManimCode._is_safe_reference(ref):
                    conflicts.append({
                        'type':
                        'label_object_overlap',
                        'severity':
                        'medium',
                        'operation':
                        op,
                        'description':
                        f"Label object ({op['object']}) may overlap with target object ({ref})"
                    })

        return conflicts

    @staticmethod
    def _is_safe_reference(ref):
        if not ref:
            return False
        safe_patterns = [
            'get_center()', 'ORIGIN', 'UP', 'DOWN', 'LEFT', 'RIGHT'
        ]
        return any(pattern in ref for pattern in safe_patterns)

    @staticmethod
    def _detect_reference_conflicts(reference, operations):
        conflicts = []

        # Group by positioning type
        move_to_ops = [op for op in operations if op['type'] == 'move_to']
        center_ops = [op for op in operations if op['type'] == 'center']

        # Multiple move_to to same object
        if len(move_to_ops) > 1:
            conflicts.append({
                'type':
                'multiple_move_to',
                'severity':
                'high',
                'reference':
                reference,
                'operations':
                move_to_ops,
                'description':
                f'Multiple objects moving to same position ({reference})'
            })

        # Multiple center calls
        if len(center_ops) > 1:
            conflicts.append({
                'type':
                'multiple_center',
                'severity':
                'high',
                'operations':
                center_ops,
                'description':
                'Multiple objects using center() positioning'
            })

        return conflicts

    @staticmethod
    def _format_conflict_description(conflict):
        op = conflict.get('operation', {})
        line = op.get('line', '?')

        if conflict['type'] in [
                'missing_spacing', 'insufficient_spacing',
                'text_geometry_overlap', 'label_object_overlap'
        ]:
            return f"Line {line}: {conflict['description']} - {op.get('code', '')}"
        elif conflict['type'] == 'multiple_move_to':
            objects = [op['object'] for op in conflict['operations']]
            return f"Multiple objects moving to same position ({conflict['reference']}): {', '.join(objects)}"
        elif conflict['type'] == 'multiple_center':
            objects = [op['object'] for op in conflict['operations']]
            return f"Multiple objects using center() positioning: {', '.join(objects)}"
        else:
            return conflict.get('description', 'Unknown conflict')