# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union

import json
from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.utils import get_logger
from omegaconf import DictConfig
from PIL import Image

logger = get_logger()


class GenerateRemotionCode(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.num_parallel = getattr(self.config, 'llm_num_parallel', 10)
        self.images_dir = os.path.join(self.work_dir, 'images')
        self.remotion_code_dir = os.path.join(self.work_dir, 'remotion_code')
        os.makedirs(self.remotion_code_dir, exist_ok=True)

    async def execute_code(self, messages: Union[str, List[Message]],
                           **kwargs) -> List[Message]:
        with open(os.path.join(self.work_dir, 'segments.txt'), 'r') as f:
            segments = json.load(f)
        with open(os.path.join(self.work_dir, 'audio_info.txt'), 'r') as f:
            audio_infos = json.load(f)
        logger.info('Generating remotion code.')

        tasks = []
        for i, (segment, audio_info) in enumerate(zip(segments, audio_infos)):
            # "remotion" field takes precedence, fall back to "manim"
            animation_requirement = segment.get('remotion',
                                                segment.get('manim'))

            # Load visual plan if available
            visual_plan_path = os.path.join(self.work_dir, 'visual_plans',
                                            f'plan_{i+1}.json')
            visual_plan = {}
            if os.path.exists(visual_plan_path):
                try:
                    with open(visual_plan_path, 'r', encoding='utf-8') as f:
                        visual_plan = json.load(f)
                except Exception as e:
                    logger.warning(
                        f'Failed to load visual plan for segment {i+1}: {e}')
            else:
                # Robustness: if step5 failed to persist the plan, synthesize a minimal one
                # from the existing storyboard/manim requirement so downstream guidance exists.
                try:
                    os.makedirs(
                        os.path.dirname(visual_plan_path), exist_ok=True)

                    visual_plan = GenerateRemotionCode._synthesize_visual_plan_from_segment(
                        segment)
                    with open(visual_plan_path, 'w', encoding='utf-8') as f:
                        json.dump(visual_plan, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.warning(
                        f'Failed to synthesize visual plan for segment {i+1}: {e}'
                    )

            if animation_requirement is not None:
                tasks.append(
                    (segment, audio_info['audio_duration'], i, visual_plan))

        remotion_code = [''] * len(segments)

        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = {
                executor.submit(self._generate_remotion_code_static, seg, dur,
                                idx, self.config, self.images_dir, v_plan): idx
                for seg, dur, idx, v_plan in tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                remotion_code[idx] = future.result()

        for i, code in enumerate(remotion_code):
            remotion_file = os.path.join(self.remotion_code_dir,
                                         f'Segment{i + 1}.tsx')
            with open(remotion_file, 'w', encoding='utf-8') as f:
                f.write(code)
        return messages

    @staticmethod
    def _generate_remotion_code_static(segment, audio_duration, i, config,
                                       image_dir, visual_plan):
        """Static method for multiprocessing"""
        llm = LLM.from_config(config)
        return GenerateRemotionCode._generate_remotion_impl(
            llm, segment, audio_duration, i, image_dir, config, visual_plan)

    @staticmethod
    def get_image_size(filename):
        with Image.open(filename) as img:
            return f'{img.width}x{img.height}'

    @staticmethod
    def get_all_images_info(segment, i, image_dir):
        all_images_info = []

        foreground = segment.get('foreground', [])

        # Fallback: Check for existing foreground images even if not in segment info
        if not foreground:
            pattern = os.path.join(image_dir,
                                   f'illustration_{i + 1}_foreground_*.png')
            found_files = sorted(glob.glob(pattern))
            for fpath in found_files:
                # Extract index from filename to match expected structure if needed,
                # or just treat as a foreground image.
                # Filename format: illustration_{i+1}_foreground_{idx+1}.png
                try:
                    # Try to find a description file
                    base_name = os.path.basename(fpath)
                    desc_name = base_name.replace('.png', '.txt')
                    desc_path = os.path.join(
                        os.path.dirname(image_dir), 'illustration_prompts',
                        desc_name)
                    description = 'Foreground element'
                    if os.path.exists(desc_path):
                        with open(desc_path, 'r', encoding='utf-8') as df:
                            description = df.read().strip()

                    size = GenerateRemotionCode.get_image_size(fpath)
                    image_info = {
                        'filename': base_name,
                        'size': size,
                        'description': description,
                    }
                    all_images_info.append(image_info)
                except Exception as e:
                    logger.warning(
                        f'Error processing fallback image {fpath}: {e}')

        for idx, _req in enumerate(foreground):
            foreground_image = os.path.join(
                image_dir, f'illustration_{i + 1}_foreground_{idx + 1}.png')
            if os.path.exists(foreground_image):
                size = GenerateRemotionCode.get_image_size(foreground_image)
                image_info = {
                    'filename': os.path.basename(
                        foreground_image),  # Use basename for Remotion
                    'size': size,
                    'description': _req,
                }
                all_images_info.append(image_info)

        image_info_file = os.path.join(
            os.path.dirname(image_dir), 'image_info.txt')
        if os.path.exists(image_info_file):
            with open(image_info_file, 'r') as f:
                for line in f.readlines():
                    if not line.strip():
                        continue
                    image_info = json.loads(line)
                    if image_info['filename'] in segment.get('user_image', []):
                        all_images_info.append(image_info)
        return all_images_info

    @staticmethod
    def _synthesize_visual_plan_from_segment(segment: dict) -> dict:
        """Best-effort plan synthesis when Visual Director plan files are missing.

        This keeps the pipeline deterministic and ensures the Remotion generator receives
        explicit beats/layout guidance even if step5 output is unavailable.
        """
        # "remotion" field takes precedence, fall back to "manim"
        animation_req = (segment.get('remotion') or segment.get('manim')
                         or '').strip()

        # Heuristic layout detection
        req_lower = animation_req.lower()
        if 'three-object' in req_lower or 'three object' in req_lower or 'left-middle-right' in req_lower:
            layout = 'Grid Layout'
        elif 'two-object' in req_lower or 'two object' in req_lower or 'left-right' in req_lower:
            layout = 'Asymmetrical Balance'
        else:
            layout = 'Center Focus'

        # Required short labels are often quoted in the animation requirement.
        # Keep them short to avoid subtitle-like paragraphs.
        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", animation_req)
        required_labels: List[str] = []
        for a, b in quoted:
            s = (a or b or '').strip()
            if 1 <= len(s) <= 10 and re.search(r'[\u4e00-\u9fff]', s):
                required_labels.append(s)
        required_labels = list(dict.fromkeys(required_labels))[:2]

        label_hint = (f"Required label(s): {', '.join(required_labels)}"
                      if required_labels else '(No required label detected)')

        return {
            'background_concept':
            'Clean cinematic backdrop with empty center safe-area',
            'foreground_assets': [],
            'layout_composition':
            layout,
            'text_placement':
            'Centered keyword card within SAFE container',
            'visual_metaphor':
            f'{label_hint}. Map narration to 1-3 simple visual elements, no random effects.',
            'beats': [
                '0-25%: Establish the main visual(s) according to the requirement.',
                '25-80%: Add 1 supporting change/connection that matches the narration.',
                '80-100%: Resolve with the keyword/required label on a high-contrast card.',
            ],
            'motion_guide':
            'Use spring-based entrances + subtle camera drift; sequence elements by beats; avoid chaos.',
        }

    @staticmethod
    def _generate_remotion_impl(llm, segment, audio_duration, i, image_dir,
                                config, visual_plan):
        component_name = f'Segment{i + 1}'
        content = segment['content']
        # "remotion" field takes precedence, fall back to "manim"
        animation_requirement = segment.get('remotion',
                                            segment.get('manim', ''))
        images_info = GenerateRemotionCode.get_all_images_info(
            segment, i, image_dir)

        # Inject image info with code snippets.
        images_info_str = ''
        if images_info:
            images_info_str += 'Available Images (You MUST use these exact import/usage codes):\n'
            for img in images_info:
                fname = img['filename']
                w, h = img['size'].split('x')
                is_portrait = int(h) > int(w)
                style_hint = "maxHeight: '80%'" if is_portrait else "maxWidth: '80%'"

                images_info_str += f"- Name: {fname} ({img['size']}, {img['description']})\n"
                images_info_str += (
                    f"  USAGE CODE: <Img src={{staticFile(\"images/{fname}\")}} "
                    f"style={{{{ {style_hint}, objectFit: 'contain' }}}} />\n")
        else:
            images_info_str = 'No images offered. Use CSS shapes/Text only.'

        beats = visual_plan.get('beats', []) if visual_plan else []

        motion_guide_section = ''
        # Updated Visual Director logic
        # We now look for the new format from Step 5 (Visual Director).
        timeline_events = visual_plan.get('timeline_events', [])
        layout_mode = visual_plan.get(
            'layout_mode', visual_plan.get('layout_composition',
                                           'Center Focus'))

        timeline_text = ''
        if timeline_events:
            timeline_text = '**TIMELINE EXECUTION (Exact Timings)**:\n'
            for ev in timeline_events:
                t_str = f"{int(ev.get('time_percent', 0)*100)}%"
                timeline_text += f"- At {t_str} of video: {ev.get('action')}\n"

        metaphor_expl = visual_plan.get('visual_metaphor_explanation', '')
        asset_desc = visual_plan.get('main_visual_asset',
                                     {}).get('description', '')

        motion_guide_section = f"""
**VISUAL DIRECTOR'S BLUEPRINT (MANDATORY)**:
You are the animator. You MUST follow the Director's blueprint below.

0.  **SCENE CONTEXT**:
    *   Metaphor: {metaphor_expl}
    *   Main Asset Focus: {asset_desc}

1.  **LAYOUT MODE**: {layout_mode}
    *   Setup your CSS layout immediately based on this mode.
    *   **STABILITY FIRST**: Use standard Flexbox layouts (Row/Column).
        Avoid absolute positioning unless necessary.

2.  **TIMELINE & ACTION**:
{timeline_text or beats}
    *   **TIMING**: Use the exact times provided.

3.  **ENGINEERING RULES (VIOLATION = FAIL)**:
    *   **LAYOUT SYSTEM (ANTI-OVERLAP)**:
        *   **MANDATORY**: Use `display: 'flex'` and `flexDirection: 'column'`
            (or row) for the main container layout.
        *   **FORBIDDEN**: Do NOT place text and images both at
            `position: 'absolute', top: '50%', left: '50%'`. They WILL collide.
        *   **STRATEGY**:
            - Create a `FlexContainer` with `justifyContent: 'center', alignItems: 'center', gap: 50`.
            - Put Text in one logic block, Images in another.
        *   **SAFE AREA**: Wrap everything in a `<div style={{ width: '85%', height: '85%' }}>`.
            Never touch edges.

    *   **ASSET & TEXT VISIBILITY**:
        *   **Text on Images**: If text MUST overlap an image, the text container MUST have
            `backgroundColor: 'rgba(255,255,255,0.9)'` (if black text)
            or `rgba(0,0,0,0.7)` (if white text).
        *   **Z-Index**: Always set `zIndex: 10` for Text and `zIndex: 1` for Images.
        *   **Font Size**: Minimum `40px` for titles, `24px` for labels.

    *   **ASSET OVERLOAD PROTECTION**:
        *   If you have **3 or more images**:
            *   **MANDATORY**: Use a Grid layout (`display: 'grid', gridTemplateColumns: '1fr 1fr'`)
                or Flex Wrap.
            *   **SCALE DOWN**: Force image heights to max `250px`.
            *   If too many images for one row, wrap to a second row.

    *   **NO FULLSCREEN BACKGROUNDS**:
        *   The root container **MUST** be transparent. No `backgroundColor: 'white'`.
        *   We will composite a background later.
"""

        if config.foreground == 'image':
            image_usage = f"""**Image usage (CRITICAL: THESE ARE ASSETS, NOT BACKGROUNDS)**
- You'll receive an actual image list with three fields per image: filename, size, and description.
- Images will be placed in the `public/images` folder. You can reference them using
  `staticFile("images/filename")` or just string path if using `Img` tag with `src`.
- **THESE IMAGES ARE ISOLATED ELEMENTS** (e.g., a single icon, a character, a prop).
- **DO NOT** stretch them to fill the screen like a background wallpaper.
- **DO** position them creatively:
    *   Float them in 3D space.
    *   Slide them in from the side.
    *   Scale them up/down with `spring`.
    *   Use them as icons next to text.
- Pay attention to the size field, write Remotion code that respects the image's aspect ratio.
- IMPORTANT: If images files are not empty, **you MUST use them all**.
  These are custom-generated assets for this specific scene.
    *   If the image is a character or object, place it in the foreground.
    *   If you are unsure where to put it, center it and fade it in.
    *   **FAILURE TO USE PROVIDED IMAGES IS A CRITICAL ERROR.**
    *   Here is the image files list:

{images_info_str}

**CRITICAL WARNING**:
- **DO NOT HALLUCINATE IMAGES**. You MUST ONLY use the filenames listed above.
- If the list above is "No images offered.", you **MUST NOT** use any `Img` tags or `staticFile` calls.
  Use CSS shapes, colors, and text only.
- Do not invent filenames like "book.png", "city.png", etc. if they are not in the list.
- **FORBIDDEN BACKGROUND FILES**: Do NOT use any filename matching `illustration_*_background.png` even if it exists.
- **FORBIDDEN FULL-SCREEN BACKGROUND**: Never render a full-screen `Img` background.
  This pipeline composites background later.
- DO NOT let the image and the text/elements overlap. Reorganize them in your animation.
"""
        else:
            image_usage = ''

        prompt = f"""You are a **Senior Motion Graphics Designer** and **Instructional Designer**,
    creating high-end, cinematic, and beautiful educational animations using React (Remotion).
Your goal is to create a visual experience that complements the narration, NOT just subtitles on a screen.

**Task**: Create a Remotion component
- Component name: {component_name}
- Content (Narration): {content}
- Duration: {audio_duration} seconds
- Code language: **TypeScript (React)**

{image_usage}

- 如果图片存在，你需要使用所有的图片，图片需要放置在屏幕显眼的位置，不要放置在角落
- 你的动画时长需要符合Duration的要求
- 你的动画需要符合Content原始需求，尽量高雅，不允许使用火柴人
- 保证所有元素不重叠，不被屏幕边缘切分
- 你的屏幕是16:9的


Please create Remotion code that meets the above requirements and creates a visually stunning animation.
"""

        logger.info(f'Generating remotion code for: {content}')
        _response_message = llm.generate(
            [Message(role='user', content=prompt)], temperature=0.3)
        response = _response_message.content

        # Robust code extraction using regex
        code_match = re.search(
            r'```(?:typescript|tsx|js|javascript)?\s*(.*?)```', response,
            re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            # Fallback: if no code blocks, assume the whole response is code
            # but try to strip leading/trailing text if it looks like markdown
            code = response
            if 'import React' in code:

                # Try to find the start of the code
                idx = code.find('import React')
                code = code[idx:]

        code = code.strip()

        def fix_easing_syntax(code: str) -> str:
            pattern = r'Easing\.(\w+)\(Easing\.(\w+)\}\)'
            replacement = r'Easing.\1(Easing.\2)'

            return re.sub(pattern, replacement, code)

        code = fix_easing_syntax(code)

        # Post-process for offline Windows compatibility (deterministic safety net)
        code = GenerateRemotionCode._strip_external_font_loading(code)
        return code

    @staticmethod
    def _strip_external_font_loading(code: str) -> str:
        """Remove external font loading patterns that break offline environments.

        If the LLM imports `@remotion/fonts` and calls `loadFont()` with a Google Fonts CSS URL,
        Remotion will crash at runtime while evaluating compositions.
        We remove the import and top-level loadFont() calls. Keeping `fontFamily` styles is safe.
        """
        try:
            if 'fonts.googleapis.com' not in code and 'fonts.gstatic.com' not in code and '@remotion/fonts' not in code:
                return code

            # Remove import of loadFont
            code = re.sub(
                r"^\s*import\s*\{\s*loadFont\s*\}\s*from\s*['\"]@remotion/fonts['\"];\s*\n",
                '',
                code,
                flags=re.MULTILINE,
            )

            # Remove any top-level loadFont({...}); blocks (best-effort)
            code = re.sub(
                r'^\s*loadFont\(\{[\s\S]*?\}\);\s*\n\s*\n',
                '',
                code,
                flags=re.MULTILINE,
            )
            code = re.sub(
                r'^\s*//\s*Load.*\n\s*loadFont\(\{[\s\S]*?\}\);\s*\n\s*\n',
                '',
                code,
                flags=re.MULTILINE,
            )

            # As an extra guard, replace any remaining google font URLs with empty string
            code = code.replace('https://fonts.googleapis.com/', '')
            code = code.replace('https://fonts.gstatic.com/', '')
            return code
        except Exception:
            return code
