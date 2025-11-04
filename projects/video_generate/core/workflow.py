import os
import re
import shutil
import subprocess
import sys
import time
import uuid

import json
import numpy as np
from openai import OpenAI
from PIL import Image


def clean_content(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'【/?[^】]+】', '', text)


video_agent_dir = os.path.dirname(os.path.abspath(__file__))
if video_agent_dir not in sys.path:
    sys.path.insert(0, video_agent_dir)

# 导入增强质量保证系统
try:
    from .enhanced_quality_system import (VisualQualityAssessment,
                                          AnimationContentMatcher)
    QUALITY_SYSTEM_AVAILABLE = True
    print('成功导入增强质量保证系统')
except ImportError as e:
    QUALITY_SYSTEM_AVAILABLE = False
    print(f'质量保证系统导入失败: {e}')

# 导入增强提示词系统
try:
    from .enhanced_manim_prompts import EnhancedManimPromptSystem
    ENHANCED_PROMPTS_AVAILABLE = True
    print('成功导入增强提示词系统')
except ImportError as e:
    ENHANCED_PROMPTS_AVAILABLE = False
    print(f'增强提示词系统导入失败: {e}')

# 导入新的质量控制系统
try:
    from .manim_quality_controller import ManimQualityController
    from .optimized_manim_prompts import OptimizedManimPrompts
    OPTIMIZED_QUALITY_AVAILABLE = True
    print('成功导入优化质量控制系统')
except ImportError as e:
    OPTIMIZED_QUALITY_AVAILABLE = False
    print(f'优化质量控制系统导入失败: {e}')

# 导入背景图生成工具类
try:
    from .background_image import BackgroundImageGenerator
    BACKGROUNDIMAGE_AVAILABLE = True
    print('成功导入背景图生成器')
except ImportError as e:
    print(f'无法导入背景图生成器: {e}')
    BACKGROUNDIMAGE_AVAILABLE = False

# 导入平衡空间约束系统
try:
    from .balanced_spatial_system import BalancedSpatialSystem
    BALANCED_SPATIAL_AVAILABLE = True
    print('成功导入平衡空间约束系统')
except ImportError as e:
    BALANCED_SPATIAL_AVAILABLE = False
    print(f'平衡空间约束系统导入失败: {e}')

# 导入新的动画制作模式系统
try:
    from .animation_production_modes import (AnimationProductionMode,
                                             AnimationStatus, AnimationTask,
                                             AnimationTaskManager,
                                             PlaceholderGenerator)
    # 延迟导入 AnimationStudio 避免循环依赖
    HUMAN_ANIMATION_AVAILABLE = True
    print('成功导入人工控制动画制作系统')
except ImportError as e:
    HUMAN_ANIMATION_AVAILABLE = False
    print(f'人工控制动画制作系统导入失败: {e}')

# 魔搭模型配置
MODAI_TOKEN = os.environ.get('MODELSCOPE_API_KEY')
if not os.environ.get('MODELSCOPE_API_KEY'):
    print('使用内置API密钥')

OPENAI_CLIENT = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key=MODAI_TOKEN,
)


def fix_common_manim_issues(code):
    """
    修复常见的 Manim 代码问题
    """
    if not code:
        return code

    # 修复TRANSPARENT常量问题
    if 'TRANSPARENT' in code and 'import' not in code.split(
            'TRANSPARENT')[0].split('\n')[-1]:
        # TRANSPARENT不是从manim导入的，改成透明背景的正确写法
        code = code.replace('= TRANSPARENT', '= "#00000000"')  # 透明背景的 RGBA 表示
        code = code.replace('(TRANSPARENT)', '("#00000000")')
        code = code.replace(' TRANSPARENT ', ' "#00000000" ')

    return code


def clean_llm_code_output(code):
    """
    清理LLM输出中的Markdown格式标记和多余内容，智能提取纯Python代码
    """
    if not code:
        return code

    # 首先尝试从markdown代码块中提取代码
    code = extract_python_code_from_markdown(code)

    # 修复常见的 Manim 错误
    code = fix_common_manim_issues(code)

    # 移除首尾空白字符
    code = code.strip()

    # 再次清理可能残留的markdown标记
    if code.startswith('```python'):
        code = code[9:].strip()
    elif code.startswith('```'):
        code = code[3:].strip()

    if code.endswith('```'):
        code = code[:-3].strip()

    # 按行处理，移除非Python代码行
    lines = code.split('\n')
    cleaned_lines = []
    in_python_code = False

    key_words = [
        'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except',
        'finally:', 'with ', 'self.', 'return', 'import', 'from'
    ]

    for line in lines:
        stripped_line = line.strip()

        # 空行处理：代码块里的要保留，外面的跳过
        if not stripped_line:
            if in_python_code:
                cleaned_lines.append(line)
            continue

        # 检测是否为Python代码行
        if is_python_code_line(stripped_line):
            in_python_code = True
            cleaned_lines.append(line)
        elif in_python_code and (stripped_line.startswith(' ')
                                 or stripped_line.startswith('\t')):
            # 缩进行，可能是代码的一部分
            cleaned_lines.append(line)
        elif stripped_line.startswith(
                '#') and not contains_chinese(stripped_line):
            # 英文注释保留
            cleaned_lines.append(line)
        elif in_python_code and any(
                stripped_line.startswith(kw) for kw in key_words):
            # Python关键字行
            cleaned_lines.append(line)
        elif contains_chinese(
                stripped_line) and not is_python_code_line(stripped_line):
            # 包含中文且不是Python代码的行，跳过
            continue
        elif stripped_line.startswith(
                '```'
        ) or '修复' in stripped_line or '说明' in stripped_line or '问题' in stripped_line or stripped_line == '`':
            # 明显的markdown或说明文字，跳过
            continue
        else:
            # 其他可能的代码行
            if in_python_code:
                cleaned_lines.append(line)

    result = '\n'.join(cleaned_lines).strip()

    # 最后清理可能残留的markdown字符
    while result.endswith('`') or result.endswith('\\'):
        if result.endswith('`'):
            result = result[:-1].strip()
        elif result.endswith('\\'):
            result = result[:-1].strip()

    return result


def extract_python_code_from_markdown(text):
    """
    从markdown格式文本中提取Python代码块
    """
    import re

    # 匹配```python...```代码块
    python_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
    if python_blocks:
        return python_blocks[0]

    # 匹配```...```代码块
    code_blocks = re.findall(r'```\n(.*?)\n```', text, re.DOTALL)
    if code_blocks:
        # 选择最可能是Python代码的块（包含from manim import等）
        for block in code_blocks:
            if 'from manim import' in block or 'class Scene' in block:
                return block
        return code_blocks[0]

    # 匹配没有换行的```代码块
    simple_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    if simple_blocks:
        for block in simple_blocks:
            if 'from manim import' in block or 'class Scene' in block:
                return block
        return simple_blocks[0]

    # 如果没找到代码块，返回原文本
    return text


def contains_chinese(text):
    """
    检测文本是否包含中文字符
    """
    import re
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def is_python_code_line(line):
    """
    判断一行是否为Python代码
    """
    line = line.strip()
    if not line:
        return False

    # Python关键字和常见语法
    python_indicators = [
        'from ', 'import ', 'def ', 'class ', 'if ', 'elif ', 'else:', 'for ',
        'while ', 'try:', 'except', 'finally:', 'with ', 'return', 'yield',
        'break', 'continue', 'pass', 'raise', 'self.', '= ', '== ', '!= ',
        '< ', '> ', '<= ', '>= ', 'and ', 'or ', 'not ', 'in ', 'is ',
        'lambda', '__init__', '__str__', '__repr__'
    ]

    # 检查是否以Python语法开始
    if any(line.startswith(indicator) for indicator in python_indicators):
        return True

    # 检查是否包含Python语法
    if any(indicator in line for indicator in
           ['()', '[]', '{}', ' = ', 'self.', 'def ', 'class ']):
        return True

    # 如果包含中文，很可能不是代码
    if contains_chinese(line):
        return False

    # 检查是否为赋值、函数调用等
    if '=' in line or '(' in line and ')' in line:
        return True

    return False


# 音频失败的回退方案


# 二次检查和增强动画内容
def optimize_animation(segment_content, segment_type, main_theme,
                       context_segments, total_segments, segment_index):
    """
    智能动画优化器 - 对动画内容进行二次检查和优化
    """

    print(f'启动智能动画优化器 - 段落{segment_index + 1}')
    # 构建上下文信息
    prev_context = ''
    next_context = ''
    if segment_index > 0:
        prev_segments = context_segments[max(0, segment_index
                                             - 2):segment_index]
        prev_context = ' '.join(
            [seg.get('content', '') for seg in prev_segments])

    if segment_index < len(total_segments) - 1:
        next_segments = context_segments[segment_index + 1:segment_index + 3]
        next_context = ' '.join(
            [seg.get('content', '') for seg in next_segments])

    optimization_prompt = f"""你是顶级的科普教育动画导演，请对以下动画段落进行智能分析和优化建议：

**基本信息**：
- 主题：{main_theme}
- 当前段落类型：{segment_type}
- 当前内容：{segment_content}
- 段落位置：第{segment_index + 1}段 / 共{len(total_segments)}段

**上下文**：
- 前文内容：{prev_context[-200:] if prev_context else '无'}
- 后文内容：{next_context[:200] if next_context else '无'}

**优化任务**：
1. **内容分析**：分析这段内容的核心概念、情感色彩、教学价值
2. **动画建议**：基于内容特点和上下文，建议最合适的动画元素和视觉效果
3. **文案优化**：如果文案不够生动或有问题，提出优化建议
4. **主题呼应**：确保与整体主题{main_theme}保持一致

请以JSON格式返回优化建议：
{{
    "content_analysis": {{
        "core_concepts": ["概念1", "概念2"],
        "emotional_tone": "幽默/严肃/激动等",
        "teaching_value": "教学价值描述",
        "visual_opportunities": ["可视化机会1", "可视化机会2"]
    }},
    "animation_recommendations": {{
        "primary_elements": ["主要动画元素"],
        "visual_effects": ["视觉效果"],
        "color_scheme": "建议色彩方案",
        "animation_style": "动画风格建议",
        "timing_suggestions": "时间节奏建议"
    }},
    "script_optimization": {{
        "needs_improvement": true/false,
        "optimized_content": "优化后的文案（如果需要）",
        "improvement_reasons": ["改进原因"]
    }},
    "context_integration": {{
        "connects_to_previous": "与前文的连接点",
        "prepares_for_next": "为后文的铺垫",
        "theme_alignment": "与主题的呼应"
    }}
}}"""

    try:
        result = modai_model_request(
            optimization_prompt,
            model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
            max_tokens=800,
            temperature=0.6)
        print(f'API返回原始结果: {result[:200]}...')

        # 尝试解析JSON
        import json

        # 清理可能的markdown格式
        if '```json' in result:
            result = result.split('```json')[1].split('```')[0].strip()
        elif '```' in result:
            result = result.split('```')[1].split('```')[0].strip()

        # 尝试修复常见的JSON格式问题
        if not result.startswith('{'):
            # 找到第一个{
            start_idx = result.find('{')
            if start_idx != -1:
                result = result[start_idx:]

        if not result.endswith('}'):
            # 找到最后一个}
            end_idx = result.rfind('}')
            if end_idx != -1:
                result = result[:end_idx + 1]

        # 检查是否有未闭合的字符串
        quote_count = result.count('"')
        if quote_count % 2 != 0:
            last_quote_pos = result.rfind('"')
            for i in range(last_quote_pos + 1, len(result)):
                if result[i] in [',', '}', ']']:
                    result = result[:i] + '"' + result[i:]
                    print(f'修复了未闭合的字符串，在位置 {i} 添加引号')
                    break

            if quote_count == result.count('"'):
                result = result.rstrip() + '"}'
                print('在末尾添加缺失的引号和括号')

        # 处理缺失的逗号和括号
        lines = result.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.endswith(('", ', '",', '"', '}', ']')):
                if '"' in line and not line.endswith('"'):
                    line = line + '"'
            cleaned_lines.append(line)
        result = '\n'.join(cleaned_lines)

        if not result.endswith('}'):
            result = result.rstrip().rstrip(',') + '\n    }\n}'
            print('添加缺失的结束结构')

        # 使用本地的JSON解析函数
        def extract_json_with_fallback(text, default_value):
            try:
                # 尝试从markdown代码块中提取JSON
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', text,
                                       re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = text.strip()
                return json.loads(json_str)
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                print(f'JSON解析失败，使用默认值: {e}')
                return default_value

        optimization_data = extract_json_with_fallback(
            result.strip(), {
                'content_analysis': {
                    'core_concepts': ['MCP TOOL CALLING', '工具调用'],
                    'emotional_tone': '幽默科普',
                    'teaching_value': '解释AI工具调用机制'
                },
                'animation_recommendations': {
                    'primary_style': '演示动画',
                    'visual_elements': ['工具图标', '调用流程'],
                    'timing_strategy': '节奏紧凑'
                }
            })

        core_concepts_count = len(
            optimization_data.get('content_analysis',
                                  {}).get('core_concepts', []))
        print(f'智能分析完成，发现 {core_concepts_count} 个核心概念')
        return optimization_data

    except Exception as e:
        print(f'分析过程异常: {e}')
        print(f'原始返回: {result[:500]}...')

        try:
            print('尝试强化JSON修复...')
            # 1: 尝试提取主要结构
            content_analysis_match = re.search(
                r'"content_analysis":\s*\{([^}]*)\}', result, re.DOTALL)
            animation_match = re.search(
                r'"animation_recommendations":\s*\{([^}]*)\}', result,
                re.DOTALL)

            if content_analysis_match or animation_match:
                print('通过正则表达式提取部分数据')
                # 构建基本的JSON结构
                repair_data = {
                    'content_analysis': {
                        'core_concepts': ['MCP TOOL CALLING', '工具调用'],
                        'emotional_tone': '幽默科普',
                        'teaching_value': '解释AI工具调用机制',
                        'visual_opportunities': ['流程图', '动画演示']
                    },
                    'animation_recommendations': {
                        'primary_elements': ['工具调用流程', 'API接口动画'],
                        'visual_effects': ['连接线动画', '数据流'],
                        'color_scheme': '科技蓝配橙色',
                        'animation_style': '现代扁平',
                        'timing_suggestions': '35-45秒节奏'
                    },
                    'script_optimization': {
                        'needs_improvement': False,
                        'optimized_content': segment_content,
                        'improvement_reasons': []
                    },
                    'context_integration': {
                        'connects_to_previous': 'AI基础能力',
                        'prepares_for_next': '实际应用案例',
                        'theme_alignment': 'MCP TOOL CALLING核心机制'
                    }
                }

                if content_analysis_match:
                    content_text = content_analysis_match.group(1)
                    concepts_match = re.search(r'"core_concepts":\s*\[(.*?)\]',
                                               content_text, re.DOTALL)
                    if concepts_match:
                        concepts_str = concepts_match.group(1)
                        concepts = re.findall(r'"([^"]*)"', concepts_str)
                        if concepts:
                            repair_data['content_analysis'][
                                'core_concepts'] = concepts[:4]
                print('JSON修复成功，使用部分提取的数据')
                return repair_data
        except Exception as repair_e:
            print(f'JSON修复也失败: {repair_e}')

        # 返回基础分析结果
        return {
            'content_analysis': {
                'core_concepts': ['概率预测', '语言模型'],
                'emotional_tone': '科普',
                'teaching_value': '解释AI工作原理',
                'visual_opportunities': ['公式展示', '概率图']
            },
            'animation_recommendations': {
                'primary_elements': ['公式动画', '概率展示'],
                'visual_effects': ['渐入', '高亮'],
                'color_scheme': '科技蓝色',
                'animation_style': '现代科普',
                'timing_suggestions': '稳定节奏'
            },
            'script_optimization': {
                'needs_improvement': False,
                'optimized_content': segment_content,
                'improvement_reasons': []
            },
            'context_integration': {
                'connects_to_previous': '训练过程',
                'prepares_for_next': '应用效果',
                'theme_alignment': 'AI核心原理'
            }
        }
    except Exception as e:
        print(f'智能优化失败: {e}')
        return {'error': str(e)}


def enhanced_script_and_animation_generator(original_content, content_type,
                                            main_theme, optimization_data,
                                            class_name):
    """
    基于优化建议生成增强的文案和动画代码
    """
    print('生成增强版文案和动画...')
    script_opt = optimization_data.get('script_optimization', {})
    anim_rec = optimization_data.get('animation_recommendations', {})
    content_analysis = optimization_data.get('content_analysis', {})

    if script_opt.get('needs_improvement', False):
        optimized_script = script_opt.get('optimized_content',
                                          original_content)
        print(f"文案已优化: {script_opt.get('improvement_reasons', [])}")
    else:
        optimized_script = original_content

    # 生成增强动画代码
    enhanced_animation_prompt = f"""你是顶级Manim动画专家，请基于以下详细分析创建震撼的科普教育动画：

**动画规格**：
- 类名：{class_name}
- 内容类型：{content_type}
- 主题：{main_theme}

**文案内容**：
{optimized_script}

**智能分析结果**：
- 核心概念：{content_analysis.get('core_concepts', [])}
- 情感色彩：{content_analysis.get('emotional_tone', '轻松科普')}
- 可视化机会：{content_analysis.get('visual_opportunities', [])}

**动画建议**：
- 主要元素：{anim_rec.get('primary_elements', [])}
- 视觉效果：{anim_rec.get('visual_effects', [])}
- 色彩方案：{anim_rec.get('color_scheme', '多彩生动')}
- 动画风格：{anim_rec.get('animation_style', '现代科普')}
- 时间节奏：{anim_rec.get('timing_suggestions', '舒缓流畅')}

**创作要求**：
1. **内容丰富**：充分体现文案中的所有精彩内容，不要简化
2. **视觉震撼**：使用多种动画效果、颜色渐变、粒子效果等
3. **教学清晰**：重点突出，层次分明，易于理解
4. **幽默生动**：体现文案的幽默感和生动性
5. **技术精湛**：使用高级Manim技术，避免简单展示

请生成完整的Manim代码，让这个动画成为教学视频中的亮点！"""

    try:
        enhanced_code = modai_model_request(
            enhanced_animation_prompt,
            model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
            max_tokens=1500,
            temperature=0.7)
        print('增强动画代码生成完成')
        return optimized_script, enhanced_code.strip()
    except Exception as e:
        print(f'增强动画生成失败: {e}')
        return optimized_script, ''


# 动画判断
def should_add_animation_elements(content, content_type, context_info=None):
    """
    智能判断是否需要添加动画元素，以及添加什么类型的元素
    """

    context_info = context_info or {}
    animation_elements = {
        'use_formula': False,
        'use_code': False,
        'use_chart': False,
        'use_diagram': False,
        'use_comparison': False,
        'use_emoji': False,
        'use_bubble': False,
        'suggested_elements': []
    }

    # 公式相关触发词
    formula_triggers = [
        '等于', '计算', '算法', '数学', '方程', '函数', '变量', '参数', '求解', '结果是'
    ]
    if any(trigger in content
           for trigger in formula_triggers) or '=' in content:
        animation_elements['use_formula'] = True
        animation_elements['suggested_elements'].append(
            'mathematical_notation')

    # 代码相关触发词
    code_triggers = ['程序', '代码', '编程', '函数', '变量', '算法实现', '代码示例', '编写', '运行']
    if any(trigger in content for trigger in code_triggers):
        animation_elements['use_code'] = True
        animation_elements['suggested_elements'].append('code_snippet')

    # 图表相关触发词
    chart_triggers = ['数据', '统计', '增长', '下降', '比较', '趋势', '占比', '百分比', '排行']
    if any(trigger in content for trigger in chart_triggers):
        animation_elements['use_chart'] = True
        animation_elements['suggested_elements'].append('data_visualization')

    # 对比相关触发词
    comparison_triggers = [
        '不同', '区别', '对比', '相比', '而', '但是', '然而', '优缺点', '优势'
    ]
    if any(trigger in content for trigger in comparison_triggers):
        animation_elements['use_comparison'] = True
        animation_elements['suggested_elements'].append('comparison_layout')
    # 情感和趣味元素判断

    emotion_triggers = ['有趣', '神奇', '惊人', '厉害', '酷', '棒', '哇', '真的']
    if any(trigger in content for trigger in emotion_triggers):
        animation_elements['use_emoji'] = True
        animation_elements['use_bubble'] = True
        animation_elements['suggested_elements'].extend(
            ['emoji_reaction', 'speech_bubble'])

    # 基于内容类型的默认建议
    type_defaults = {
        'definition': ['concept_highlight', 'definition_card'],
        'example': ['case_study', 'step_by_step'],
        'explanation': ['flow_diagram', 'cause_effect'],
        'emphasis': ['highlight_effect', 'attention_grabber']
    }

    if content_type in type_defaults:
        animation_elements['suggested_elements'].extend(
            type_defaults[content_type])

    return animation_elements


# 英文翻译功能
def translate_text_to_english(text):
    """
    将中文文本翻译为英文
    """

    prompt = """

# 角色
你是一位专业的翻译专家，擅长将中文文本准确流畅地翻译成英文。

## 技能
- 接收到中文内容后，将其准确翻译成英文，确保译文保持原文的意义、语气和风格。
- 充分考虑中文的语境和文化内涵，使英文表达既忠实原文又符合英语习惯。
- 禁止同一句子生成多份译文。
- 输出内容需符合英语语法规范，表达清晰、流畅，并具有良好的可读性。
- 准确传达原文所有信息，避免随意添加或删减内容。
- 仅提供与中文到英文翻译相关的服务。
- 只输出翻译结果，不要任何说明。

"""

    try:
        print(f'[翻译DEBUG] 原文: {text[:50]}...')
        full_prompt = f'{prompt}\n原文：{text}\n译文：'
        print(f'[翻译DEBUG] 完整提示词: {full_prompt[:100]}...')
        result = modai_model_request(
            full_prompt,
            model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
            max_tokens=512,
            temperature=0.3)
        print(f'[翻译DEBUG] 翻译结果: {result}')
        print(f'[翻译DEBUG] 结果类型: {type(result)}')
        return result.strip() if result else ''
    except Exception as e:
        print(f'英文翻译失败: {e}')
        return ''


def enhanced_generate_manim_code(content_type,
                                 content,
                                 class_name,
                                 surrounding_text='',
                                 total_duration=8.0,
                                 context_info=None):
    """
    增强版动画代码生成 - 集成智能元素判断和丰富动画效果
    """

    context_info = context_info or {}

    animation_elements = should_add_animation_elements(content, content_type,
                                                       context_info)
    if content_type == 'definition':
        prompt = f"""你是专业的教育动画设计师。创建一个生动有趣的定义展示动画：

**定义内容**: {content}
**智能建议元素**: {animation_elements['suggested_elements']}
**使用表情符号**: {'是' if animation_elements['use_emoji'] else '否'}
**使用气泡对话**: {'是' if animation_elements['use_bubble'] else '否'}
**总时长**: {total_duration:.1f}秒

设计要求：
1. 类名必须是 {class_name}
2. 风格：轻松科普，专业且有趣
3. 色彩：深色主体文字 + 鲜艳强调色 + 白色背景适配
4. 动画丰富度：
   - 文字出现用Write或FadeIn配合轻微bouncing
   - 重要词汇闪烁或颜色变化
   - 适当添加Indicate、Circumscribe等强调动画
   - 可以添加小图标、箭头、装饰元素

请生成完整Manim代码，确保动画生动、信息清晰、节奏舒适："""

    elif content_type == 'formula':
        prompt = f"""创建一个引人入胜的公式展示动画：

**公式内容**: {content}
**智能元素建议**: {animation_elements['suggested_elements']}

设计要求：
1. 类名: {class_name}
2. 核心策略：优先用Text，MathTex作为增强（避免LaTeX问题）
3. 视觉丰富：公式分步骤出现，等号运算符特殊强调
4. 趣味元素：{'添加计算气泡和表情符号' if animation_elements['use_bubble'] else '使用简洁风格'}

请生成完整代码："""

    elif content_type == 'code':
        prompt = f"""设计一个编程教学动画：

**代码内容**: {content}
**智能建议**: {animation_elements['suggested_elements']}

要求：
1. 类名: {class_name}
2. 代码展示：用Text类（font="Courier"）而非Code类
3. 语法高亮：关键字蓝色、变量绿色、字符串橙色
4. 执行模拟：用箭头、高亮等展示程序运行流程

请创建生动的代码教学动画："""

    else:
        prompt = f"""创建{content_type}类型的教学动画：

**内容**: {content}
**建议元素**: {animation_elements['suggested_elements']}
**时长**: {total_duration:.1f}秒

要求：
1. 类名: {class_name}
2. 视觉丰富：多色彩、多层次、多动画效果
3. 重点突出：关键信息用特殊颜色和动画

请创建引人入胜的动画："""

    try:
        result = modai_model_request(
            prompt,
            model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
            max_tokens=1200,
            temperature=0.8)
        return result.strip()
    except Exception as e:
        print(f'增强动画生成失败: {e}')
        return create_simple_manim_scene(content_type, content, class_name, '')


def create_manual_background(title_text='', output_dir='output', topic=None):
    """默认背景样式"""

    from PIL import Image, ImageDraw, ImageFont
    import os
    import textwrap

    os.makedirs(output_dir, exist_ok=True)
    width, height = 1920, 1080
    background_color = (255, 255, 255)
    title_color = (0, 0, 0)

    config = {
        'title_font_size': 50,
        'subtitle_font_size': 54,
        'title_max_width': 15,
        'subtitle_color': (0, 0, 0),
        'line_spacing': 15,
        'padding': 50,
        'line_width': 8,
        'subtitle_offset': 40,
        'line_position_offset': 190
    }

    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)

    def _get_font(size):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        import matplotlib.font_manager as fm
        font_names = [
            'SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Microsoft YaHei'
        ]
        # 首先尝试加载本地字体文件
        local_font = os.path.join(script_dir, 'asset', '字小魂扶摇手书(商用需授权).ttf')
        try:
            return ImageFont.truetype(local_font, size)
        except Exception as e:
            print(f'本地字体加载失败: {local_font}, 错误: {str(e)}')
        # 尝试使用matplotlib查找系统中的中文字体
        for font_name in font_names:
            try:
                font_path = fm.findfont(fm.FontProperties(family=font_name))
                return ImageFont.truetype(font_path, size)
            except Exception as e:
                print(f'无法找到字体: {font_name}, 错误: {str(e)}')
                continue

        print('所有字体加载失败，使用默认字体')
        return ImageFont.load_default()

    title_font = _get_font(config['title_font_size'])
    subtitle_font = _get_font(config['subtitle_font_size'])

    title_display = title_text or 'AI知识科普'
    title_lines = textwrap.wrap(title_display, width=config['title_max_width'])
    y_position = config['padding']
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        draw.text((config['padding'], y_position),
                  line,
                  font=title_font,
                  fill=title_color)
        y_position += (bbox[3] - bbox[1]) + config['line_spacing']
    subtitle_lines = ['硬核知识分享', '魔搭社区出品']
    y_position = config['padding']
    for i, line in enumerate(subtitle_lines):
        bbox = draw.textbbox((0, 0), line, font=subtitle_font)
        x_offset = width - bbox[2] - (config['padding'] + 30) + (
            i * config['subtitle_offset'])
        draw.text((x_offset, y_position),
                  line,
                  font=subtitle_font,
                  fill=config['subtitle_color'])
        y_position += bbox[3] - bbox[1] + 5

    line_y = height - config['padding'] - config['line_position_offset']
    draw.line([(0, line_y), (width, line_y)],
              fill=(0, 0, 0),
              width=config['line_width'])

    if topic:
        # 清理topic中的特殊字符，避免路径问题
        import re
        safe_topic = re.sub(r'[^\w\u4e00-\u9fff\-_]', '_',
                            topic)  # 只保留字母、数字、中文、横线、下划线
        safe_topic = safe_topic[:50]  # 限制长度
        theme_dir = os.path.join(output_dir, safe_topic)
        os.makedirs(theme_dir, exist_ok=True)
        output_path = os.path.join(theme_dir, f'background_{uuid.uuid4()}.png')
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir,
                                   f'background_{uuid.uuid4()}.png')
    image.save(output_path)
    print(f'使用统一背景样式生成: {output_path}')
    return output_path


def create_subtitle_image(text,
                          width=1720,
                          height=120,
                          font_size=28,
                          text_color='black',
                          bg_color='rgba(0,0,0,0)'):
    """使用PIL创建字幕图片，自动适应高度"""

    print(f"[字幕生成] 开始创建字幕图片，文本: {text[:30]}{'...' if len(text) > 30 else ''}")
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('msyh.ttc', font_size)
    except:  # noqa
        try:
            font = ImageFont.truetype('arial.ttf', font_size)
        except:  # noqa
            font = ImageFont.load_default()

    def split_long_text_for_subtitles(text, max_chars_per_subtitle=50):
        """将长文本智能分割成多个字幕片段"""
        if len(text) <= max_chars_per_subtitle:
            return [text]
        # 按句子分割
        sentences = re.split(r'([。！？；，、])', text)
        subtitle_parts = []
        current_part = ''
        for sentence in sentences:
            if not sentence.strip():
                continue
            test_part = current_part + sentence
            if len(test_part) <= max_chars_per_subtitle:
                current_part = test_part
            else:
                if current_part:
                    subtitle_parts.append(current_part.strip())
                current_part = sentence

        if current_part.strip():
            subtitle_parts.append(current_part.strip())
        return subtitle_parts

    def smart_wrap_text(text, font, max_width, max_lines=2):
        """换行逻辑改进"""
        lines = []

        sample_char_width = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox(
            (0, 0), '中', font=font)[2]
        chars_per_line = int((max_width * 0.9) // sample_char_width)
        total_capacity = chars_per_line * max_lines
        if len(text) > total_capacity:
            truncate_pos = total_capacity - 3
            punctuation = ['。', '！', '？', '；', '，', '、']
            best_cut = truncate_pos

            for i in range(
                    min(len(text), truncate_pos), max(0, truncate_pos - 20),
                    -1):
                if text[i] in punctuation:
                    best_cut = i + 1
                    break
            text = text[:best_cut]

        # 按标点符号分句
        import re
        sentences = re.split(r'([。！？；，、])', text)
        current_line = ''
        for part in sentences:
            if not part.strip():
                continue

            test_line = current_line + part
            bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0),
                                                                     test_line,
                                                                     font=font)
            line_width = bbox[2] - bbox[0]
            if line_width <= max_width * 0.9 and len(lines) < max_lines:
                current_line = test_line
            else:
                if current_line.strip() and len(lines) < max_lines:
                    lines.append(current_line.strip())
                    current_line = part
                elif len(lines) >= max_lines:
                    break
        if current_line.strip() and len(lines) < max_lines:
            lines.append(current_line.strip())
        final_lines = []
        for line in lines:
            if len(final_lines) >= max_lines:
                break

            bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0),
                                                                     line,
                                                                     font=font)
            line_width = bbox[2] - bbox[0]
            if line_width <= max_width * 0.9:
                final_lines.append(line)
            else:
                chars = list(line)
                temp_line = ''
                for char in chars:
                    if len(final_lines) >= max_lines:
                        break

                    test_line = temp_line + char
                    bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox(
                        (0, 0), test_line, font=font)
                    test_width = bbox[2] - bbox[0]

                    if test_width <= max_width * 0.9:
                        temp_line = test_line
                    else:
                        if temp_line and len(final_lines) < max_lines:
                            final_lines.append(temp_line)
                        temp_line = char

                if temp_line and len(final_lines) < max_lines:
                    final_lines.append(temp_line)

        return final_lines[:max_lines]

    min_font_size = 18
    max_height = 400
    original_font_size = font_size
    lines = []

    while font_size >= min_font_size:
        try:
            if font_size != original_font_size:
                font = ImageFont.truetype('msyh.ttc', font_size)
        except:  # noqa
            font = ImageFont.load_default()

        lines = smart_wrap_text(text, font, width, max_lines=2)
        line_height = font_size + 8
        total_text_height = len(lines) * line_height

        all_lines_fit = True
        for line in lines:
            bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0),
                                                                     line,
                                                                     font=font)
            line_width = bbox[2] - bbox[0]
            if line_width > width * 0.95:
                all_lines_fit = False
                break

        if total_text_height <= height and all_lines_fit:
            break
        elif total_text_height <= max_height and all_lines_fit:
            height = min(total_text_height + 20, max_height)
            break
        else:
            font_size = int(font_size * 0.9)

    line_height = font_size + 8
    total_text_height = len(lines) * line_height
    actual_height = total_text_height + 16
    img = Image.new('RGBA', (width, actual_height), bg_color)
    draw = ImageDraw.Draw(img)
    y_start = 8
    for i, line in enumerate(lines):
        if not line.strip():
            continue

        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = max(0, (width - text_width) // 2)
        y = y_start + i * line_height

        if y + line_height <= actual_height and x >= 0 and x + text_width <= width:
            draw.text((x, y), line, fill=text_color, font=font)
    print(f'[字幕生成] 字幕图片创建完成，尺寸: {width}x{actual_height}')
    return img, actual_height


def split_content_into_lines(text, max_chars_per_line=20, max_lines=4):
    import re

    text = re.sub(r'([，。；！？、])', r'\1\n', text)
    fragments = [f.strip() for f in text.split('\n') if f.strip()]
    lines = []
    current_line = ''

    for fragment in fragments:
        if len(fragment) > max_chars_per_line:
            words = list(fragment)
            temp_line = current_line

            for char in words:
                if len(temp_line + char) <= max_chars_per_line:
                    temp_line += char
                else:
                    if temp_line:
                        lines.append(temp_line)
                        temp_line = char

                    if len(lines) >= max_lines - 1:
                        break
            current_line = temp_line

        else:
            test_line = current_line + fragment if not current_line else current_line + fragment
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = fragment

                if len(lines) >= max_lines - 1:
                    break

    if current_line and len(lines) < max_lines:
        lines.append(current_line)

    if len(lines) > max_lines:
        last_lines = lines[max_lines - 1:]
        combined_last = ''.join(last_lines)

        if len(combined_last) > max_chars_per_line * 1.5:
            combined_last = combined_last[:int(max_chars_per_line
                                               * 1.5)] + '...'
        lines = lines[:max_lines - 1] + [combined_last]

    return lines[:max_lines]


def create_bilingual_subtitle_image(zh_text,
                                    en_text='',
                                    width=1720,
                                    height=120):
    """
    创建双语字幕
    """

    try:
        import tempfile
        from PIL import Image, ImageDraw, ImageFont

        zh_font_size = 32
        en_font_size = 22
        zh_en_gap = 6

        # 生成中文字幕
        zh_img, zh_height = create_subtitle_image(zh_text, width, height,
                                                  zh_font_size, 'black')

        # 生成英文字幕
        if en_text.strip():
            en_img, en_height = create_subtitle_image(en_text, width, height,
                                                      en_font_size, 'gray')
            total_height = zh_height + en_height + zh_en_gap

            combined_img = Image.new('RGBA', (width, total_height),
                                     (0, 0, 0, 0))
            combined_img.paste(zh_img, (0, 0), zh_img)
            combined_img.paste(en_img, (0, zh_height + zh_en_gap), en_img)
            final_img = combined_img
            final_height = total_height
        else:
            final_img = zh_img
            final_height = zh_height

        temp_path = os.path.join(tempfile.gettempdir(),
                                 f'subtitle_{uuid.uuid4()}.png')
        final_img.save(temp_path)
        print(f'[字幕生成] 双语字幕图片已保存到: {temp_path}')
        return temp_path, final_height
    except Exception as e:
        print(f'字幕生成失败: {e}')
        try:
            import tempfile
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGBA', (width, 100), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            draw.text((50, 30), zh_text[:50], fill=(255, 255, 255), font=font)
            temp_path = os.path.join(tempfile.gettempdir(),
                                     f'subtitle_fallback_{uuid.uuid4()}.png')
            img.save(temp_path)
            print(f'[字幕生成] 回退字幕图片已保存到: {temp_path}')
            return temp_path, 100
        except:  # noqa
            return '', 100


def add_background_music(video_path, output_path, music_volume=0.1):
    """
    为视频添加背景音乐
    """

    try:
        from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
        import moviepy.audio.fx.all as afx

        video = VideoFileClip(video_path)
        bg_music_path = os.path.join(
            os.path.dirname(__file__), 'asset', 'bg_audio.mp3')
        if os.path.exists(bg_music_path):
            bg_music = AudioFileClip(bg_music_path)
            if bg_music.duration < video.duration:
                bg_music = afx.audio_loop(bg_music, duration=video.duration)
            else:
                bg_music = bg_music.subclip(0, video.duration)
            bg_music = bg_music.volumex(music_volume)

            if video.audio:
                final_audio = CompositeAudioClip([video.audio, bg_music])
            else:
                final_audio = bg_music
            final_video = video.set_audio(final_audio)
        else:
            print('未找到背景音乐文件，跳过背景音乐')
            final_video = video

        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=24,
            verbose=False,
            logger=None,
            audio_bitrate='192k')

        print(f'背景音乐合成完成: {output_path}')
        return output_path
    except Exception as e:
        print(f'背景音乐合成失败: {e}')
        try:
            shutil.copy2(video_path, output_path)
            return output_path
        except:  # noqa
            return video_path


def split_text_by_punctuation(text):
    """
    使用LLM智能分句
    """

    text = re.sub(r'\s+', ' ', text).strip()
    prompt = f"""请将以下文本智能分句，确保：
1. 每个句子语义完整，不破坏逻辑
2. 标点符号保持在句子末尾，不要分离
3. 每句长度适中：至少10-15个字，最多35-40个字
4. 优先在自然语义边界分句（如：因此、所以、但是、而且等连接词前后）
5. 保持原文意思不变

文本：{text}

请返回JSON格式的句子列表，格式：
{{"sentences": ["句子1", "句子2", "句子3"]}}"""

    try:
        response = modai_model_request(
            prompt, max_tokens=1024, temperature=0.1)

        # 如果response为空，直接使用正则表达式处理
        if not response or not response.strip():
            print('LLM返回空响应，使用正则表达式处理...')
            raise Exception('Empty response from LLM')

        import json

        # 检查是否包含推理过程而非JSON答案
        if '我需要将给定的文本智能分句' in response and '{"sentences"' not in response:
            print('检测到推理过程响应，尝试提取JSON部分...')
            # 查找是否有隐藏的JSON部分
            json_match = re.search(r'\{[^}]*"sentences"[^}]*\}', response,
                                   re.DOTALL)
            if json_match:
                response = json_match.group(0)
                print(f'找到JSON片段: {response[:100]}...')
            else:
                print('未找到有效JSON，使用正则表达式处理...')
                raise Exception('No valid JSON found in reasoning response')

        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]

        response = response.strip()
        if not response.startswith('{'):
            start_idx = response.find('{')
            if start_idx != -1:
                response = response[start_idx:]

        if not response.endswith('}'):
            end_idx = response.rfind('}')
            if end_idx != -1:
                response = response[:end_idx + 1]

        quote_count = response.count('"')
        if quote_count % 2 != 0:
            last_quote_pos = response.rfind('"')
            for i in range(last_quote_pos + 1, len(response)):
                if response[i] in [',', '}', ']']:
                    response = response[:i] + '"' + response[i:]
                    break

        # 使用本地的JSON解析函数
        def extract_json_with_fallback(text, default_value):
            try:
                # 尝试从markdown代码块中提取JSON
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', text,
                                       re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = text.strip()
                return json.loads(json_str)
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                print(f'JSON解析失败，使用默认值: {e}')
                return default_value

        result = extract_json_with_fallback(response, {'sentences': []})

        # 处理不同的返回类型
        if isinstance(result, dict):
            sentences = result.get('sentences', [])
        elif isinstance(result, list):
            # 如果返回的是列表，直接使用
            sentences = result
        else:
            sentences = []

        segments = []
        for sentence in sentences:
            sentence = str(sentence).strip()
            if len(sentence) > 3:
                segments.append({'type': 'text', 'content': sentence})

        segments = []
        for sentence in sentences:
            sentence = str(sentence).strip()
            if len(sentence) > 3:
                segments.append({'type': 'text', 'content': sentence})

        if not segments:
            raise Exception('LLM分句返回为空')
        print(f'LLM智能分句成功，共分出 {len(segments)} 个句子')
        return segments

    except Exception as e:
        print(f'LLM返回格式错误: {e}')
        print(f'原始响应: {response[:200]}...')

        try:
            sentences_match = re.search(r'"sentences":\s*\[(.*?)\]', response,
                                        re.DOTALL)
            if sentences_match:
                sentences_str = sentences_match.group(1)
                sentences = []
                sentence_matches = re.findall(r'"([^"]*)"', sentences_str)
                for sentence in sentence_matches:
                    if len(sentence.strip()) > 3:
                        sentences.append(sentence.strip())

                if sentences:
                    segments = []
                    for sentence in sentences:
                        segments.append({'type': 'text', 'content': sentence})

                    print(f'JSON修复成功，提取到 {len(segments)} 个句子')
                    return segments
        except Exception as repair_e:
            print(f'JSON修复也失败: {repair_e}')
    except Exception as e:
        print(f'LLM智能分句失败: {e}')

    print('使用正则表达式处理...')
    sentence_pattern = r'[^。！？；…!?]*?[。！？；…!?]'
    sentences = re.findall(sentence_pattern, text)

    remaining_text = text
    for sentence in sentences:
        remaining_text = remaining_text.replace(sentence, '', 1)

    if remaining_text.strip():
        sentences.append(remaining_text.strip())

    segments = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 3:
            segments.append({'type': 'text', 'content': sentence})

    print(f'正则表达式分句完成，共分出 {len(segments)} 个句子')
    return segments


def render_manim_scene(code,
                       scene_name,
                       output_dir,
                       content_type=None,
                       content=None,
                       max_retries=10):
    """
    渲染Manim场景并生成透明MOV视频 - 集成预处理和质量控制
    """
    import os
    import subprocess
    import tempfile
    import shutil
    import re

    # 步骤1: 预处理代码
    current_code = code

    # 最优先使用平衡空间约束系统进行预处理（简洁有效）
    if BALANCED_SPATIAL_AVAILABLE:
        print('平衡空间约束系统预处理...')
        try:
            # 创建平衡系统
            balanced_system = BalancedSpatialSystem()

            # 分析代码质量
            analysis = balanced_system.analyze_and_score(code)

            print('布局质量分析:')
            print(f"   - 元素数量: {analysis['element_count']}")
            print(f"   - 间距问题: {analysis['spacing_issues']}")
            print(f"   - 布局分数: {analysis['layout_score']}")
            print(
                f"   - 过度工程化: {'是' if analysis['is_over_engineered'] else '否'}"
            )

            # 如果需要优化，进行简单优化
            if analysis['layout_score'] < 80 or analysis['spacing_issues'] > 0:
                print('启动简单优化...')
                optimized_code = balanced_system.optimize_simple_code(code)

                # 重新分析
                new_analysis = balanced_system.analyze_and_score(
                    optimized_code)
                improvement = new_analysis['layout_score'] - analysis[
                    'layout_score']

                if improvement > 0:
                    current_code = optimized_code
                    print(f'[成功] 简单优化完成，质量提升: +{improvement}')
                else:
                    print('保持原始代码')
            else:
                print('[信息] 代码质量良好，无需优化')

        except Exception as e:
            print(f'平衡空间约束系统预处理失败: {e}')

    # 使用原代码进行简单清理
    else:
        # 原有的简单清理
        if isinstance(code, bytes):
            current_code = code.decode('utf-8', errors='ignore')
        else:
            current_code = code

        # 清理LLM生成的Markdown格式标记
        current_code = clean_llm_code_output(current_code)

    os.makedirs(output_dir, exist_ok=True)

    for attempt in range(max_retries + 1):
        print(f'尝试渲染 (第 {attempt + 1}/{max_retries + 1} 次)...')
        code_file = os.path.join(output_dir,
                                 f'{scene_name}_attempt_{attempt}.py')

        try:
            # 确保编码设置
            encoding_header = '''# -*- coding: utf-8 -*-

import sys
import os

# 强制设置编码为UTF-8，解决中文渲染问题

if hasattr(sys, 'setdefaultencoding'):
    sys.setdefaultencoding('utf-8')

os.environ['PYTHONIOENCODING'] = 'utf-8'

'''
            if '# -*- coding: utf-8 -*-' not in current_code:
                current_code = encoding_header + current_code
            elif 'PYTHONIOENCODING' not in current_code:
                current_code = current_code.replace(
                    '# -*- coding: utf-8 -*-\n', encoding_header)

            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(current_code)
            if '# -*- coding: utf-8 -*-' not in current_code:
                current_code = encoding_header + current_code
            elif 'PYTHONIOENCODING' not in current_code:
                current_code = current_code.replace(
                    '# -*- coding: utf-8 -*-\n', encoding_header)

            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(current_code)
        except UnicodeEncodeError:
            clean_code = current_code.encode(
                'ascii', errors='ignore').decode('ascii')
            clean_code = '# -*- coding: utf-8 -*-\n' + clean_code
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(clean_code)

        output_path = os.path.join(output_dir, f'{scene_name}.mov')

        try:
            class_match = re.search(r'class\s+(\w+)\s*\(Scene\)', current_code)
            actual_scene_name = class_match.group(
                1) if class_match else scene_name

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_code_file = os.path.join(temp_dir,
                                              f'{scene_name}_temp.py')
                shutil.copy2(code_file, temp_code_file)
                print(f'渲染场景: {actual_scene_name}')
                env = os.environ.copy()
                env['PYTHONWARNINGS'] = 'ignore'
                env['MANIM_DISABLE_OPENCACHING'] = '1'
                env['PYTHONIOENCODING'] = 'utf-8'
                env['LANG'] = 'zh_CN.UTF-8'
                env['LC_ALL'] = 'zh_CN.UTF-8'

                cmd = [
                    'manim', 'render', '-ql', '--transparent', '--format=mov',
                    '--resolution=1280,720', '--disable_caching',
                    os.path.basename(temp_code_file), actual_scene_name
                ]

                result = subprocess.run(
                    cmd,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    timeout=300,
                    env=env)

                print(f'返回码: {result.returncode}')

                output_text = (result.stdout or '') + (result.stderr or '')

                warnings_to_ignore = [
                    'pkg_resources is deprecated', 'UserWarning',
                    'DeprecationWarning', 'FutureWarning', 'manim_voiceover'
                ]

                is_only_warning = False
                if result.returncode == 1:
                    has_real_error = False
                    has_warning = False

                    for warning in warnings_to_ignore:
                        if warning in output_text:
                            has_warning = True

                    real_error_indicators = [
                        'SyntaxError', 'NameError', 'ImportError',
                        'AttributeError', 'TypeError', 'ValueError',
                        'ModuleNotFoundError', 'Traceback', 'Error:',
                        'Failed to render'
                    ]

                    for error_indicator in real_error_indicators:
                        if error_indicator in output_text:
                            has_real_error = True
                            break

                    if has_warning and not has_real_error:
                        is_only_warning = True
                        print('检测到警告但可能渲染成功，检查输出文件...')

                temp_media_dir = os.path.join(temp_dir, 'media', 'videos')
                if os.path.exists(temp_media_dir):
                    for root, dirs, files in os.walk(temp_media_dir):
                        for file in files:
                            if file == f'{actual_scene_name}.mov':
                                found_file = os.path.join(root, file)
                                print(f'在临时目录找到文件: {found_file}')
                                shutil.copy2(found_file, output_path)
                                print(f'成功生成透明视频: {output_path}')
                                if verify_and_fix_mov_file(output_path):
                                    print('MOV文件验证通过')
                                else:
                                    print('MOV文件验证失败，尝试转换...')
                                    fixed_path = convert_mov_to_compatible(
                                        output_path)
                                    if fixed_path:
                                        output_path = fixed_path
                                        print(f'MOV文件已修复: {fixed_path}')

                                scaled_path = scale_video_to_fit(
                                    output_path, target_size=(1280, 720))
                                if scaled_path and scaled_path != output_path:
                                    print(f'视频已缩放以适应屏幕: {scaled_path}')
                                    return scaled_path

                                return output_path

                success_indicators = [
                    'File ready at' in output_text, 'Rendered' in output_text,
                    'INFO     Previewed File at:' in output_text,
                    'Combining to Movie file' in output_text
                ]

                if any(success_indicators) or (is_only_warning
                                               and result.returncode == 1):
                    print('Manim报告成功但未找到预期MOV文件，扩大搜索范围...')
                    search_dirs = [temp_media_dir]
                    if os.path.exists(temp_dir):
                        for root, dirs, files in os.walk(temp_dir):
                            search_dirs.extend(
                                [os.path.join(root, d) for d in dirs])

                    found_file = None
                    for search_dir in set(search_dirs):
                        if not os.path.exists(search_dir):
                            continue
                        print(f'搜索目录: {search_dir}')

                        for root, dirs, files in os.walk(search_dir):
                            # 查找所有.mov文件
                            mov_files = [
                                f for f in files if f.endswith('.mov')
                            ]
                            if mov_files:
                                latest_file = max(
                                    mov_files,
                                    key=lambda f: os.path.getmtime(
                                        os.path.join(root, f)))
                                found_file = os.path.join(root, latest_file)
                                print(f'找到MOV文件: {found_file}')
                                break

                        if found_file:
                            break

                    if found_file and os.path.exists(found_file):
                        try:
                            shutil.copy2(found_file, output_path)
                            print(f'成功复制MOV文件: {output_path}')
                            return output_path
                        except Exception as copy_err:
                            print(f'复制文件失败: {copy_err}')
                    else:
                        print('在所有搜索目录中都未找到有效的MOV文件')

                if result.returncode != 0 and not is_only_warning:
                    raise subprocess.CalledProcessError(
                        result.returncode, cmd, result.stdout, result.stderr)

        except subprocess.CalledProcessError as e:
            error_msg = ''
            try:
                if e.stderr:
                    error_msg = e.stderr
                elif e.stdout:
                    error_msg = e.stdout
                else:
                    error_msg = str(e)
            except UnicodeDecodeError:
                error_msg = '编码错误，无法显示详细错误信息'
            print(f'第 {attempt + 1} 次渲染失败: {error_msg[:200]}...')

            # 使用LLM修复错误
            if attempt < max_retries and content_type and content:
                print('尝试使用LLM修复错误...')
                fixed_code = fix_manim_error_with_llm(current_code, error_msg,
                                                      content_type, scene_name)

                if fixed_code:
                    current_code = fixed_code
                    print('代码已修复，准备重试...')
                    continue
                else:
                    print('LLM修复失败')

            if attempt == max_retries:
                print('所有渲染尝试均失败')
                return None

        except Exception as e:
            print(f'第 {attempt + 1} 次渲染过程出错: {e}')
            if attempt == max_retries:
                return None
    return None


def scale_video_to_fit(video_path, target_size=(1280, 720)):
    """
    缩放视频以确保内容适合目标尺寸，避免内容超出屏幕边界
    """

    try:
        from moviepy.editor import VideoFileClip
        import os

        if not os.path.exists(video_path):
            return video_path

        print(f'检查视频尺寸: {video_path}')
        clip = VideoFileClip(video_path)
        original_size = clip.size
        print(f'原始尺寸: {original_size}')

        target_width, target_height = target_size
        original_width, original_height = original_size

        scale_x = target_width / original_width
        scale_y = target_height / original_height
        scale_factor = min(scale_x, scale_y, 1.0)

        if scale_factor < 0.95:
            print(f'需要缩放，缩放比例: {scale_factor:.2f}')
            scaled_clip = clip.resize(scale_factor)

            base_path, ext = os.path.splitext(video_path)
            scaled_path = f'{base_path}_scaled{ext}'
            scaled_clip.write_videofile(
                scaled_path,
                codec='libx264',
                audio_codec='aac' if scaled_clip.audio else None,
                fps=24,
                verbose=False,
                logger=None)

            clip.close()
            scaled_clip.close()
            print(f'视频缩放完成: {scaled_path}')
            return scaled_path
        else:
            print('视频尺寸合适，无需缩放')
            clip.close()
            return video_path
    except Exception as e:
        print(f'视频缩放失败: {e}')
        return video_path


def verify_and_fix_mov_file(mov_path):
    """
    验证MOV文件是否能被正确读取
    """

    try:
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(mov_path)
        frame = clip.get_frame(0)
        clip.close()

        if frame is not None:
            return True
        else:
            return False
    except Exception as e:
        print(f'MOV验证失败: {e}')
        return False


def convert_mov_to_compatible(mov_path):
    """
    将有问题的MOV文件转换为兼容格式
    """

    try:

        from moviepy.editor import VideoFileClip
        import os

        base_path, ext = os.path.splitext(mov_path)
        fixed_path = f'{base_path}_fixed.mov'

        clip = VideoFileClip(mov_path)

        clip.write_videofile(
            fixed_path,
            codec='libx264',
            audio_codec='aac' if clip.audio else None,
            fps=24,
            verbose=False,
            logger=None,
            ffmpeg_params=['-pix_fmt', 'yuva420p'])

        clip.close()
        if verify_and_fix_mov_file(fixed_path):
            return fixed_path
        else:
            return None
    except Exception as e:
        print(f'MOV修复失败: {e}')
        return None


def create_simple_manim_scene(content_type, content, scene_name, output_dir):
    """
    简单的回退场景
    """

    import tempfile
    import shutil

    os.makedirs(output_dir, exist_ok=True)
    if content_type == 'formula':
        formula_text = content.replace('\\', '').replace('{',
                                                         '').replace('}', '')
        simple_code = f'''from manim import *

class {scene_name}(Scene):
    def construct(self):
        try:
            # 优先尝试Text显示，避免LaTeX问题
            formula = Text(r"{formula_text}", font_size=36, color=BLUE)
            formula.move_to(ORIGIN)

            # 如果内容看起来像数学公式，尝试MathTex
            if any(char in r"{content}" for char in ['=', '+', '-', '*', '/', '^', '_']):
                try:
                    math_formula = MathTex(r"{content}")
                    math_formula.scale(1.2)
                    formula = math_formula
                except:
                    pass  # 如果MathTex失败，继续使用Text

            # 分阶段展示：出现 → 完整展示 → 停留
            self.play(Write(formula), run_time=2)
            self.wait(4)  # 充分时间理解公式
            self.play(Indicate(formula, color=YELLOW), run_time=1)  # 强调
            self.wait(2)  # 继续停留
            self.play(FadeOut(formula), run_time=1)
        except Exception as e:
            # 改进的错误回退
            print(f"公式渲染错误: {{e}}")
            text = Text("数学公式展示", font_size=28, color=BLUE)
            text.move_to(ORIGIN)
            self.play(Write(text), run_time=1.5)
            self.wait(5)
            self.play(FadeOut(text), run_time=1)
'''

    elif content_type == 'code':
        code_lines = content.split('\n')
        if len(code_lines) > 4:
            code_lines = code_lines[:4] + ['...']
        code_display = '\\n'.join(code_lines)
        simple_code = f'''from manim import *

class {scene_name}(Scene):
    def construct(self):
        try:
            # 显示标题
            title = Text("代码示例:", font_size=28, color=YELLOW)
            title.to_edge(UP)
            self.play(Write(title), run_time=1)
            self.wait(0.5)

            # 显示实际代码（使用等宽字体）
            code_text = Text("""{code_display}""",
                           font_size=20,
                           font="Courier",
                           color=WHITE)
            code_text.next_to(title, DOWN, buff=0.5)


            # 逐行显示代码，更慢的节奏
            self.play(Write(code_text), run_time=3)
            self.wait(4)  # 充分时间阅读代码

            # 可选：添加执行效果提示
            if "print" in code_text.text:
                output_hint = Text("执行结果会在这里显示", font_size=18, color=GREEN)
                output_hint.next_to(code_text, DOWN, buff=0.3)
                self.play(Write(output_hint), run_time=1)
                self.wait(2)
                self.play(FadeOut(output_hint), run_time=0.5)

            # 代码强调效果
            self.play(Circumscribe(code_text, color=YELLOW), run_time=1)
            self.wait(2)  # 继续展示
            self.play(FadeOut(code_text), FadeOut(title), run_time=1.5)
        except:
            text = Text("代码展示", font_size=36)
            self.play(Write(text), run_time=1.5)
            self.wait(5)
            self.play(FadeOut(text), run_time=1)
'''

    elif content_type == 'chart':
        simple_code = f'''from manim import *

class {scene_name}(Scene):
    def construct(self):
        try:
            # 最简单的柱状图，避免复杂参数
            chart = BarChart([1, 2, 3, 4])
            chart.scale(0.8)

            # 分阶段展示图表
            self.play(Create(chart), run_time=3)  # 慢慢创建图表
            self.wait(4)  # 充分时间理解数据


            # 高亮最高的柱子
            self.play(Indicate(chart), run_time=1)
            self.wait(2)  # 继续展示

            self.play(FadeOut(chart), run_time=1.5)
        except:
            text = Text("图表展示", font_size=36)
            self.play(Write(text), run_time=1.5)
            self.wait(5)
            self.play(FadeOut(text), run_time=1)
'''

    elif content_type == 'definition':
        content_lines = split_content_into_lines(
            content, max_chars_per_line=12, max_lines=6)
        content_display = '\\n'.join(content_lines)
        simple_code = f'''from manim import *
class {scene_name}(Scene):
    def construct(self):
        try:
            # 确保透明背景
            config.background_color = "#00000000"
            config.transparent = True

            # 显示定义标题 - 固定在安全位置
            title = Text("定义:", font_size=28, color=BLUE, weight=BOLD)
            title.move_to(UP * 3.2)  # 固定在顶部安全区域
            self.play(Write(title), run_time=1)
            self.wait(0.5)

            # 改进：强制多行显示，使用较小字体确保完全可见
            definition_text = Text("""{content_display}""",
                                 font_size=20,  # 更小字体确保安全
                                 color=WHITE,
                                 line_spacing=1.2)  # 紧凑行间距

            # 固定位置策略：居中显示，确保不超出边界
            definition_text.move_to(ORIGIN + UP * 0.3)  # 稍微偏上，为边框留空间

            # 超严格边界检查 - Manim安全区域：宽度12，高度6
            safe_width = 9    # 极保守的宽度限制
            safe_height = 4   # 极保守的高度限制

            # 强制缩放到安全区域内
            if definition_text.width > safe_width:
                scale_w = safe_width / definition_text.width
                definition_text.scale(scale_w)
                print(f"宽度缩放: {{scale_w:.2f}}")


            if definition_text.height > safe_height:
                scale_h = safe_height / definition_text.height
                definition_text.scale(scale_h)
                print(f"高度缩放: {{scale_h:.2f}}")


            # 确保最终位置在安全区域

            definition_text.move_to(ORIGIN)

            # 添加边框 - 使用固定尺寸确保不超出
            box = SurroundingRectangle(definition_text,
                                     color=BLUE,
                                     buff=0.3,
                                     stroke_width=2)


            # 最终安全检查：如果边框太大，再次缩放
            max_box_width = 10
            max_box_height = 5
            if box.width > max_box_width or box.height > max_box_height:
                final_scale = min(max_box_width / box.width, max_box_height / box.height)
                definition_group = VGroup(definition_text, box)
                definition_group.scale(final_scale)
                definition_group.move_to(ORIGIN)

            # 分阶段展示
            self.play(Write(definition_text), run_time=2.5)
            self.play(Create(box), run_time=1)
            self.wait(3)  # 充分时间理解定义

            # 强调效果
            self.play(Indicate(definition_text, color=YELLOW, scale_factor=1.1), run_time=1.5)
            self.wait(2)  # 继续停留

            # 清理
            self.play(FadeOut(definition_text), FadeOut(box), FadeOut(title), run_time=1.5)
        except Exception as e:
            print(f"定义渲染错误: {{e}}")
            # 错误回退：使用最简单的文本显示
            text = Text("定义展示", font_size=24, color=BLUE)
            text.move_to(ORIGIN)
            self.play(Write(text), run_time=1.5)
            self.wait(5)
            self.play(FadeOut(text), run_time=1)

'''

    elif content_type == 'theorem':
        content_preview = content[:100]
        simple_code = f'''from manim import *


class {scene_name}(Scene):
    def construct(self):
        try:
            # 显示定理标题
            title = Text("定理:", font_size=32, color=GOLD)
            title.to_edge(UP)
            self.play(Write(title))
            self.wait(0.5)

            # 显示定理内容
            theorem_text = Text("""{content_preview}""",
                              font_size=24,
                              color=YELLOW)
            theorem_text.next_to(title, DOWN, buff=0.5)

            # 添加金色边框
            box = SurroundingRectangle(theorem_text, color=GOLD, buff=0.3)
            self.play(Write(theorem_text))
            self.play(Create(box))
            self.play(Flash(theorem_text, color=GOLD))
            self.wait(2)
            self.play(FadeOut(theorem_text), FadeOut(box), FadeOut(title))
        except:
            text = Text("定理展示", font_size=36)
            self.play(Write(text))
            self.wait(2)
            self.play(FadeOut(text))
'''

    elif content_type == 'example':
        content_preview = content[:100]
        simple_code = f'''from manim import *

class {scene_name}(Scene):
    def construct(self):
        try:
            # 显示例子标题
            title = Text("例子:", font_size=32, color=GREEN)
            title.to_edge(UP)
            self.play(Write(title))
            self.wait(0.5)

            # 显示例子内容
            example_text = Text("""{content_preview}""",
                              font_size=22,
                              color=WHITE)
            example_text.next_to(title, DOWN, buff=0.5)
            self.play(Write(example_text))
            self.wait(2)
            self.play(FadeOut(example_text), FadeOut(title))
        except:
            text = Text("例子展示", font_size=36)
            self.play(Write(text))
            self.wait(2)
            self.play(FadeOut(text))
'''

    elif content_type == 'emphasis':
        content_preview = content[:80]
        simple_code = f'''from manim import *

class {scene_name}(Scene):
    def construct(self):
        try:
            # 显示强调内容
            emphasis_text = Text("""{content_preview}""",
                               font_size=36,
                               color=RED,
                               weight=BOLD)

            # 添加强调动画
            self.play(Write(emphasis_text))
            self.play(Circumscribe(emphasis_text, color=RED))
            self.play(Flash(emphasis_text, color=RED))
            self.wait(2)
            self.play(FadeOut(emphasis_text))
        except:
            text = Text("重点强调", font_size=36, color=RED)
            self.play(Write(text))
            self.play(Flash(text))
            self.wait(2)
            self.play(FadeOut(text))
'''

    elif content_type == 'comparison':
        simple_code = f'''from manim import *

class {scene_name}(Scene):
    def construct(self):
        try:
            # 左右对比展示
            left_text = Text("对比A", font_size=28, color=BLUE)
            right_text = Text("对比B", font_size=28, color=RED)

            left_text.to_edge(LEFT)
            right_text.to_edge(RIGHT)

            # VS标记
            vs_text = Text("VS", font_size=32, color=YELLOW)
            self.play(Write(left_text), Write(right_text))
            self.play(Write(vs_text))
            self.wait(2)
            self.play(FadeOut(left_text), FadeOut(right_text), FadeOut(vs_text))
        except:
            text = Text("对比展示", font_size=36)
            self.play(Write(text))
            self.wait(2)
            self.play(FadeOut(text))
'''

    elif content_type == 'step':
        simple_code = f'''from manim import *

class {scene_name}(Scene):
    def construct(self):
        try:
            # 步骤展示
            step1 = Text("步骤 1", font_size=24, color=WHITE)
            step2 = Text("步骤 2", font_size=24, color=WHITE)
            step3 = Text("步骤 3", font_size=24, color=WHITE)

            steps = [step1, step2, step3]
            for i, step in enumerate(steps):
                step.shift(UP * (1 - i) * 0.8)
                self.play(Write(step))
                self.wait(0.5)
            self.wait(1)
            self.play(*[FadeOut(step) for step in steps])
        except:
            text = Text("步骤展示", font_size=36)
            self.play(Write(text))
            self.wait(2)
            self.play(FadeOut(text))
'''

    else:
        return None

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_code_file = os.path.join(temp_dir, f'{scene_name}_simple.py')
            with open(temp_code_file, 'w', encoding='utf-8') as f:
                f.write(simple_code)

            env = os.environ.copy()
            env['PYTHONWARNINGS'] = 'ignore'
            env['MANIM_DISABLE_OPENCACHING'] = '1'

            cmd = [
                'manim', 'render', '-ql', '--transparent', '--format=mov',
                '--disable_caching',
                os.path.basename(temp_code_file), scene_name
            ]

            result = subprocess.run(
                cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=120,
                env=env)

            print(f'简单回退返回码: {result.returncode}')

            output_text = (result.stdout or '') + (result.stderr or '')

            # 不应该被视为错误的警告
            warnings_to_ignore = [
                'pkg_resources is deprecated', 'UserWarning',
                'DeprecationWarning', 'FutureWarning', 'manim_voiceover'
            ]

            is_only_warning = False
            if result.returncode == 1:
                has_warning = any(warning in output_text
                                  for warning in warnings_to_ignore)
                real_error_indicators = [
                    'SyntaxError', 'NameError', 'ImportError',
                    'AttributeError', 'TypeError', 'ValueError',
                    'ModuleNotFoundError', 'Traceback', 'Error:',
                    'Failed to render'
                ]

                has_real_error = any(error in output_text
                                     for error in real_error_indicators)
                if has_warning and not has_real_error:
                    is_only_warning = True
                    print('简单回退检测到警告但可能成功，检查输出文件...')

            output_path = os.path.join(output_dir, f'{scene_name}_simple.mov')
            temp_media_dir = os.path.join(temp_dir, 'media', 'videos')

            if os.path.exists(temp_media_dir):
                for root, dirs, files in os.walk(temp_media_dir):
                    for file in files:
                        if file.endswith('.mov'):
                            generated_path = os.path.join(root, file)
                            shutil.copy2(generated_path, output_path)
                            print(f'生成简单回退视频: {output_path}')
                            return output_path

            success_indicators = [
                'File ready at' in output_text, 'Rendered' in output_text,
                'INFO     Previewed File at:' in output_text,
                'Combining to Movie file' in output_text
            ]

            if any(success_indicators) or (is_only_warning
                                           and result.returncode == 1):
                print('简单回退可能成功但未找到标准路径文件，继续搜索...')
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.mov'):
                            generated_path = os.path.join(root, file)
                            shutil.copy2(generated_path, output_path)
                            print(f'找到并生成简单回退视频: {output_path}')
                            return output_path

            print('简单回退未找到MOV文件')
            return None
    except subprocess.TimeoutExpired:
        print('简单回退渲染超时')
        return None
    except Exception as e:
        print(f'简单回退渲染失败: {e}')
        return None


def generate_ai_science_knowledge_video(topic,
                                        output_dir='output',
                                        animation_mode='auto'):
    """
    生成一个AI知识科普视频的主工作流（能切换动画制作模式）
    """
    print(f"开始生成主题为 '{topic}' 的AI科普视频")
    print(f'动画制作模式: {animation_mode}')

    # 解析动画模式
    if HUMAN_ANIMATION_AVAILABLE:
        if animation_mode == 'human':
            mode = AnimationProductionMode.HUMAN_CONTROLLED
        else:
            mode = AnimationProductionMode.AUTO
    else:
        print('人工动画系统不可用，使用自动模式')
        mode = AnimationProductionMode.AUTO
        animation_mode = 'auto'

    # 移除Windows不支持的特殊字符
    def clean_filename_for_windows(filename):
        invalid_chars = r'[<>:"|?*/\\]'
        name = re.sub(invalid_chars, '_', filename)
        name = name.strip('. ')
        if len(name) > 50:
            name = name[:50]
        if not name or name.isspace():
            name = 'default_topic'
        return name

    topic_safe = clean_filename_for_windows(topic)
    full_output_dir = os.path.join(output_dir, topic_safe)
    print(f'输出目录为: {full_output_dir}')
    os.makedirs(full_output_dir, exist_ok=True)

    # 1. 生成幽默风趣的文案
    print('开始生成轻松幽默的文案')

    script_path = os.path.join(full_output_dir, 'script.txt')
    if os.path.exists(script_path):
        print('发现本地文案缓存，直接读取...')
        with open(script_path, 'r', encoding='utf-8') as f:
            script = f.read()
        print(f'本地文案读取成功，长度: {len(script)} 字符')
    else:
        print('本地无文案缓存，开始生成...')
        script = generate_script(topic)
        print(f'文案生成完成，长度: {len(script)} 字符')

        # 保存文案到目录
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)
        print('文案已保存到本地缓存')

    # 2. 解析结构化内容
    print('开始解析结构化内容')
    segments_path = os.path.join(full_output_dir, 'segments.json')

    if os.path.exists(segments_path):
        print('发现本地结构化内容缓存，直接读取...')
        try:
            with open(segments_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            print(f'本地结构化内容读取成功，共 {len(segments)} 个片段')
        except:  # noqa
            print('本地结构化内容读取失败，重新解析...')
            segments = parse_structured_content(script)
            print(f'解析完成，共 {len(segments)} 个片段')
    else:
        print('本地无结构化内容缓存，开始解析...')
        segments = parse_structured_content(script)
        print(f'解析完成，共 {len(segments)} 个片段')

    # === 集中清理 segments 的 content 和 parent_segment.content 字段 ===
    for seg in segments:
        if 'content' in seg:
            seg['content'] = clean_content(seg['content'])
        if 'parent_segment' in seg and isinstance(
                seg['parent_segment'],
                dict) and 'content' in seg['parent_segment']:
            seg['parent_segment']['content'] = clean_content(
                seg['parent_segment']['content'])

    # 保存结构化内容到本地缓存（无论新生成还是读取后都清理）
    with open(segments_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print('结构化内容已保存到本地缓存（已清理结构标记）')

    # 3. 段落分句处理
    print('应用文本分句处理...')
    try:
        final_segments = []
        for segment in segments:
            if segment['type'] == 'text' and len(segment['content']) > 100:
                print(f"分割长文本片段: {segment['content'][:50]}...")
                subsegments = split_text_by_punctuation(segment['content'])
                for subseg_dict in subsegments:
                    if subseg_dict['content'].strip():
                        final_segments.append({
                            'content':
                            subseg_dict['content'].strip(),
                            'type':
                            'text',
                            'parent_segment':
                            segment
                        })
            else:
                final_segments.append(segment)
        segments = final_segments
        print(f'分句处理完成，共 {len(segments)} 个片段')

    except Exception as e:
        print(f'分句处理失败: {e}')
        import traceback
        traceback.print_exc()

    segments_path = os.path.join(full_output_dir, 'segments.json')
    with open(segments_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    # 4. 生成TTS语音文件
    print('开始生成TTS语音')
    audio_paths = []
    tts_dir = os.path.join(full_output_dir, 'audio')
    os.makedirs(tts_dir, exist_ok=True)

    tts_cache_path = os.path.join(full_output_dir, 'tts_cache.json')
    tts_cache = {}
    if os.path.exists(tts_cache_path):
        print('发现本地TTS音频缓存，检查完整性...')
        try:
            with open(tts_cache_path, 'r', encoding='utf-8') as f:
                tts_cache = json.load(f)
            print(f'TTS缓存读取成功，包含 {len(tts_cache)} 个音频信息')
        except:  # noqa
            print('TTS缓存读取失败，重新生成...')
            tts_cache = {}

    for i, segment in enumerate(segments):
        tts_text = segment.get('content', '')
        if not tts_text and segment['type'] != 'text':
            if 'explanation' in segment:
                tts_text = segment['explanation']
            else:
                seg_type = segment.get('type', '')
                content = segment.get('content', '')
                explanation_prompt = f'请为这个{seg_type}生成简短的解说词（30字以内）：{content}'
                try:
                    explanation = modai_model_request(
                        explanation_prompt, max_tokens=100, temperature=0.5)
                    explanation = explanation.strip()
                    segment['explanation'] = explanation
                    tts_text = explanation
                    print(f'为第 {i+1} 段动画生成解说: {explanation}')
                except:  # noqa
                    # 备用
                    if seg_type == 'formula':
                        tts_text = '这里展示了一个重要公式'
                    elif seg_type == 'chart':
                        tts_text = '这里展示了相关图表数据'
                    elif seg_type == 'code':
                        tts_text = '这里演示了代码实现'
                    elif seg_type == 'definition':
                        tts_text = '这里解释了重要定义'
                    elif seg_type == 'theorem':
                        tts_text = '这里讲解了重要定理'
                    elif seg_type == 'example':
                        tts_text = '这里展示了实际例子'
                    elif seg_type == 'emphasis':
                        tts_text = '这里强调了关键重点'
                    elif seg_type == 'comparison':
                        tts_text = '这里对比了不同内容'
                    elif seg_type == 'step':
                        tts_text = '这里演示了操作步骤'
                    else:
                        tts_text = '这里展示了相关内容'

                    segment['explanation'] = tts_text

        if not tts_text:
            tts_text = segment.get('text', '') or segment.get('desc', '')

        audio_path = os.path.join(tts_dir, f'segment_{i+1}.mp3')
        audio_duration = None

        segment_key = f'segment_{i+1}'
        if segment_key in tts_cache and os.path.exists(audio_path):
            cached_info = tts_cache[segment_key]
            audio_duration = cached_info.get('duration', 3.0)
            print(f'第 {i+1} 段使用本地TTS缓存，时长: {audio_duration:.1f}秒')
            audio_paths.append(audio_path)
        else:
            if tts_text:
                print(
                    f"第 {i+1} 段准备生成TTS: '{tts_text[:50]}{'...' if len(tts_text) > 50 else ''}'"
                )
                success = edge_tts_generate(tts_text, audio_path,
                                            'YunjianNeural')
                if success and os.path.exists(audio_path):
                    audio_duration = get_audio_duration(audio_path)
                    print(f'第 {i+1} 段语音生成完成，时长: {audio_duration:.1f}秒')

                    tts_cache[segment_key] = {
                        'text': tts_text,
                        'duration': audio_duration,
                        'path': audio_path
                    }

                    audio_paths.append(audio_path)
                else:
                    print(f'第 {i+1} 段TTS失败，创建静音回退...')
                    audio_duration = 3.0
                    create_silent_audio(audio_path, duration=audio_duration)
                    audio_paths.append(audio_path)
                    print(f'第 {i+1} 段使用静音回退')
            else:
                print(f'第 {i+1} 段无文案，创建静音...')
                # 没有文案时创建静音片段
                audio_duration = 2.0
                create_silent_audio(audio_path, duration=audio_duration)
                audio_paths.append(audio_path)
                print(f'第 {i+1} 段无文案，使用静音')

        segment['audio_duration'] = audio_duration

    try:
        with open(tts_cache_path, 'w', encoding='utf-8') as f:
            json.dump(tts_cache, f, ensure_ascii=False, indent=2)
        print('TTS缓存已保存')
    except Exception as e:
        print(f'TTS缓存保存失败: {e}')

    # 5. 生成插画图片
    print('生成插画图片...')
    illustration_paths = []
    try:
        # 生成插画描述
        text_segments = [seg for seg in segments if seg['type'] == 'text']
        if text_segments:
            illustration_prompts_path = os.path.join(
                full_output_dir, 'illustration_prompts.json')
            if os.path.exists(illustration_prompts_path):
                print('发现本地插画描述缓存，直接读取...')
                with open(
                        illustration_prompts_path, 'r', encoding='utf-8') as f:
                    illustration_prompts = json.load(f)
                print(f'本地插画描述读取成功: {len(illustration_prompts)} 个')
            else:
                print('本地无插画描述缓存，开始生成...')
                illustration_prompts = generate_illustration_prompts(
                    [seg['content'] for seg in text_segments])
                print(f'插画描述生成完成: {len(illustration_prompts)} 个')

                with open(
                        illustration_prompts_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        illustration_prompts, f, ensure_ascii=False, indent=2)
                print('插画描述已保存到本地缓存')

            images_dir = os.path.join(full_output_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)

            image_paths_path = os.path.join(images_dir, 'image_paths.json')
            if os.path.exists(image_paths_path):
                print('发现本地插画缓存，直接读取...')
                with open(image_paths_path, 'r', encoding='utf-8') as f:
                    image_paths = json.load(f)
                print(f'本地插画读取成功: {len(image_paths)} 个')
            else:
                print('本地无插画缓存，开始生成...')
                print('插画生成可能需要几分钟，请耐心等待...')

                image_paths = generate_images(
                    illustration_prompts, output_dir=full_output_dir)
                print(f'插画图片生成完成: {len(image_paths)} 个')

                for i, img_path in enumerate(image_paths):
                    if os.path.exists(img_path):
                        new_path = os.path.join(images_dir,
                                                f'illustration_{i+1}.png')
                        shutil.move(img_path, new_path)
                        image_paths[i] = new_path

                with open(image_paths_path, 'w', encoding='utf-8') as f:
                    json.dump(image_paths, f, ensure_ascii=False, indent=2)
                print('插画路径已保存到本地缓存')

            fg_out_dir = os.path.join(images_dir, 'output_black_only')
            if not os.path.exists(fg_out_dir):
                os.makedirs(fg_out_dir, exist_ok=True)
                print('创建透明背景插画目录...')

            if len(os.listdir(fg_out_dir)) == 0 or len(
                    os.listdir(fg_out_dir)) < len(image_paths):
                print('开始处理插画背景...')
                keep_only_black_for_folder(images_dir, fg_out_dir)
                print('插画背景处理完成')
            else:
                print('发现本地透明背景插画，使用去背景版本...')

            text_idx = 0
            for i, segment in enumerate(segments):
                if segment['type'] == 'text':
                    if text_idx < len(image_paths):
                        transparent_path = os.path.join(
                            fg_out_dir, f'illustration_{text_idx+1}.png')
                        if os.path.exists(transparent_path):
                            illustration_paths.append(transparent_path)
                            print(f'使用去背景插画: {transparent_path}')
                        else:
                            print(f'去背景插画不存在: {transparent_path}，使用原图替代')
                            illustration_paths.append(image_paths[text_idx])
                        text_idx += 1
                    else:
                        illustration_paths.append(None)
                else:
                    illustration_paths.append(None)

            print(f'插画路径构建完成: {len(illustration_paths)} 个')
        else:
            print('没有文本片段，跳过插画生成')
            illustration_paths = [None] * len(segments)
    except Exception as e:
        print(f'插画生成失败: {e}')
        print('使用空插画列表继续...')
        illustration_paths = [None] * len(segments)

    # 6. 生成背景图
    print('生成背景图...')
    if BACKGROUNDIMAGE_AVAILABLE:
        try:
            generator = BackgroundImageGenerator(topic=topic)
            unified_background_path = generator.generate(
                title_text=topic,
                subtitle_lines=['硬核知识分享', '魔搭社区出品'],
                line_position_offset=190)
            print(f'使用背景生成器: {unified_background_path}')

        except Exception as e:
            print(f'背景生成失败: {e}')
            unified_background_path = create_manual_background(
                title_text=topic, output_dir=full_output_dir, topic=topic)

    else:
        unified_background_path = create_manual_background(
            title_text=topic, output_dir=full_output_dir, topic=topic)
    if not unified_background_path or not os.path.exists(
            unified_background_path):
        print('背景图生成失败')
        return None

    # 6. 生成manim动画 - 支持多种制作模式
    print('生成动画...')
    foreground_paths = []

    # 初始化动画工作室（如果是人工模式）
    animation_studio = None
    task_manager = None
    placeholder_generator = None

    if HUMAN_ANIMATION_AVAILABLE and mode != AnimationProductionMode.AUTO:
        # 动态导入避免循环依赖
        from human_animation_studio import AnimationStudio
        animation_studio = AnimationStudio(
            full_output_dir, workflow_instance=sys.modules[__name__])
        task_manager = animation_studio.task_manager
        placeholder_generator = animation_studio.placeholder_generator
        print('人工动画工作studio已启动')

    for i, segment in enumerate(segments):
        # 恢复原始逻辑：只为显式标记的特殊类型段落创建动画
        segment_type = segment['type']

        # 只为非'text'类型的段落创建动画
        if segment_type == 'text':
            print(f'第 {i+1} 段是纯文本片段，跳过动画生成，使用插画')
            foreground_paths.append(None)
            continue

        print(
            f"生成第 {i+1} 个动画: {segment_type} (原类型: {segment['type']}) - {segment['content'][:30]}..."
        )

        # 根据制作模式处理动画
        if mode == AnimationProductionMode.HUMAN_CONTROLLED:
            # 人工控制模式：创建任务并生成占位符
            audio_duration = segment.get('audio_duration', 8.0)
            task_id = task_manager.create_task(
                segment_index=i + 1,
                content=segment['content'],
                content_type=segment_type,
                mode=mode,
                audio_duration=audio_duration)

            # 生成占位符视频
            placeholder_path = os.path.join(full_output_dir,
                                            f'scene_{i+1}_placeholder.mov')
            task = task_manager.get_task(task_id)
            placeholder_video = placeholder_generator.create_placeholder(
                task, placeholder_path)

            if placeholder_video:
                foreground_paths.append(placeholder_video)
                print(f'第 {i+1} 个动画占位符已生成: {placeholder_video}')
            else:
                foreground_paths.append(None)
                print(f'第 {i+1} 个动画占位符生成失败')

            continue

        # 自动模式的自动生成部分
        context_info = segment.get('context_info', {})
        surrounding_text = segment.get('surrounding_text', '')
        audio_duration = segment.get('audio_duration', None)

        manim_code = generate_manim_code(
            content=segment['content'],
            content_type=segment_type,
            scene_number=i + 1,
            context_info=context_info,
            surrounding_text=surrounding_text,
            audio_duration=audio_duration,
            main_theme=topic,
            context_segments=segments,
            segment_index=i,
            total_segments=segments)

        video_path = None

        if manim_code:
            if QUALITY_SYSTEM_AVAILABLE:
                print('评估动画质量...')

                vqa = VisualQualityAssessment()
                quality_assessment = vqa.assess_animation_quality(
                    manim_code, segment['content'], segment_type)

                acm = AnimationContentMatcher()
                match_result = acm.validate_match(manim_code,
                                                  segment['content'],
                                                  segment_type)

                quality_score = quality_assessment.get('overall_quality_score',
                                                       0)
                match_score = match_result.get('match_score', 0)

                print(f'质量得分: {quality_score}/100, 匹配度: {match_score}/100')
                if quality_score < 70 or match_score < 70:
                    print('质量不达标，尝试优化...')

                    improvements = quality_assessment.get(
                        'improvement_suggestions', [])
                    improvements.extend(
                        match_result.get('improvement_suggestions', []))

                    if improvements:
                        improvement_prompt = f"""
基于以下评估反馈，请优化动画代码：

原始内容：{segment['content']}
动画类型：{segment_type}

改进建议：
{chr(10).join(f"- {imp}" for imp in improvements)}

请生成改进后的Manim动画代码。
"""

                        improved_code = generate_manim_code(
                            content=segment['content'],
                            content_type=segment_type,
                            scene_number=i + 1,
                            context_info=context_info,
                            surrounding_text=surrounding_text,
                            audio_duration=audio_duration,
                            main_theme=topic,
                            context_segments=segments,
                            segment_index=i,
                            total_segments=segments,
                            improvement_prompt=improvement_prompt)

                        if improved_code:
                            manim_code = improved_code
                            print('已生成优化版本')

            code_file = os.path.join(full_output_dir,
                                     f"scene_{i+1}_{segment['type']}.py")
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(manim_code)

            scene_name = f'Scene{i+1}'
            scene_dir = os.path.join(full_output_dir, f'scene_{i+1}')
            video_path = render_manim_scene(
                manim_code,
                scene_name,
                scene_dir,
                content_type=segment['type'],
                content=segment['content'])

            # 如果渲染失败，用简单回退
            if not video_path:
                print('主渲染失败，尝试简单回退...')
                try:
                    fallback_dir = os.path.join(full_output_dir,
                                                f'scene_{i+1}_fallback')
                    video_path = create_simple_manim_scene(
                        segment['type'], segment['content'], scene_name,
                        fallback_dir)
                except Exception as e:
                    print(f'简单回退也失败: {e}')
                    video_path = None

        foreground_paths.append(video_path)

    # 7. 生成双语字幕
    print('生成双语字幕...')
    subtitle_paths = []
    subtitle_segments_list = []
    subtitle_dir = os.path.join(full_output_dir, 'subtitles')
    os.makedirs(subtitle_dir, exist_ok=True)

    for i, segment in enumerate(segments):
        if segment['type'] != 'text':
            zh_text = segment.get('explanation', '') or segment.get(
                'content', '')
            en_text = translate_text_to_english(zh_text)

            def split_subtitles(text, max_chars=30):
                import re
                sentences = re.split(r'([。！？；，、])', text)
                subtitles = []
                current = ''
                for s in sentences:
                    if not s.strip():
                        continue
                    test = current + s
                    if len(test) <= max_chars:
                        current = test
                    else:
                        if current:
                            subtitles.append(current.strip())
                        current = s
                if current.strip():
                    subtitles.append(current.strip())
                return subtitles

            subtitle_segments = split_subtitles(zh_text, max_chars=30)
            subtitle_img_paths = []
            for idx, sub_text in enumerate(subtitle_segments):
                sub_en = translate_text_to_english(sub_text)
                subtitle_path, subtitle_height = create_bilingual_subtitle_image(
                    zh_text=sub_text, en_text=sub_en, width=1720, height=120)
                if subtitle_path:
                    final_subtitle_path = os.path.join(
                        subtitle_dir, f'bilingual_subtitle_{i+1}_{idx+1}.png')
                    shutil.move(subtitle_path, final_subtitle_path)
                    subtitle_img_paths.append(final_subtitle_path)
                    print(
                        f'[字幕调试] 动画片段 {i+1} 第{idx+1}段字幕图片已保存: {final_subtitle_path}'
                    )
                else:
                    print(f'[字幕调试] 动画片段 {i+1} 第{idx+1}段字幕图片生成失败: {sub_text}')

            subtitle_paths.append(
                subtitle_img_paths[0] if subtitle_img_paths else None)
            if subtitle_img_paths:
                subtitle_segments_list.append(subtitle_img_paths)
            else:
                subtitle_segments_list.append([])
                print(f'[字幕调试] 动画片段 {i+1} 没有生成有效字幕，添加空列表')

        else:
            zh_text = segment.get('content', '')
            en_text = translate_text_to_english(zh_text)
            subtitle_path, subtitle_height = create_bilingual_subtitle_image(
                zh_text=zh_text, en_text=en_text, width=1720, height=120)
            if subtitle_path:
                final_subtitle_path = os.path.join(
                    subtitle_dir, f'bilingual_subtitle_{i+1}.png')
                shutil.move(subtitle_path, final_subtitle_path)
                subtitle_paths.append(final_subtitle_path)
                subtitle_segments_list.append([final_subtitle_path])
            else:
                subtitle_paths.append(None)
                subtitle_segments_list.append([])

    # 8. 统计展示
    successful_renders = sum(1 for path in foreground_paths
                             if path and os.path.exists(path))
    total_renders = len(segments)
    print('\n制作统计:')
    print(f'  文案片段: {len(segments)}')
    print(
        f'  语音文件: {len([p for p in audio_paths if p and os.path.exists(p)])}')
    print(f'  动画渲染: {successful_renders}/{total_renders}')
    print(f'  字幕文件: {len([p for p in subtitle_paths if p])}')

    # 人工模式特殊处理：启动人工动画工作室
    if mode == AnimationProductionMode.HUMAN_CONTROLLED:
        print(f'\n人工控制模式：已创建 {len([p for p in foreground_paths if p])} 个占位符')
        print('现在启动人工动画制作工作室...')

        # 先生成带占位符的预览视频
        print('生成带占位符的预览视频...')
        preview_path = os.path.join(full_output_dir,
                                    'preview_with_placeholders.mp4')

        enhanced_video_path = compose_final_video(unified_background_path,
                                                  foreground_paths,
                                                  audio_paths, subtitle_paths,
                                                  illustration_paths, segments,
                                                  preview_path,
                                                  subtitle_segments_list)

        if enhanced_video_path and os.path.exists(enhanced_video_path):
            print(f'占位符预览视频已生成: {enhanced_video_path}')
            print('你可以先查看这个预览视频了解整体效果')

            # 询问是否要立即启动人工工作室
            print('\n' + '=' * 60)
            print('动画制作选项:')
            print('1. 现在启动人工工作室制作动画')
            print('2. 稍后手动启动工作室')
            print('3. 直接使用占位符生成最终视频')

            try:
                choice = input('请选择 (1-3): ').strip()

                if choice == '1':
                    # 立即启动人工工作室
                    print('\n启动人工动画制作工作室...')
                    try:
                        from human_animation_studio import AnimationStudio
                        print('人工动画工作室已准备就绪')
                        print('你可以使用工作室的各种功能来制作动画')

                        # 工作室退出后检查是否有完成的动画，重新合成最终视频
                        print('\n检查动画制作结果...')
                        updated_foreground_paths = []

                        for i, segment in enumerate(segments):
                            if segment['type'] == 'text':
                                updated_foreground_paths.append(None)
                                continue

                            # 检查是否有完成的动画
                            final_animation_path = os.path.join(
                                animation_studio.finals_dir,
                                f'scene_{i+1}_final.mov')
                            if os.path.exists(final_animation_path):
                                updated_foreground_paths.append(
                                    final_animation_path)
                                print(f'发现完成的动画: scene_{i+1}')
                            else:
                                updated_foreground_paths.append(
                                    foreground_paths[i])  # 保持占位符

                        # 重新合成最终视频
                        final_video_path = os.path.join(
                            full_output_dir, 'final.mp4')
                        enhanced_video_path = compose_final_video(
                            unified_background_path, updated_foreground_paths,
                            audio_paths, subtitle_paths, illustration_paths,
                            segments, final_video_path, subtitle_segments_list)

                        if enhanced_video_path and os.path.exists(
                                enhanced_video_path):
                            print(f'最终视频已更新: {enhanced_video_path}')

                    except ImportError as e:
                        print(f' 无法启动人工工作室: {e}')
                        print(
                            "请手动运行: python human_animation_studio.py \"项目目录\"")
                    except Exception as e:
                        print(f' 人工工作室运行出错: {e}')

                elif choice == '2':
                    print('\n你可以稍后运行以下命令启动人工工作室:')
                    print(
                        f"python human_animation_studio.py \"{full_output_dir}\""
                    )

                elif choice == '3':
                    print('\n继续使用占位符生成最终视频...')

            except KeyboardInterrupt:
                print('\n用户中断，继续生成带占位符的最终视频...')

        return full_output_dir  # 人工模式在这里返回，后续的自动合成逻辑跳过

    # 9. 合成最终视频 (仅自动模式)
    if successful_renders > 0 or len(audio_paths) > 0:
        print('\n开始合成最终视频...')
        if unified_background_path:
            final_video_path = os.path.join(full_output_dir, 'final.mp4')

            enhanced_video_path = compose_final_video(
                unified_background_path, foreground_paths, audio_paths,
                subtitle_paths, illustration_paths, segments, final_video_path,
                subtitle_segments_list)
            if enhanced_video_path and os.path.exists(enhanced_video_path):
                bg_music_path = os.path.join(video_agent_dir, 'asset',
                                             'bg_audio.mp3')
                if os.path.exists(bg_music_path):
                    final_with_music = os.path.join(full_output_dir,
                                                    'final_with_music.mp4')
                    add_background_music(
                        enhanced_video_path,
                        final_with_music,
                        music_volume=0.15)
                    print(f'视频合成完成（含背景音乐）: {final_with_music}')
                else:
                    print(f'视频合成完成: {enhanced_video_path}')
                    print('未找到背景音乐文件，跳过背景音乐')
    segments_path = os.path.join(full_output_dir, 'segments.json')
    with open(segments_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print('\n视频生成完成！')
    print(f'输出目录: {full_output_dir}')
    return full_output_dir


def fix_latex_issues_in_manim_code(code, scene_name):
    """
    修复Manim代码中的LaTeX/MiKTeX问题
    """
    print('修复LaTeX相关问题...')

    def extract_formula_content(code_content):
        """从代码中提取公式内容"""
        formulas = []
        patterns = [
            r'MathTex\(r?"([^"]*?)"\)',
            r"MathTex\(r?'([^']*?)'\)",
            r'MathTex\("([^"]*?)"\)',
            r"MathTex\('([^']*?)'\)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, code_content)
            for match in matches:
                clean_content = match.replace('\\\\', '').replace('\\', '')
                clean_content = clean_content.replace('{', '').replace('}', '')
                clean_content = clean_content.replace('frac', '分数').replace(
                    'sqrt', '√')
                formulas.append(clean_content)

        return formulas

    def replace_mathtex_with_text(code_content):
        """将MathTex替换为Text"""
        code_content = re.sub(r'from manim import.*MathTex.*',
                              'from manim import *', code_content)

        def replace_mathtex(match):
            formula_content = match.group(1)
            clean_content = formula_content.replace('\\\\',
                                                    '').replace('\\', '')
            clean_content = clean_content.replace('{', '').replace('}', '')
            clean_content = clean_content.replace('frac',
                                                  '分数').replace('sqrt', '√')
            clean_content = clean_content.replace('r"', '').replace("r'", '')
            return f'Text("{clean_content}", font="Arial", color=WHITE, font_size=24)'

        patterns = [
            r'MathTex\(r?"([^"]*?)"\)',
            r"MathTex\(r?'([^']*?)'\)",
            r'MathTex\("([^"]*?)"\)',
            r"MathTex\('([^']*?)'\)",
        ]

        for pattern in patterns:
            code_content = re.sub(pattern, replace_mathtex, code_content)
        return code_content

    try:
        if 'MathTex' in code:
            print('检测到MathTex，尝试智能替换...')

            formulas = extract_formula_content(code)
            print(f'提取到 {len(formulas)} 个公式: {formulas}')

            fixed_code = replace_mathtex_with_text(code)
            if 'MathTex' not in fixed_code and 'Text' in fixed_code:
                print('MathTex成功替换为Text')
                return fixed_code

        print('使用LaTeX增强修复模板...')
        formulas = extract_formula_content(code) if 'MathTex' in code else []
        formula_display = formulas[0] if formulas else '数学公式'
        fixed_code_template = f'''# -*- coding: utf-8 -*-

# LaTeX修复版本 - 使用Text替代MathTex避免LaTeX依赖

import sys
import os

if hasattr(sys, 'setdefaultencoding'):
    sys.setdefaultencoding('utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'


from manim import *

class {scene_name}(Scene):
    def construct(self):
        # LaTeX修复：使用Text类替代MathTex，避免LaTeX依赖问题
        # 创建主要内容展示

        main_content = Text("{formula_display}", font_size=32, color=BLUE)
        main_content.move_to(ORIGIN)

        # 创建装饰性元素
        bg_rect = RoundedRectangle(
            width=main_content.width + 1,
            height=main_content.height + 0.8,
            corner_radius=0.2,
            color=BLUE,
            fill_opacity=0.1,
            stroke_width=2
        )

        bg_rect.move_to(main_content.get_center())

        # 动画序列
        self.play(DrawBorderThenFill(bg_rect), run_time=1.5)
        self.play(Write(main_content), run_time=2)
        self.wait(2)

        # 添加强调效果
        self.play(Indicate(main_content, scale_factor=1.2), run_time=1)
        self.wait(2)

        # 结束动画
        self.play(FadeOut(main_content), FadeOut(bg_rect), run_time=1.5)

'''
        return fixed_code_template

    except Exception as e:
        print(f'LaTeX修复过程中出错: {e}')
        return f'''# -*- coding: utf-8 -*-

from manim import *

class {scene_name}(Scene):
    def construct(self):
        text = Text("内容展示", font_size=36, color=WHITE)
        self.play(Write(text))
        self.wait(3)
        self.play(FadeOut(text))
'''


def fix_manim_error_with_llm(code,
                             error_message,
                             content_type,
                             scene_name,
                             enable_layout_optimization: bool = True):
    """
    使用LLM修复Manim错误 - 可控制是否进行布局优化
    """

    print(f'开始使用LLM修复 {scene_name} 的错误...')

    # 布局优化功能已集成到 balanced_spatial_system
    layout_issues = []
    if enable_layout_optimization:
        print('布局优化功能已启用，采用 balanced_spatial_system 的策略')
    else:
        print('布局优化功能已关闭，本次仅修复渲染/语法错误，不改动布局')

    latex_error_keywords = [
        'MiKTeX', 'latex error', 'pdflatex', 'LaTeX Error', 'MathTex rendering'
    ]
    is_latex_error = any(keyword in error_message
                         for keyword in latex_error_keywords)

    if is_latex_error and ('MathTex' in code):
        print('检测到明确的LaTeX渲染错误，使用LaTeX修复策略...')
        return fix_latex_issues_in_manim_code(code, scene_name)

    print('使用智能LLM修复...'
          + ('(包含布局优化)' if enable_layout_optimization else '(不进行布局优化)'))

    # 构建修复提示
    fix_prompt = f"""你是Manim调试专家。分析以下代码和错误信息，提供修复方案。

场景名称: {scene_name}
内容类型: {content_type}

Manim代码:
{code}

完整报错traceback（含stderr）:
{error_message}
"""

    # 如果启用布局优化，添加到提示中通用的布局要求
    if enable_layout_optimization:
        fix_prompt += f"""

检测到的布局问题:
{chr(10).join(f"- {issue}" for issue in layout_issues)}

## 布局优化要求
1. 使用区域化布局系统，避免元素重叠
2. 每个文本元素用to_edge()或适当的相对定位
3. 保持最小间距：垂直≥0.4，水平≥0.5
4. 画面边界：left≥-6.0, right≤6.0, top≤3.5, bottom≥-3.5
5. 分段清理：每个概念讲完后用FadeOut清理元素"""
    else:
        fix_prompt += """

注意：本次修复仅限于解决渲染失败或语法错误，请尽量不要更改现有布局与排版；保持元素位置、整体风格与现有代码一致。
"""

    fix_prompt += """

请提供修复后的完整代码，要求：
1. 修复所有语法和逻辑错误
2. 保持原有功能不变，不要简化内容
3. 如果有MathTex错误，替换为Text但保持原始内容
4. 确保动画生动有趣，体现原始内容的价值
5. 代码完整可执行
"""

    if enable_layout_optimization:
        fix_prompt += """
附加（若可能）：
- 优化布局，避免元素冲突与越界
- 使用合理的空间管理与间距
"""

    fix_prompt += """
请直接返回修复后的完整Python代码：
"""

    try:
        fix_result = modai_model_request(
            fix_prompt, max_tokens=2048, temperature=0.1)

        # 清理LLM输出
        cleaned_code = clean_llm_code_output(fix_result)

        if cleaned_code and 'from manim import' in cleaned_code and 'class' in cleaned_code:
            print('错误和布局问题修复完成')
            return cleaned_code
        else:
            print('修复后的代码不完整或格式错误')
            return None

    except Exception as e:  # noqa
        print(f'LLM修复失败: {e}')
        return None


def compose_final_video(background_path, foreground_paths, audio_paths,
                        subtitle_paths, illustration_paths, segments,
                        output_path, subtitle_segments_list):
    """
    合成插画+动画的最终视频
    """

    try:
        import moviepy.editor as mp

        print('开始合成最终视频...')
        segment_durations = []
        total_duration = 0

        for i, audio_path in enumerate(audio_paths):
            actual_duration = 3.0

            if audio_path and os.path.exists(audio_path):
                try:
                    audio_clip = mp.AudioFileClip(audio_path)
                    actual_duration = max(audio_clip.duration, 3.0)
                    audio_clip.close()
                except:  # noqa
                    actual_duration = 3.0

            if i < len(foreground_paths
                       ) and foreground_paths[i] and os.path.exists(
                           foreground_paths[i]):
                try:
                    animation_clip = mp.VideoFileClip(
                        foreground_paths[i], has_mask=True)
                    animation_duration = animation_clip.duration
                    animation_clip.close()

                    if animation_duration > actual_duration:
                        actual_duration = animation_duration
                        print(f'片段 {i+1} 使用动画时长: {actual_duration:.1f}秒')
                except:  # noqa
                    pass

            segment_durations.append(actual_duration)
            total_duration += actual_duration

        print(f'总时长: {total_duration:.1f}秒，{len(segment_durations)}个片段')
        print('重新组织合成逻辑...')

        print('步骤1：合成每个片段的完整视频...')
        segment_videos = []

        for i, (duration,
                segment) in enumerate(zip(segment_durations, segments)):
            print(
                f"合成片段 {i+1}: {segment.get('type', 'unknown')} - {duration:.1f}秒"
            )

            current_video_clips = []

            if background_path and os.path.exists(background_path):
                bg_clip = mp.ImageClip(background_path, duration=duration)
                bg_clip = bg_clip.resize((1920, 1080))
                current_video_clips.append(bg_clip)

            if segment.get('type') == 'text' and i < len(
                    illustration_paths
            ) and illustration_paths[i] and os.path.exists(
                    illustration_paths[i]):
                try:
                    illustration_clip = mp.ImageClip(
                        illustration_paths[i], duration=duration)
                    original_w, original_h = illustration_clip.size
                    available_w, available_h = 1920, 800
                    scale_w = available_w / original_w
                    scale_h = available_h / original_h
                    scale = min(scale_w, scale_h, 1.0)

                    if scale < 1.0:
                        new_w = int(original_w * scale)
                        new_h = int(original_h * scale)
                        illustration_clip = illustration_clip.resize(
                            (new_w, new_h))
                    else:
                        new_w, new_h = original_w, original_h

                    # 向左运动动画
                    exit_duration = 1.0
                    start_animation_time = max(duration - exit_duration, 0)
                    print(
                        f'调试: 片段时长={duration:.2f}秒, 退出动画时长={exit_duration}秒, 动画开始时间={start_animation_time:.2f}秒'
                    )
                    print(
                        f'调试: 插画静止时间={start_animation_time:.2f}秒, 动画运动时间={exit_duration}秒'
                    )

                    def illustration_pos_factory(idx, start_x, end_x, new_h,
                                                 start_animation_time,
                                                 exit_duration):

                        def illustration_pos(t):
                            y = (1080 - new_h) // 2
                            if t < start_animation_time:
                                x = start_x
                                print(
                                    f'调试: 片段{idx}  时间={t:.2f}秒, 静止位置=({x}, {y})，动画将在{start_animation_time:.2f}秒开始'
                                )
                            elif t < start_animation_time + exit_duration:
                                progress = (
                                    t - start_animation_time) / exit_duration
                                progress = min(max(progress, 0), 1)  # 限制在0~1
                                x = start_x + (end_x - start_x) * progress
                                print(
                                    f'调试: 片段{idx}  时间={t:.2f}秒, 运动位置=({x}, {y})，进度={progress:.1%}'
                                )
                            else:
                                x = end_x
                                print(
                                    f'调试: 片段{idx}  时间={t:.2f}秒, 已运动结束，插画在屏幕外 ({x}, {y})'
                                )
                            return (x, y)

                        return illustration_pos

                    print(
                        f'插画动画设置: 片段时长 {duration:.1f}秒，动画在最后 {exit_duration}秒开始'
                    )
                    illustration_clip = illustration_clip.set_position(
                        illustration_pos_factory(i, (1920 - new_w) // 2,
                                                 -new_w, new_h,
                                                 start_animation_time,
                                                 exit_duration))
                    current_video_clips.append(illustration_clip)
                    print('添加插画层')
                except Exception as e:
                    print(f'插画加载失败: {e}')

            elif segment.get('type') != 'text' and i < len(
                    foreground_paths
            ) and foreground_paths[i] and os.path.exists(foreground_paths[i]):
                try:
                    fg_clip = mp.VideoFileClip(
                        foreground_paths[i], has_mask=True)
                    original_w, original_h = fg_clip.size
                    available_w, available_h = 1920, 800
                    scale_w = available_w / original_w
                    scale_h = available_h / original_h
                    scale = min(scale_w, scale_h, 1.0)

                    if scale < 1.0:
                        new_w = int(original_w * scale)
                        new_h = int(original_h * scale)
                        fg_clip = fg_clip.resize((new_w, new_h))

                    fg_clip = fg_clip.set_position(('center', 'center'))
                    fg_clip = fg_clip.set_duration(duration)
                    current_video_clips.append(fg_clip)
                    print('添加动画层')
                except Exception as e:
                    print(f'动画加载失败: {e}')

            if segment.get('type') != 'text' and i < len(
                    subtitle_segments_list):
                try:
                    subtitle_imgs = subtitle_segments_list[i]
                    if subtitle_imgs and isinstance(
                            subtitle_imgs, list) and len(subtitle_imgs) > 0:
                        n = len(subtitle_imgs)
                        seg_duration = duration / n
                        for idx, subtitle_path in enumerate(subtitle_imgs):
                            try:
                                from PIL import Image
                                subtitle_img = Image.open(subtitle_path)
                                subtitle_w, subtitle_h = subtitle_img.size
                                subtitle_clip = mp.ImageClip(
                                    subtitle_path, duration=seg_duration)
                                subtitle_clip = subtitle_clip.resize(
                                    (subtitle_w, subtitle_h))
                                subtitle_y = 850
                                print(f'字幕位置设置为 y={subtitle_y}')
                                subtitle_clip = subtitle_clip.set_position(
                                    ('center', subtitle_y))
                                subtitle_clip = subtitle_clip.set_start(
                                    idx * seg_duration)
                                current_video_clips.append(subtitle_clip)
                                print(f'添加动画片段字幕 {idx+1}/{n}')
                            except Exception as e:
                                print(f'动画片段字幕 {idx+1} 处理失败: {e}')
                    else:
                        print(f'动画片段 {i+1} 没有有效字幕，跳过字幕层')
                except Exception as e:
                    print(f'动画片段 {i+1} 字幕处理异常: {e}')
            else:
                if i < len(subtitle_paths
                           ) and subtitle_paths[i] and os.path.exists(
                               subtitle_paths[i]):
                    try:
                        from PIL import Image
                        subtitle_img = Image.open(subtitle_paths[i])
                        subtitle_w, subtitle_h = subtitle_img.size
                        subtitle_clip = mp.ImageClip(
                            subtitle_paths[i], duration=duration)
                        subtitle_clip = subtitle_clip.resize(
                            (subtitle_w, subtitle_h))
                        subtitle_y = 850
                        print(f'字幕位置设置为 y={subtitle_y}')
                        subtitle_clip = subtitle_clip.set_position(
                            ('center', subtitle_y))
                        current_video_clips.append(subtitle_clip)
                        print('添加字幕层（底部对齐）')
                    except Exception as e:
                        print(f'字幕加载失败: {e}')

            if current_video_clips:
                segment_video = mp.CompositeVideoClip(
                    current_video_clips, size=(1920, 1080))
                segment_videos.append(segment_video)
                print(f'片段 {i+1} 合成完成')
            else:
                print(f'片段 {i+1} 无有效内容，跳过')

        if not segment_videos:
            print('没有有效的视频片段')
            return None

        print('  步骤2：按时间顺序连接所有片段...')
        final_video = mp.concatenate_videoclips(
            segment_videos, method='compose')
        print(f'视频连接完成，总时长: {final_video.duration:.1f}秒')

        print('步骤3：合成音频...')
        if audio_paths:
            try:
                print(f'连接 {len(audio_paths)} 个音频片段...')
                valid_audio_clips = []
                for i, (audio_path, duration) in enumerate(
                        zip(audio_paths, segment_durations)):
                    try:
                        if audio_path and os.path.exists(audio_path):
                            audio_clip = mp.AudioFileClip(audio_path)
                            audio_clip = audio_clip.set_fps(44100)
                            try:
                                audio_clip = audio_clip.set_channels(2)
                            except Exception:
                                pass
                            if audio_clip.duration > duration:
                                audio_clip = audio_clip.subclip(0, duration)
                            elif audio_clip.duration < duration:
                                from moviepy.editor import AudioClip
                                silence = AudioClip(
                                    lambda t: [0, 0],
                                    duration=duration
                                    - audio_clip.duration).set_fps(44100)
                                try:
                                    silence = silence.set_channels(2)
                                except Exception:
                                    pass
                                audio_clip = mp.concatenate_audioclips(
                                    [audio_clip, silence])
                            valid_audio_clips.append(audio_clip)
                            print(f'音频片段 {i+1}: {audio_clip.duration:.2f}s')
                        else:
                            print(f'音频片段 {i+1} 无效，跳过')
                    except Exception as e:
                        print(f'音频片段 {i+1} 处理失败: {e}')
                        from moviepy.editor import AudioClip
                        silence = AudioClip(
                            lambda t: [0, 0], duration=duration).set_fps(44100)
                        valid_audio_clips.append(silence)

                if valid_audio_clips:
                    final_audio = mp.concatenate_audioclips(valid_audio_clips)
                    print(f'音频连接完成，总时长: {final_audio.duration:.1f}秒')
                    if final_audio.duration > final_video.duration:
                        final_audio = final_audio.subclip(
                            0, final_video.duration)
                        print('音频已裁剪到视频时长')
                    elif final_audio.duration < final_video.duration:
                        from moviepy.editor import AudioClip
                        silence = AudioClip(
                            lambda t: [0, 0],
                            duration=final_video.duration
                            - final_audio.duration)
                        final_audio = mp.concatenate_audioclips(
                            [final_audio, silence])
                        print('音频已补足到视频时长')

                    final_video = final_video.set_audio(final_audio)
                    print(
                        f'音频合成成功，时长: {final_audio.duration:.1f}秒 (视频: {final_video.duration:.1f}秒)'
                    )
                else:
                    print('没有有效音频，生成静音视频')
            except Exception as e:
                print(f'音频合成失败: {e}')
                print('将生成无声视频')
        else:
            print('没有音频片段，生成静音视频')

        try:
            import moviepy.audio.fx.all as afx
            bg_music_path = os.path.join(
                os.path.dirname(__file__), 'asset', 'bg_audio.mp3')
            if os.path.exists(bg_music_path):
                print('添加背景音乐...')
                bg_music = mp.AudioFileClip(bg_music_path)
                bg_music = afx.audio_loop(
                    bg_music, duration=final_video.duration)
                bg_music = bg_music.volumex(0.2)
                if final_video.audio:
                    tts_audio = final_video.audio.set_duration(
                        final_video.duration).volumex(1.0)
                    bg_audio = bg_music.set_duration(
                        final_video.duration).volumex(0.15)
                    mixed_audio = mp.CompositeAudioClip(
                        [tts_audio,
                         bg_audio]).set_duration(final_video.duration)
                else:
                    mixed_audio = bg_music.set_duration(
                        final_video.duration).volumex(0.3)
                final_video = final_video.set_audio(mixed_audio)
                print('背景音乐添加完成')
            else:
                print('未找到背景音乐文件')
        except Exception as e:
            print(f'背景音乐添加失败: {e}')

        print('渲染最终视频...')
        if final_video is None:
            print('错误: final_video为None，无法渲染')
            return None

        try:
            print(f'视频总时长: {final_video.duration:.1f}秒')
            print(f'视频分辨率: {final_video.size}')
            print(f"音频状态: {'有音频' if final_video.audio else '无音频'}")
            print(f'final_video类型: {type(final_video)}')
            print(f'final_video属性: {dir(final_video)}')

            if final_video.audio:
                print(f'音频类型: {type(final_video.audio)}')
                print(f'音频时长: {final_video.audio.duration:.1f}秒')
                try:
                    audio_fps = final_video.audio.fps
                    print(f'音频采样率: {audio_fps} Hz')
                except AttributeError:
                    if hasattr(final_video.audio,
                               'clips') and final_video.audio.clips:
                        first_clip = final_video.audio.clips[0]
                        if hasattr(first_clip, 'fps'):
                            print(f'首个音频片段采样率: {first_clip.fps} Hz')
        except Exception as e:
            print(f'错误: 获取视频信息失败: {e}')
            import traceback
            traceback.print_exc()
            return None

        try:
            print(f'开始渲染到: {output_path}')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            final_video.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                logger=None,
                verbose=False,
                threads=2,
                bitrate='5000k',
                audio_bitrate='192k',
                audio_fps=44100,
                write_logfile=False)

            print(f'视频渲染完成: {output_path}')
            if os.path.exists(
                    output_path) and os.path.getsize(output_path) > 1024:
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f'文件大小: {file_size_mb:.1f} MB')

                try:
                    test_clip = mp.VideoFileClip(output_path)
                    actual_duration = test_clip.duration
                    test_clip.close()
                    print(
                        f'实际时长: {actual_duration:.1f}秒 (预期: {final_video.duration:.1f}秒)'
                    )

                    if abs(actual_duration - final_video.duration) < 1.0:
                        print('视频文件验证通过')
                        return output_path
                    else:
                        print('视频时长不匹配，但文件已生成')
                        return output_path
                except Exception as e:
                    print(f'视频文件验证失败: {e}')
                    return output_path
            else:
                print('视频文件生成失败或文件过小')
                return None

        except Exception as e:
            print(f'视频渲染失败: {e}')
            import traceback
            traceback.print_exc()

            try:
                print('尝试生成无音频视频...')
                final_video = final_video.set_audio(None)
                final_video.write_videofile(
                    output_path,
                    fps=24,
                    codec='libx264',
                    audio_codec=None,
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    logger=None,
                    verbose=False,
                    threads=2,
                    bitrate='5000k',
                    write_logfile=False)
                print(f'无音频视频渲染完成: {output_path}')
                return output_path
            except Exception as e2:
                print(f'无音频视频渲染也失败: {e2}')
                traceback.print_exc()
                return None

    except Exception as e:
        print(f'视频合成失败: {e}')
        return None


def keep_only_black_for_folder(input_dir, output_dir, threshold=80):
    """去插画背景"""
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_dir, fname)
            base_name, _ = os.path.splitext(fname)
            output_png = os.path.join(output_dir, base_name + '.png')
            try:
                img = Image.open(input_path).convert('RGBA')
                arr = np.array(img)

                print(f'处理图片: {fname}')
                print(f'  原始尺寸: {img.size}')
                print(f'  原始模式: {img.mode}')
                print(
                    f'  颜色范围: R[{arr[..., 0].min()}-{arr[..., 0].max()}], G[{arr[..., 1].min()}-{arr[..., 1].max()}]'
                    f', B[{arr[..., 2].min()}-{arr[..., 2].max()}]')

                gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[
                    ..., 2]
                mask = gray < threshold

                transparent_pixels = np.sum(mask)
                total_pixels = mask.size
                transparency_ratio = transparent_pixels / total_pixels
                print(
                    f'检测到黑色像素: {transparent_pixels}/{total_pixels} ({transparency_ratio:.1%})'
                )

                arr[..., 3] = np.where(mask, 255, 0)

                img2 = Image.fromarray(arr, 'RGBA')
                img2.save(output_png, 'PNG')

                if os.path.exists(output_png):
                    output_size = os.path.getsize(output_png)
                    print(f'输出文件: {output_png} ({output_size} bytes)')

                    try:
                        output_img = Image.open(output_png)
                        output_arr = np.array(output_img)
                        if output_img.mode == 'RGBA':
                            alpha_channel = output_arr[..., 3]
                            unique_alpha = np.unique(alpha_channel)
                            print(f'透明通道值: {unique_alpha}')
                        else:
                            print(f'警告: 输出图片不是RGBA模式: {output_img.mode}')
                    except Exception as verify_e:
                        print(f'验证输出文件失败: {verify_e}')

                print(f'处理完成: {fname} -> 保留黑色部分，背景透明')

            except Exception as e:
                print(f'处理图片失败: {input_path}, 错误: {e}')
                try:
                    backup_img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
                    backup_img.save(output_png, 'PNG')
                    print(f'创建备用透明图片: {output_png}')
                except:  # noqa
                    pass


def generate_illustration_prompts(segments):
    prompts = []
    system_prompt = """You is a scene description expert for AI knowledge science stickman videos. Based on the given knowledge point or storyboard, generate a detailed English description for a minimalist black-and-white stickman illustration with an AI/technology theme. Requirements:
- The illustration must depict only ONE scene, not multiple scenes, not comic panels, not split images. Absolutely do NOT use any comic panels, split frames, multiple windows, or any kind of visual separation. Each image is a single, unified scene.
- All elements (stickmen, objects, icons, patterns, tech elements, decorations) must appear together in the same space, on the same pure white background, with no borders, no frames, and no visual separation.
- All icons, patterns, and objects are decorative elements floating around or near the stickman, not separate scenes or frames. For example, do NOT draw any boxes, lines, or frames that separate parts of the image. All elements must be together in one open space.
- The background must be pure white. Do not describe any darkness, shadow, dim, black, gray, or colored background. Only describe a pure white background.
- All elements (stickmen, objects, tech elements, decorations) must be either solid black fill or outlined in black, to facilitate cutout. No color, no gray, no gradients, no shadows.
- The number of stickman characters should be chosen based on the meaning of the sentence: if the scene is suitable for a single person, use only one stickman; if it is suitable for interaction, use two or three stickmen. Do not force two or more people in every scene.
- All stickman characters must be shown as FULL BODY, with solid black fill for both body and face.
- Each stickman has a solid black face, with white eyes and a white mouth, both drawn as white lines. Eyes and mouth should be irregular shapes to express different emotions, not just simple circles or lines. Use these white lines to show rich, varied, and vivid emotions.
- Do NOT include any speech bubbles, text bubbles, comic panels, split images, or multiple scenes.
- All characters and elements must be fully visible, not cut off or overlapped.
"- Only add clear, readable English text in the image if it is truly needed to express the knowledge point or scene meaning, such as AI, Token, LLM, or any other relevant English word. Do NOT force the use of any specific word in every scene. If no text is needed, do not include any text. "
- All text in the image must be clear, readable, and not distorted, garbled, or random.
- Scene can include rich, relevant, and layered minimalist tech/AI/futuristic elements (e.g., computer, chip, data stream, AI icon, screen, etc.), and simple decorative elements to enhance atmosphere, but do not let elements overlap or crowd together.
- All elements should be relevant to the main theme and the meaning of the current subtitle segment.
- Output 80-120 words in English, only the scene description, no style keywords, and only use English text in the image if it is truly needed for the scene. """ # noqa

    for seg in segments:
        prompt = (
            f'Please generate a detailed English scene description for an AI knowledge science stickman '
            f'illustration based on: {seg}\nRemember: The illustration must depict only ONE scene, '
            f'not multiple scenes, not comic panels, not split images. Absolutely do NOT use any comic panels, '
            f'split frames, multiple windows, or any kind of visual separation. '
            f'All elements must be solid black or outlined in black, and all faces must use irregular '
            f'white lines for eyes and mouth to express emotion. All elements should be relevant to the '
            f'main theme and the meaning of the current subtitle segment. All icons, patterns, and objects '
            f'are decorative elements floating around or near the stickman, not separate scenes or frames. '
            f'For example, do NOT draw any boxes, lines, or frames that separate parts of the image. '
            f'All elements must be together in one open space.')
        desc = modai_model_request(
            prompt,
            model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
            max_tokens=256,
            temperature=0.5,
            system_prompt=system_prompt)
        prompts.append(desc.strip())
    return prompts


def generate_images(prompts,
                    model_id='AIUSERS/jianbihua',
                    negative_prompt=None,
                    output_dir=None):
    import os
    import requests
    import time
    import json
    from PIL import Image
    from io import BytesIO
    if output_dir:
        save_dir = os.path.join(output_dir, 'images')
    else:
        save_dir = os.path.join(os.path.dirname(__file__), 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    base_url = 'https://api-inference.modelscope.cn/'
    import os
    api_key = os.environ.get('MODELSCOPE_API_KEY')
    if not api_key:
        raise ValueError('请设置环境变量 MODELSCOPE_API_KEY')
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    def create_placeholder(path):
        img = Image.new('RGB', (512, 512), (255, 255, 255))
        img.save(path)

    results = []
    for idx, desc in enumerate(prompts):
        prompt = desc
        img_path = os.path.join(save_dir, f'illustration_{idx+1}.jpg')
        try:
            resp = requests.post(
                f'{base_url}v1/images/generations',
                headers={
                    **headers, 'X-ModelScope-Async-Mode': 'true'
                },
                data=json.dumps(
                    {
                        'model': model_id,
                        'prompt': prompt,
                        'negative_prompt': negative_prompt or ''
                    },
                    ensure_ascii=False).encode('utf-8'))
            resp.raise_for_status()
            task_id = resp.json()['task_id']
            for _ in range(30):
                result = requests.get(
                    f'{base_url}v1/tasks/{task_id}',
                    headers={
                        **headers, 'X-ModelScope-Task-Type': 'image_generation'
                    },
                )
                result.raise_for_status()
                data = result.json()
                if data['task_status'] == 'SUCCEED':
                    img_url = data['output_images'][0]
                    image = Image.open(BytesIO(requests.get(img_url).content))
                    image.save(img_path)
                    results.append(img_path)
                    break
                elif data['task_status'] == 'FAILED':
                    print(f'插画生成失败: {desc}')
                    create_placeholder(img_path)
                    results.append(img_path)
                    break
                time.sleep(5)
            else:
                print(f'插画生成超时: {desc}')
                create_placeholder(img_path)
                results.append(img_path)
        except Exception as e:
            print(f'插画生成异常: {e}')
            create_placeholder(img_path)
            results.append(img_path)
    return results


# 启动
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        topic = sys.argv[1]
        if len(sys.argv) > 2:
            output_dir = sys.argv[2]
        else:
            output_dir = 'output'
        if len(sys.argv) > 3:
            animation_mode = sys.argv[3]  # "auto", "human"
        else:
            animation_mode = 'auto'
    else:
        topic = 'MCP'
        output_dir = 'output'
        animation_mode = 'auto'

    print('\n AI知识科普视频生成系统')
    print('=' * 50)
    print(f'主题: {topic}')
    print(f'输出目录: {output_dir}')
    print(f'动画模式: {animation_mode}')

    if animation_mode == 'human':
        print('\n人工控制模式说明:')
        print('- 系统将生成占位符代替动画')
        print('- 完成基础视频后会启动人工动画工作室')
        print('- 你可以与AI对话制作每个动画')
        print('- 支持预览、修改、批准流程')
    else:
        print('\n自动模式: 全自动生成所有内容')

    print('=' * 50)

    try:
        output_path = generate_ai_science_knowledge_video(
            topic, output_dir, animation_mode)
        if output_path:
            print('\n 全部完成！')
            print(f'输出目录：{output_path}')

            # 根据模式显示不同的结果文件
            if animation_mode == 'human':
                preview_file = os.path.join(output_path,
                                            'preview_with_placeholders.mp4')
                if os.path.exists(preview_file):
                    print(f'占位符预览：{preview_file}')
                final_file = os.path.join(output_path, 'final.mp4')
                if os.path.exists(final_file):
                    print(f'最终视频：{final_file}')
                print('\n如需制作动画，请运行:')
                print(f"python human_animation_studio.py \"{output_path}\"")
            else:
                print(f"视频文件：{os.path.join(output_path, 'final.mp4')}")
                if os.path.exists(
                        os.path.join(output_path, 'final_with_music.mp4')):
                    print(
                        f"带背景音乐：{os.path.join(output_path, 'final_with_music.mp4')}"
                    )
        else:
            print('\n 视频生成失败')
    except KeyboardInterrupt:
        print('\n用户中断程序')
    except Exception as e:
        print(f'\n 程序运行出错: {e}')
        import traceback
        traceback.print_exc()
