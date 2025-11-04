import re
from typing import List, Dict

from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM
from ms_agent.llm.openai_llm import OpenAI


class AnalyzeCode(CodeAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def run(self, inputs, **kwargs):
        code = inputs[-1].content
        lines = code.split('\n')
        issues = self.detect_layout_issues(code)

        # 基础统计
        element_count = len([
            line for line in lines
            if any(kw in line
                   for kw in ['Text(', 'Circle(', 'Rectangle(', 'VGroup('])
        ])

        # 计算分数
        layout_score = 85  # 基础分更高

        # 根据问题类型扣分
        for issue_group in issues:
            if '边界越界风险' in issue_group:
                layout_score -= 15
            elif '重叠风险' in issue_group:
                layout_score -= 10
            elif '布局拥挤' in issue_group:
                layout_score -= 8
            elif '复杂度问题' in issue_group:
                layout_score -= 12
            elif '动画节奏问题' in issue_group:
                layout_score -= 5

        # 内容丰富度加分
        if element_count > 0:
            layout_score += min(element_count * 1, 15)

        layout_score = max(0, min(100, layout_score))

        # 计算间距问题数量
        spacing_issues = 0
        for issue_group in issues:
            if '重叠风险' in issue_group or '布局拥挤' in issue_group:
                spacing_issues += 1

        # 检查是否过度工程化
        is_over_engineered = False
        if element_count > 8 or len([
            line for line in lines
            if 'class' in line and line.strip().startswith('class')
        ]) > 1:
            is_over_engineered = True

        return {
            'layout_score': layout_score,
            'element_count': element_count,
            'spacing_issues': spacing_issues,
            'is_over_engineered': is_over_engineered,
            'issues': issues,
            'issue_count': len(issues),
            'needs_fix': len(issues) > 0,
            'fix_prompt':
                self.generate_fix_prompt(code, issues) if issues else ''
        }

    def generate_fix_prompt(self, code, issues):
        """根据发现的问题生成修复建议"""

        if not issues:
            return ''

        # 分析代码复杂度和内容丰富度
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

        richness_level = '丰富' if (text_count > 5 and animation_count > 5
                                    and color_count > 3) else '中等' if (
                text_count > 3) else '简单'

        fix_prompt = f"""**布局问题修复任务**

检测到以下问题需要修复:
{''.join(issues)}

**修复目标**:
• 解决所有检测到的布局问题
• 保持动画效果的丰富性和多样性（当前内容丰富度: {richness_level}）
• 保持代码简洁性，避免过度工程化
• 确保最终效果不失原有创意和表现力

**修复指导原则**:

**边界控制**:
• 安全区域: x ∈ (-6.5, 6.5), y ∈ (-3.5, 3.5)
• 字体大小: 建议12-48之间，标题可适当增大但不超过60
• 使用相对定位: to_edge(), next_to(), align_to() 代替绝对坐标

**重叠避免**:
• 元素间距: buff >= 0.3 (紧凑布局可用0.25)
• 避免多元素center(): 用arrange()或next_to()分布
• VGroup组织: 相关元素用VGroup统一管理位置

**布局优化**:
• 元素分层: 标题→主内容→补充信息的视觉层次
• 空间利用: 合理分布，避免集中堆积
• 动态调整: 根据内容量自适应布局

**保持动画丰富性**:
• 多样效果: Write, Create, FadeIn, Transform, Indicate 等搭配使用
• 颜色丰富: 保持现有的色彩搭配和强调效果
• 节奏控制: run_time和wait时间适中（1-3秒范围）
• 视觉亮点: 保持特效、高亮、动态变化等创意元素

**代码简化**:
• 直接实现: 在Scene类中直接完成，避免Helper类
• 合理组织: 相关功能可以适当封装，但不过度分解
• 清晰注释: 保持代码可读性

**修复策略建议**:
- 对于边界问题: 用to_edge()和shift()替代move_to()固定坐标
- 对于重叠问题: 增加buff参数，使用arrange()分布元素
- 对于拥挤问题: 适当使用分组显示或分步展示
- 对于复杂性: 简化结构但保持功能完整性

**创意保持要求**:
请确保修复过程中：
1. 保持所有动画效果和视觉创意
2. 不减少色彩和特效的使用
3. 维持内容的教学价值和表现力
4. 动画节奏保持流畅有趣

请返回完整的修复后代码，确保既解决了布局问题，又保持了动画的丰富性和创意性。"""
        return fix_prompt

    def detect_layout_issues(self, code):
        issues = []
        lines = code.split('\n')

        boundary_violations = []
        for i, line in enumerate(lines, 1):
            line_clean = line.strip()
            if not line_clean or line_clean.startswith('#'):
                continue

            # 检查move_to是否越界
            if re.search(r'\.move_to\(\[?\s*[+-]?([8-9]|[1-9]\d)', line):
                boundary_violations.append(f'第{i}行: 绝对位置越界 - {line_clean}')

            # shift太大也会出界
            if re.search(r'\.shift\(\s*[A-Z_]*\s*\*\s*([6-9]|[1-9]\d)', line):
                boundary_violations.append(f'第{i}行: shift位移过大 - {line_clean}')

            # 字体大小检查
            font_match = re.search(r'font_size\s*=\s*([0-9]+)', line)
            if font_match:
                size = int(font_match.group(1))
                if size > 64:  # 太大了会显示不全
                    boundary_violations.append(
                        f'第{i}行: 字体过大({size}) - {line_clean}')
                elif size < 10:  # 太小了看不清
                    boundary_violations.append(
                        f'第{i}行: 字体过小({size}) - {line_clean}')

        if boundary_violations:
            issues.append('边界越界风险:')
            issues.extend(f'   • {v}' for v in boundary_violations)

        # 通用重叠风险检测 (基于空间关系分析)
        overlap_risks = []

        # 1. 提取所有空间定位操作
        spatial_operations = self._extract_spatial_operations(lines)

        # 2. 分析空间关系冲突
        spatial_conflicts = self._analyze_spatial_conflicts(spatial_operations)

        # 3. 转换为可读的问题描述
        for conflict in spatial_conflicts:
            overlap_risks.append(self._format_conflict_description(conflict))

        if overlap_risks:
            issues.append('重叠风险:')
            issues.extend(f'   • {r}' for r in overlap_risks)

        # 检查拥挤问题
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

        if text_elements > 12:  # 文本太多了
            crowding_issues.append(f'文本元素过多 ({text_elements}个)，建议分组或分页')

        if total_elements > 20:  # 总数太多，画面会乱
            crowding_issues.append(f'画面元素过多 ({total_elements}个)，可能显示拥挤')

        if crowding_issues:
            issues.append('布局拥挤:')
            issues.extend(f'   • {c}' for c in crowding_issues)

        # 看看代码是不是太复杂了
        complexity_issues = []

        if 'Helper' in code or 'helper' in code.lower():
            helper_count = code.count('Helper') + code.count('helper')
            complexity_issues.append(f'使用了Helper类模式 ({helper_count}处)，建议简化')

        create_methods = code.count('def create_')
        if create_methods > 4:  # create方法太多
            complexity_issues.append(
                f'过多create方法 ({create_methods}个)，建议合并相关功能')

        # 检查VGroup嵌套层数
        vgroup_depth = 0
        max_depth = 0
        for line in lines:
            if 'VGroup(' in line:
                vgroup_depth += line.count('VGroup(')
                max_depth = max(max_depth, vgroup_depth)
            if ')' in line:
                vgroup_depth -= line.count(')')

        if max_depth > 3:
            complexity_issues.append(f'VGroup嵌套过深 ({max_depth}层)，建议简化结构')

        if complexity_issues:
            issues.append('复杂度问题:')
            issues.extend(f'   • {c}' for c in complexity_issues)

        # 动画时间检查
        animation_issues = []

        # 动画时间太长会拖节奏
        runtime_values = re.findall(r'run_time\s*=\s*([0-9.]+)', code)
        for runtime in runtime_values:
            if float(runtime) > 6.0:
                animation_issues.append(f'动画时间过长 ({runtime}秒)，可能影响节奏')

        wait_values = re.findall(r'self\.wait\(([0-9.]+)\)', code)
        for wait_time in wait_values:
            if float(wait_time) > 4.0:
                animation_issues.append(f'等待时间过长 ({wait_time}秒)，可能影响连贯性')

        if animation_issues:
            issues.append('动画节奏问题:')
            issues.extend(f'   • {a}' for a in animation_issues)

        return issues

    def _extract_spatial_operations(self, lines: List[str]) -> List[Dict]:
        """找出代码中所有的定位操作"""
        operations = []

        for i, line in enumerate(lines, 1):
            line_clean = line.strip()
            if not line_clean or line_clean.startswith('#'):
                continue

            # 提取对象名
            object_name = self._extract_object_name(line_clean)
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

            # 分析定位类型
            if '.move_to(' in line:
                operation['type'] = 'move_to'
                operation['reference'] = self._extract_move_to_target(
                    line_clean)
            elif '.next_to(' in line:
                operation['type'] = 'next_to'
                operation['reference'] = self._extract_next_to_reference(
                    line_clean)
                operation['has_spacing'] = 'buff=' in line
                operation['spacing_value'] = self._extract_buff_value(
                    line_clean)
            elif '.center()' in line:
                operation['type'] = 'center'
                operation['reference'] = 'ORIGIN'
            elif '.to_edge(' in line:
                operation['type'] = 'to_edge'
                operation['reference'] = self._extract_edge_direction(
                    line_clean)
            elif '.shift(' in line:
                operation['type'] = 'shift'
                operation['reference'] = 'relative'

            operations.append(operation)

        return operations

    def _extract_edge_direction(self, line):
        """提取边缘方向"""
        match = re.search(r'\.to_edge\(([^)]+)\)', line)
        return match.group(1).strip() if match else None

    def _extract_buff_value(self, line):
        """取出buff的值"""
        match = re.search(r'buff\s*=\s*([0-9.]+)', line)
        return float(match.group(1)) if match else None

    def _extract_next_to_reference(self, line):
        """找出next_to中的参考对象"""
        match = re.search(r'\.next_to\(([^,)]+)', line)
        return match.group(1).strip() if match else None

    def _extract_object_name(self, line):
        """从代码行中找出对象名"""
        # 处理赋值语句
        if '=' in line:
            # 提取等号左边的变量名
            var_match = re.search(r'(\w+)\s*=', line)
            if var_match and any(method in line for method in [
                    '.move_to(', '.next_to(', '.center()', '.to_edge(',
                    '.shift('
            ]):
                return var_match.group(1)

        # 处理直接调用 object.method() 模式
        # 确保匹配的是真正的对象名，不是方法链的其他部分
        method_match = re.search(
            r'(\w+)\.(?:move_to|next_to|center|to_edge|shift)\(', line)
        if method_match:
            # 这里有个坑，要确保匹配的不是括号前面的什么鬼东西
            obj_name = method_match.group(1)
            if obj_name and obj_name.replace('_',
                                             '').isalnum():  # 允许下划线和字母数字组合
                return obj_name

        return None

    def _extract_move_to_target(self, line):
        """提取move_to的目标"""
        # 提取move_to(target)中的target
        match = re.search(r'\.move_to\(([^)]+)\)', line)
        if match:
            target = match.group(1).strip()

            get_methods = [
                '.get_left()', '.get_right()', '.get_top()', '.get_bottom()'
            ]
            # 处理方法调用，提取基础对象名
            if '.get_center()' in target:
                # 从 'object.get_center()' 提取 'object'
                base_obj = target.replace('.get_center()', '')
                return base_obj if base_obj else target
            elif any(method in target for method in get_methods):  # noqa
                # 处理其他get_方法
                base_match = re.search(r'(\w+)\.get_\w+\(\)', target)
                if base_match:
                    return base_match.group(1)

            # 去除其他装饰符号
            target = target.replace('[', '').replace(']', '').strip()
            return target if target else None
        return None

    def _analyze_spatial_conflicts(self, operations):
        """分析空间关系冲突"""
        conflicts = []

        # 1. 检测相同参考点冲突
        reference_groups = {}
        for op in operations:
            if op['reference'] and op['reference'] != 'relative':
                if op['reference'] not in reference_groups:
                    reference_groups[op['reference']] = []
                reference_groups[op['reference']].append(op)

        for ref, ops_list in reference_groups.items():
            if len(ops_list) > 1:
                conflicts.extend(
                    self._detect_reference_conflicts(ref, ops_list))

        # 2. 检测无间距风险
        for op in operations:
            if op['type'] == 'next_to' and not op['has_spacing']:
                conflicts.append({
                    'type': 'missing_spacing',
                    'severity': 'medium',
                    'operation': op,
                    'description': '缺少buff参数可能导致重叠'
                })

        # 3. 检测间距过小
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
                    f"间距过小({op['spacing_value']})可能导致重叠"
                })

        # 4. 检测对象-几何体重叠风险
        conflicts.extend(self._detect_geometry_conflicts(operations))

        return conflicts

    def _is_text_object(self, obj_name):
        """看看是不是文本对象"""
        return any(keyword in obj_name.lower()
                   for keyword in ['text', 'label', 'title'])

    def _is_geometry_reference(self, ref):
        """判断是否为几何对象引用"""
        if not ref:
            return False
        ref_lower = ref.lower()
        geometry_keywords = [
            'line', 'circle', 'square', 'rectangle', 'triangle', 'polygon',
            'arc', 'ellipse'
        ]
        return any(keyword in ref_lower for keyword in geometry_keywords)

    def _is_label_object(self, obj_name):
        """判断是否为标签对象"""
        return 'label' in obj_name.lower() or obj_name.endswith('_text')

    def _detect_geometry_conflicts(self, operations):
        """检查文本和几何图形是否重叠"""
        conflicts = []

        for op in operations:
            if op['type'] == 'move_to' and op['reference']:
                ref = op['reference']

                # 检测文本直接移动到几何对象
                if self._is_text_object(
                        op['object']) and self._is_geometry_reference(ref):
                    conflicts.append({
                        'type':
                        'text_geometry_overlap',
                        'severity':
                        'high',
                        'operation':
                        op,
                        'description':
                        f"文本对象({op['object']})直接移动到几何对象({ref})位置"
                    })

                # 检测标签直接移动到对象
                elif self._is_label_object(
                        op['object']) and not self._is_safe_reference(ref):
                    conflicts.append({
                        'type':
                        'label_object_overlap',
                        'severity':
                        'medium',
                        'operation':
                        op,
                        'description':
                        f"标签对象({op['object']})可能与目标对象({ref})重叠"
                    })

        return conflicts

    def _is_safe_reference(self, ref):
        """判断是否为安全的引用"""
        safe_patterns = [
            'get_center()', 'ORIGIN', 'UP', 'DOWN', 'LEFT', 'RIGHT'
        ]
        return any(pattern in ref for pattern in safe_patterns)

    def _detect_reference_conflicts(self, reference, operations):
        """检测相同参考点的冲突"""
        conflicts = []

        # 按定位类型分组
        move_to_ops = [op for op in operations if op['type'] == 'move_to']
        center_ops = [op for op in operations if op['type'] == 'center']

        # 多个move_to同一对象
        if len(move_to_ops) > 1:
            conflicts.append({
                'type': 'multiple_move_to',
                'severity': 'high',
                'reference': reference,
                'operations': move_to_ops,
                'description': f'多个对象移动到同一位置({reference})'
            })

        # 多个center
        if len(center_ops) > 1:
            conflicts.append({
                'type': 'multiple_center',
                'severity': 'high',
                'operations': center_ops,
                'description': '多个对象使用center()定位'
            })

        return conflicts

    def _format_conflict_description(self, conflict):
        """格式化冲突描述"""
        op = conflict.get('operation', {})
        line = op.get('line', '?')

        if conflict['type'] == 'missing_spacing':
            return f"第{line}行: {conflict['description']} - {op.get('code', '')}"
        elif conflict['type'] == 'insufficient_spacing':
            return f"第{line}行: {conflict['description']} - {op.get('code', '')}"
        elif conflict['type'] == 'text_geometry_overlap':
            return f"第{line}行: {conflict['description']} - {op.get('code', '')}"
        elif conflict['type'] == 'label_object_overlap':
            return f"第{line}行: {conflict['description']} - {op.get('code', '')}"
        elif conflict['type'] == 'multiple_move_to':
            objects = [op['object'] for op in conflict['operations']]
            return f"多个对象移动到同一位置({conflict['reference']}): {', '.join(objects)}"
        elif conflict['type'] == 'multiple_center':
            objects = [op['object'] for op in conflict['operations']]
            return f"多个对象使用center()定位: {', '.join(objects)}"
        else:
            return conflict.get('description', '未知冲突')