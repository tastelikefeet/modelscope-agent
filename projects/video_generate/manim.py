


def generate_manim_code(content,
                        content_type,
                        scene_number,
                        context_info=None,
                        surrounding_text='',
                        audio_duration=None,
                        main_theme='',
                        context_segments=None,
                        segment_index=0,
                        total_segments=None,
                        improvement_prompt=None,
                        existing_code=None):
    class_name = f'Scene{scene_number}'

    if not context_info:
        context_info = {
            'emphasis_words': [],
            'explanation_flow': [],
            'timing_cues': [],
            'emotional_tone': 'neutral',
            'complexity_level': 'medium'
        }

    print(f'生成动画代码 - {content_type}: {class_name}')

    # 最优先使用平衡空间约束系统（检测+LLM修复）
    if BALANCED_SPATIAL_AVAILABLE:
        print('使用平衡空间约束系统（增强检测+多轮修复模式）...')

        # 创建平衡空间系统
        balanced_system = BalancedSpatialSystem()

        # 生成平衡的提示词（避免过度工程化）
        balanced_prompt = balanced_system.generate_balanced_prompt(
            content_type=content_type,
            content=content,
            class_name=class_name,
            audio_duration=audio_duration or 8.0)

        # 调用LLM生成初始代码
        try:
            response = modai_model_request(
                balanced_prompt,
                model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
                max_tokens=2000,
                temperature=0.7)

            # 提取代码
            if '```python' in response:
                manim_code = response.split('```python')[1].split('```')[0]
            elif '```' in response:
                manim_code = response.split('```')[1].split('```')[0]
            else:
                manim_code = response

            # 🔍 智能修复策略选择
            initial_analysis = balanced_system.analyze_and_score(manim_code)

            print('   初始代码分析:')
            print(f"   - 布局分数: {initial_analysis['layout_score']}/100")
            print(f"   - 发现问题: {initial_analysis['issue_count']}个")

            # 根据问题严重程度决定修复策略
            if initial_analysis['issue_count'] == 0:
                print('   [成功] 初始代码完美，无需修复')
                final_code = manim_code

            elif initial_analysis['issue_count'] <= 3 and initial_analysis[
                    'layout_score'] >= 80:
                print('问题较少，使用单轮精确修复')

                # 单轮修复
                fix_prompt = balanced_system.generate_fix_prompt(
                    manim_code, initial_analysis['issues'])
                fix_request = f"""
{fix_prompt}

**原始代码**:
```python
{manim_code}
```

请精确修复检测到的问题，确保保持动画效果的丰富性和创意性。
"""

                fix_response = modai_model_request(
                    fix_request,
                    model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
                    max_tokens=2500,
                    temperature=0.3)

                # 提取修复后的代码
                if '```python' in fix_response:
                    fixed_code = fix_response.split('```python')[1].split(
                        '```')[0]
                elif '```' in fix_response:
                    fixed_code = fix_response.split('```')[1].split('```')[0]
                else:
                    fixed_code = fix_response

                # 验证修复效果
                final_analysis = balanced_system.analyze_and_score(fixed_code)

                if final_analysis['layout_score'] >= initial_analysis[
                        'layout_score']:
                    print(
                        f"   [成功] 单轮修复成功: {initial_analysis['layout_score']} → {final_analysis['layout_score']}"
                    )
                    final_code = fixed_code
                else:
                    print('   [警告] 单轮修复效果不佳，使用原始代码')
                    final_code = manim_code

            else:
                print('   🔄 问题较多，启用多轮修复机制')

                # 多轮修复
                fix_result = balanced_system.multi_round_fix(
                    manim_code, max_rounds=3)

                if fix_result['success']:
                    print('   [成功] 多轮修复成功!')
                    print(f"   - 总改进: +{fix_result['total_improvement']}分")
                    print(f"   - 修复轮数: {fix_result['total_rounds']}")
                    final_code = fix_result['final_code']
                else:
                    print('   [警告] 多轮修复未完全成功，但已有改进')
                    print(f"   - 部分改进: +{fix_result['total_improvement']}分")
                    final_code = fix_result['final_code']

            # 最终简单优化
            final_code = balanced_system.optimize_simple_code(final_code)

            return final_code

        except Exception as e:
            print(f'   平衡系统处理失败: {e}')
            # 回退到简单优化
            try:
                basic_prompt = f'创建{content_type}类型的Manim动画，类名{class_name}，内容：{content}'
                response = modai_model_request(basic_prompt, max_tokens=1500)
                return clean_llm_code_output(response)
            except:  # noqa
                return create_simple_manim_scene(content_type, content,
                                                 class_name, '')

    # 优先使用新的优化系统
    if OPTIMIZED_QUALITY_AVAILABLE:
        print('[启动] 使用优化质量控制系统...')

        prompt_system = OptimizedManimPrompts()

        # 如果有现有代码，先进行分析
        if existing_code:
            print('📋 分析现有代码问题...')

        # 构建内容描述
        enhanced_content = content
        if improvement_prompt:
            enhanced_content = f'{content}\n\n改进要求：{improvement_prompt}'

        # 生成优化的提示词
        generation_prompt = prompt_system.generate_creation_prompt(
            enhanced_content, content_type)

        # 调用LLM生成代码
        enhanced_code = modai_model_request(
            prompt=generation_prompt, max_tokens=2048, temperature=0.1)

        if enhanced_code:
            # 使用质量控制器处理生成的代码
            controller = ManimQualityController(max_fix_attempts=2)
            result = controller.process_manim_code(enhanced_code, class_name,
                                                   enhanced_content)

            # 输出处理日志
            for log_entry in result.processing_log:
                print(log_entry)

            if result.success:
                print('[完成] 代码生成和质量控制完成')
                return result.final_code
            else:
                print('[警告] 质量控制部分成功，使用当前最佳版本')
                return result.final_code

    # 回退到原有系统
    elif ENHANCED_PROMPTS_AVAILABLE:
        print('使用增强提示词系统（回退模式）...')
        prompt_system = EnhancedManimPromptSystem()

        # 如果有改进提示，将其添加到内容中
        enhanced_content = content
        if improvement_prompt:
            enhanced_content = f'{content}\n\n{improvement_prompt}'

        # 传递现有代码用于布局分析
        system_prompt, user_prompt = prompt_system.create_enhanced_prompt(
            content=enhanced_content,
            content_type=content_type,
            context_segments=context_segments,
            main_theme=main_theme,
            audio_duration=audio_duration,
            existing_code=existing_code  # 新增：传递现有代码
        )

        enhanced_code = modai_model_request(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
            max_tokens=2000,
            temperature=0.3,
            role='assistant')

        if enhanced_code:
            # 清理LLM输出的格式问题
            enhanced_code = clean_llm_code_output(enhanced_code)

            validation = prompt_system.validate_generated_code(
                enhanced_code, content_type)
            print(f"代码质量得分: {validation['validation_score']}/100")

            if validation['validation_score'] >= 70:
                print('增强提示词生成成功')
                return enhanced_code
            else:
                print('代码质量较低，回退到传统方法')
                for issue in validation['issues']:
                    print(f'- {issue}')

    if context_segments and total_segments and main_theme:
        print('启动智能分析系统...')
        optimization_data = optimize_animation(
            segment_content=content,
            segment_type=content_type,
            main_theme=main_theme,
            context_segments=context_segments,
            total_segments=total_segments,
            segment_index=segment_index)

        if 'error' not in optimization_data:
            optimized_script, enhanced_code = enhanced_script_and_animation_generator(
                original_content=content,
                content_type=content_type,
                main_theme=main_theme,
                optimization_data=optimization_data,
                class_name=class_name)

            if enhanced_code:
                print('智能优化动画生成完成')
                return enhanced_code
            else:
                print('智能优化失败，使用增强版生成器')
        else:
            print(f"智能分析失败，使用增强版生成器: {optimization_data['error']}")

    print('使用增强版动画生成器...')
    total_duration = audio_duration or 8.0
    return enhanced_generate_manim_code(content_type, content, class_name,
                                        surrounding_text, total_duration,
                                        context_info)