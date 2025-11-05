import json
import os
import time
from io import BytesIO
from typing import List

import requests
from PIL import Image
from omegaconf import DictConfig

from ms_agent.agent import CodeAgent
from ms_agent.llm import Message


class GenerateImages(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')

    async def run(self, inputs, **kwargs) -> List[Message]:
        messages, context = inputs
        segments = context['segments']
        illustration_prompts = context['illustration_prompts']
        text_segments = [
            seg for seg in segments if seg.get('type') == 'text'
        ]
        images_dir = os.path.join(self.work_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        image_files = []
        for prompt in illustration_prompts:
            output_file = self.generate_images(prompt, output_dir=images_dir)
            image_files.append(output_file)

        # Prepare illustration paths list aligned to segments
        illustration_paths: List[str] = []
        # move to images folder for consistent paths
        for i, img_path in enumerate(image_paths):
            if os.path.exists(img_path):
                new_path = os.path.join(
                    images_dir, f'illustration_{i + 1}.png'
                    if img_path.lower().endswith('.png') else
                    f'illustration_{i + 1}.jpg')
                try:
                    os.replace(img_path, new_path)
                except Exception:
                    try:
                        import shutil
                        shutil.move(img_path, new_path)
                    except Exception:
                        new_path = img_path
                image_paths[i] = new_path
        json.dump(
            image_paths,
            open(image_paths_path, 'w', encoding='utf-8'),
            ensure_ascii=False,
            indent=2)

        fg_out_dir = os.path.join(images_dir, 'output_black_only')
        os.makedirs(fg_out_dir, exist_ok=True)
        # process background removal if needed
        if len([
            f for f in os.listdir(fg_out_dir)
            if f.lower().endswith('.png')
        ]) < len(image_paths):
            self.keep_only_black_for_folder(
                images_dir, fg_out_dir)

        # map illustrations back to segment indices
        text_idx = 0
        for idx, seg in enumerate(segments):
            if seg.get('type') == 'text':
                if text_idx < len(image_paths):
                    transparent_path = os.path.join(
                        fg_out_dir, f'illustration_{text_idx + 1}.png')
                    if os.path.exists(transparent_path):
                        illustration_paths[idx] = transparent_path
                    else:
                        illustration_paths[idx] = image_paths[text_idx]
                    text_idx += 1
                else:
                    illustration_paths[idx] = None
            else:
                illustration_paths[idx] = None
        else:
            illustration_paths = [None] * len(segments)

        # Attach illustration paths to asset_paths
        asset_paths['illustration_paths'] = illustration_paths

    def generate_images(self,
                        prompt,
                        img_path,
                        negative_prompt=None):
        base_url = self.config.text2image.base_url
        api_key = self.config.text2image.api_key
        model_id = self.config.text2image.model
        assert api_key is not None
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

        def create_placeholder(path):
            img = Image.new('RGB', (512, 512), (255, 255, 255))
            img.save(path)

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

    def keep_only_black_for_folder(self, input_dir, output_dir, threshold=80):
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
