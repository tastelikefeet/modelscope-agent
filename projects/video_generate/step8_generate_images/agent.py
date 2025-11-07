import asyncio
import os
from io import BytesIO

import aiohttp
import numpy as np
from ms_agent.agent import CodeAgent
from ms_agent.utils import get_logger
from omegaconf import DictConfig
from PIL import Image

logger = get_logger()


class GenerateImages(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')

    async def execute_code(self, inputs, **kwargs):
        messages, context = inputs
        illustration_prompts = context['illustration_prompts']
        context['illustration_paths'] = []
        images_dir = os.path.join(self.work_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        for i, prompt in enumerate(illustration_prompts):
            img_path = os.path.join(images_dir,
                                    f'illustration_{i + 1}_origin.png')
            black_img_path = os.path.join(images_dir,
                                          f'illustration_{i + 1}.png')
            await self.generate_images(prompt, img_path)
            self.keep_only_black_for_folder(img_path, black_img_path)
            context['illustration_paths'].append(black_img_path)

    async def generate_images(self, prompt, img_path, negative_prompt=None):
        base_url = self.config.text2image.base_url
        api_key = self.config.text2image.api_key
        model_id = self.config.text2image.model
        assert api_key is not None

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f'{base_url}v1/images/generations',
                    headers={
                        **headers, 'X-ModelScope-Async-Mode': 'true'
                    },
                    json={
                        'model': model_id,
                        'prompt': prompt,
                        'negative_prompt': negative_prompt or ''
                    }) as resp:
                resp.raise_for_status()
                task_id = (await resp.json())['task_id']

            max_wait_time = 600  # 10 min
            poll_interval = 2
            max_poll_interval = 10
            elapsed_time = 0

            while elapsed_time < max_wait_time:
                await asyncio.sleep(poll_interval)
                elapsed_time += poll_interval

                async with session.get(
                        f'{base_url}v1/tasks/{task_id}',
                        headers={
                            **headers, 'X-ModelScope-Task-Type':
                            'image_generation'
                        }) as result:
                    result.raise_for_status()
                    data = await result.json()

                    if data['task_status'] == 'SUCCEED':
                        img_url = data['output_images'][0]
                        async with session.get(img_url) as img_resp:
                            img_content = await img_resp.read()
                            image = Image.open(BytesIO(img_content))
                            image.save(img_path)
                        return img_path

                    elif data['task_status'] == 'FAILED':
                        raise RuntimeError(
                            f'Generate image failed because of error: {data}')

                poll_interval = min(poll_interval * 1.5, max_poll_interval)

    @staticmethod
    def keep_only_black_for_folder(input_image, output_image, threshold=80):
        img = Image.open(input_image).convert('RGBA')
        arr = np.array(img)

        logger.info(f'Process image: {input_image}')
        logger.info(f'  Size: {img.size}')
        logger.info(f'  Mode: {img.mode}')
        logger.info(
            f'  Color range: R[{arr[..., 0].min()}-{arr[..., 0].max()}], G[{arr[..., 1].min()}-{arr[..., 1].max()}]'
            f', B[{arr[..., 2].min()}-{arr[..., 2].max()}]')

        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        mask = gray < threshold

        transparent_pixels = np.sum(mask)
        total_pixels = mask.size
        transparency_ratio = transparent_pixels / total_pixels
        logger.info(
            f'Black pixels detected: {transparent_pixels}/{total_pixels} ({transparency_ratio:.1%})'
        )

        arr[..., 3] = np.where(mask, 255, 0)

        img2 = Image.fromarray(arr, 'RGBA')
        img2.save(output_image, 'PNG')
        output_img = Image.open(output_image)
        output_arr = np.array(output_img)
        if output_img.mode == 'RGBA':
            alpha_channel = output_arr[..., 3]
            unique_alpha = np.unique(alpha_channel)
            logger.info(f'Transparent value: {unique_alpha}')
        else:
            logger.warn(f'Output image is not RGBA mode: {output_img.mode}')
