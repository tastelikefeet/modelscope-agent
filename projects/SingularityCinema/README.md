# SingularityCinema

一个轻量优秀的短视频生成器

## 安装

1. 克隆代码
```shell
git clone https://github.com/modelscope/ms-agent.git
cd ms-agent
```

2. 安装依赖
```shell
pip isntall .
cd projects/SingularityCinema
pip install -r requirements.txt
```

安装[ffmpeg](https://www.ffmpeg.org/download.html#build-windows).

在执行上面的安装命令之前，请确保你的Python>=3.10。安装Python可以参考[Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)

## 适配性和局限性

SingularityCinema基于大模型生成台本和分镜，并生成短视频。

### 适配性

- 短视频类型：科普类、经济类，尤其包含报表、公式、原理性解释的短视频
- 语言：不限
- 读取外部材料：支持纯文本，不支持多模态
- 二次开发：完整代码均在stepN/agent.py中，没有license限制，可自由二次开发

### 局限性

- LLM测试范围：Claude，其他模型效果未测试
- AIGC模型测试范围：Qwen-Image，其他模型效果未测试

## 运行

1. 准备API Key

### 准备LLM Key

以Claude为例，你需要先申请或购买Claude模型的使用。Claude的Key可以在环境变量中设置：

```shell
OPENAI_API_KEY=xxx-xxx
```

### 准备魔搭文生图Key

目前默认模型是Qwen-Image，魔搭API Key可以在[这里](https://www.modelscope.cn/my/myaccesstoken)申请。之后在环境变量中设置：

```shell
t2i_api_key=ms-xxx-xxx
```

2. 准备你的短视频材料

你可以选择使用一句话生成视频，例如：

```text
生成一个描述GDP经济知识的短视频，约3分钟左右。
```

或者使用自己之前采集的文本材料：

```text
生成一个描述大模型技术的短视频，阅读/home/user/llm.txt获取详细内容
```

3. 运行命令

```shell
ms-agent run --config "ms-agent/projects/SingularityCinema" --query "你的自定义主题，见上面描述" --load_cache true --trust_remote_code true
```

4. 运行持续约20min左右。视频生成在output/final_video.mp4。生成完成后你可以查看这个文件，把不满足要求的地方汇总起来，输入命令行input中，工作流会继续改进。如果达到了要求，输入quit或者exit程序会自动退出。

## 技术原理

本工作流

## 目录说明
- `video_agent.py`：三步逻辑的 Agent 封装
- `workflow.yaml`：三步编排；`workflow_from_assets.yaml`：只合成编排
- `core/workflow.py`：主流程；`core/human_animation_studio.py`：人工工作室
- `core/asset/`：字体与背景音乐
- `output/`：运行产物
- `scripts/compose_from_asset_info.py`：从现有 `asset_info.json` 直接合成的辅助脚本

## 常见问题
- 退出码 1：
	- 检查是否缺少 MODELSCOPE_API_KEY（全自动模式常见）
	- 检查 ffmpeg / manim 是否可执行（PATH）
	- 查看终端最后 80 行日志定位具体异常
- 字体/背景不一致：
	- 背景由 `create_manual_background` 生成，字体/音乐来自 `core/asset/`；确保该目录可读
- TTS/事件循环冲突：
	- 已内置 loop-safe 处理；若仍报错，重试并贴出日志尾部

## 许可证与注意
- 自定义字体文件标注为“商用需授权”，请在合规授权范围内使用
- 背景音乐仅作示例，商业使用请更换或确保版权无虞
