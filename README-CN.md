# 极简语音助手脚本

简体中文 | [English](README.md)

这是一个简单的 Python 脚本项目，可以通过语音与本地大语言模型进行对话。

本项目的语音识别部分来自苹果 MLX [官方示例库](https://github.com/ml-explore/mlx-examples/tree/main/whisper)，使用[零一万物](https://www.lingyiwanwu.com)的 Yi 模型生成文字回复，详见[鸣谢](## 鸣谢)部分。

### 文件结构

```bash
├───main.py
├───models
├───prompts
├───recordings
├───tools
│   └───list_microphones.py
├───whisper
```

本项目为单脚本项目，主要程序逻辑全部在 `main.py` 中。`models/` 文件夹存放模型文件。`prompts/` 存放提示词。`recordings/` 存放临时录音。`tools/list_microphones.py` 是一个用来查看麦克风列表的简单脚本，用来在 `main.py` 中指定麦克风序号。`whisper/` 来自苹果 MLX 项目[官方示例](https://github.com/ml-explore/mlx-examples/tree/main/whisper)，用于识别用户输入语音。

## 运行指南

本项目基于 Python 编程语言，程序运行使用的 Python 版本为 3.11.5，建议使用 [Anaconda](https://www.anaconda.com) 配置 Python 环境。以下配置过程已在 macOS 系统上测试通过，Windows 与 Linux 可以使用 speech_recognition 与 pyttsx3 来替代下文中的 whisper 与 say 指令。以下为控制台/终端（Console/Terminal/Shell）指令。

### 环境配置

```
conda create -n VoiceAI python=3.11
conda activate VoiceAI
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# 安装音频处理工具
brew install portaudio
pip install pyaudio
```

### 模型文件
模型文件存放于  `models/` 文件夹下，在脚本中通过变量 `MODEL_PATH` 指定。
推荐下载 TheBloke 与 XeIaso 的 gguf 格式模型，其中 6B 模型显存占用更小：
- [TheBloke/Yi-34B-Chat-GGUF](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q8_0.gguf)
- [XeIaso/Yi-6B-Chat-GGUF](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/blob/main/yi-chat-6b.Q8_0.gguf)
语音识别模型默认存放在 `models/whisper-large-v3/`，在脚本中通过 `WHISP_PATH` 指定。可以直接下载 mlx-community 转换好的[版本](https://huggingface.co/mlx-community/whisper-large-v3-mlx)。

## 鸣谢

本项目的语音识别部分基于 OpenAI 的 whisper 模型，其实现来自苹果 MLX [官方示例](https://github.com/ml-explore/mlx-examples/tree/main/whisper)，本项目中使用的是来自 2024 年 1 月的版本 #80d1867，未来各位本地使用时可以按需抓取新版本。

本项目的回复内容由[零一万物](https://www.lingyiwanwu.com)的大语言模型 Yi 模型生成，其中 Yi-34B-Chat 的能力更强，使用 [TheBloke 制作的 8-bit 量化版本](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF)显存占用为 39.04 GB，硬件条件允许的情况下推荐使用。该模型在本地运行基于 [LangChain 语言框架](https://www.langchain.com)和 [Georgi Gerganov 团队的 llama.cpp 项目](https://github.com/ggerganov/llama.cpp)。

感谢各位程序工作者对开源社区的贡献！