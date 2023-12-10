# 语音助手

一个简单的 Python 脚本，可以通过语音与本地大语言模型进行对话。

### macOS 安装指南

以下为 macOS 的安装过程，Windows 与 Linux 可以使用 speech_recognition 与 pyttsx3 来替代下文中的 macOS 的 hear 与 say 指令。

#### 创建环境
```
conda create -n VoiceAI python=3.11
conda activate VoiceAI
pip install langchain
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python

# 安装音频处理工具
brew install portaudio
pip install pyaudio
```

#### hear 语音识别模块
使用 [hear](https://github.com/sveinbjornt/hear) 指令可以直接调用 macOS 的语音识别模块。注意要开启电脑设置里的键盘听写选项：设置 -> 键盘 -> 听写（开启开关）。

#### 模型文件
模型文件存放于  `models/` 文件夹下，在脚本中通过变量 `MODEL_PATH` 指定。
推荐下载 TheBloke 的 gguf 格式模型：
- [Yi-34B-Chat-GGUF](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q8_0.gguf)
- [Yi-6B-Chat-GGUF，适用小显存平台，尚未测试](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/tree/main)
