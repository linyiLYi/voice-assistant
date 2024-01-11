# Voice-Assistant

A simple Python script that allows for voice interaction with a local large language model. In this project, the whisper implementation comes from the official mlx example library. The large language model is the Yi model from Zero Yiwu, where the Yi-34B-Chat has stronger capabilities and is recommended for use when memory space allows.

### macOS Installation Guide

The following is the installation process for macOS. Windows and Linux can use speech_recognition and pyttsx3 to replace the hear/whisper and say commands in macOS.

### Creating an Environment
```
conda create -n VoiceAI python=3.11
conda activate VoiceAI
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

# Install audio processing tools
```
brew install portaudio
pip install pyaudio
```

#### Installing the hear Voice Recognition Module

Download the installation package from the open-source [hear](https://github.com/sveinbjornt/hear) project, unzip the [folder](https://sveinbjorn.org/files/software/hear.zip), and run sudo bash install.sh (administrator rights required). After installation, you can directly use the voice recognition function of macOS through console commands. Note that you need to enable the dictation option in computer settings: Settings -> Keyboard -> Dictation (turn on the switch). The first time you use it on macOS, you also need to allow the hear module to run in "Settings -> Privacy & Security".

#### Model Files

Model files are stored in the models/ folder and specified in the script through the MODEL_PATH variable. It is recommended to download TheBloke and XeIaso's gguf format models, among which the 6B model occupies less memory.
