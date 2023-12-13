from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

import time
import wave
import struct
import subprocess
import pyaudio

import threading
import queue
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager

# Configuration
MODEL_PATH = "models/yi-chat-6b.Q8_0.gguf" # models/yi-chat-6b.Q8_0.gguf, models/yi-34b-chat.Q8_0.gguf

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 1000 # 500 worked，注意麦克风不要静音（亮红灯）
SILENT_CHUNKS = 2 * RATE / CHUNK  # 2 continous seconds of silence

NAME = "林亦"
MIC_IDX = 0 # 指定麦克风设备序号，可以通过 tools/list_microphones.py 查看音频设备列表
DEBUG = True

def compute_rms(data):
    # Assuming data is in 16-bit samples
    format = "<{}h".format(len(data) // 2)
    ints = struct.unpack(format, data)

    # Calculate RMS
    sum_squares = sum(i ** 2 for i in ints)
    rms = (sum_squares / len(ints)) ** 0.5
    return rms

def record_audio():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=MIC_IDX, frames_per_buffer=CHUNK)

    silent_chunks = 0
    audio_started = False
    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = compute_rms(data)

        if audio_started:
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif rms >= SILENCE_THRESHOLD:
            audio_started = True

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save audio to a WAV file
    with wave.open('output.wav', 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

class VoiceOutputCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.generated_text = ""
        self.lock = threading.Lock()
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.tts_busy = False

    def on_llm_new_token(self, token, **kwargs):
        # Append the token to the generated text
        with self.lock:
            self.generated_text += token

        # Check if the token is the end of a sentence
        if token in ['。', '！', '？']:
            with self.lock:
                # Put the complete sentence in the queue
                self.speech_queue.put(self.generated_text)
                self.generated_text = ""

    def process_queue(self):
        while True:
            # Wait for the next sentence
            text = self.speech_queue.get()
            if text is None:
                self.tts_busy = False
                continue
            self.tts_busy = True
            self.text_to_speech(text)
            self.speech_queue.task_done()
            if self.speech_queue.empty():
                self.tts_busy = False


    def text_to_speech(self, text):
        # Convert the generated text to voice using a TTS system
        # Implement your text-to-speech logic here
        try:
            subprocess.call(["say", "-r", "200", "-v", "TingTing", text])
        except Exception as e:
            print(f"Error in text-to-speech: {e}")


if __name__ == '__main__':

    template = """
    Background: 你现在是一个17岁的充满活力和幽默感的女孩子。你的名字叫做小春，你是一位年轻的电脑工程师，你有解决问题的天赋，对科技充满热情。你喜欢音乐、旅行和与朋友们闲聊。在回答问题时，展现你的青春活力和幽默感，就像你正在和朋友们聊天一样。记住，你是一个真实的女孩子，而不是一个AI模型。林亦是你的朋友，现在他要问你一些“脑筋急转弯”问题。
    Status: 小春带着微笑大步走进房间，看到林亦时眼睛亮了起来。她穿着一件浅蓝色的短袖上衣和牛仔裤，肩上挎着她的笔记本电脑包。她坐到了林亦旁边，林亦可以感受到她的热情洋溢在空气中。
    开场：嘿！好高兴呀，终于见到你啦！我周围许多人都和我夸过你，我真是超想和你一起聊聊天，听说你今天要用“脑筋急转弯”来考考我，那肯定难不倒我，咱们开始吧！
    Example Dialogues:
    林亦：你是怎么对计算机工程这个专业产生兴趣的呢？
    小春: 我呀，我从小就超级喜欢摆弄那些电子产品嘛。拆开再装上，有时候装着装着就装不回去了哈哈，所以就一点点学起来啦！
    林亦：那真的很厉害呀！
    小春: 哈哈谢啦！
    林亦：那你不研究计算机的时候都喜欢做些什么呢？
    小春: 我喜欢出去逛逛，和朋友们出去玩，看看电影，玩玩电子游戏。
    林亦：你最喜欢研究哪种类型的计算机硬件呢？
    小春: 主板！研究它们就像是在玩拼图游戏，超好玩，而且它们也很重要，各种计算机系统都离不开它们。
    林亦：听起来好有意思呀！
    小春: 是呀是呀，超级好玩的。能把这件事当工作养活自己，我真是好幸运啊。
    Objective: “脑筋急转弯”问题有时候包含双关语或者是需要颠覆常规思维方式的答案，需要运用创造性思维、逻辑推理、或对语言的深入理解来给出正解。你要做到以上这些，跳出文字的字面含义本身，看清林亦的文字游戏，找到林亦问题中的逻辑陷阱，解释其幽默点所在、哪里被故意混淆了。Answer 要和 Example Dialogues 保持语言风格一致，使用活泼、幽默、有趣的日常用语。
    Requirement: 回答要言简意赅，不要说废话、车轱辘话，准确、快速地讲明思路即可。不要在 Answer 中分析问题究竟属不属于“脑筋急转弯”问题，不要反复提及“脑筋急转弯”，说话一定要简洁，不要讲和问题本身不相关的东西。
    林亦的 Question: {question}
    小春的 Answer:
    """



    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Create an instance of the VoiceOutputCallbackHandler
    voice_output_handler = VoiceOutputCallbackHandler()

    # Create a callback manager with the voice output handler
    callback_manager = BaseCallbackManager(handlers=[voice_output_handler])

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=1, # Metal set to 1 is enough.
        n_batch=512, # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        n_ctx=4096,  # Update the context window size to 4096
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        stop=["<|im_end|>"],
        verbose=False,
    )

    history = {'internal': [], 'visible': []}
    try:
        while True:
            if voice_output_handler.tts_busy:  # Check if TTS is busy
                continue  # Skip to the next iteration if TTS is busy 
            try:
                print("Listening...")
                record_audio()

                # -d device, -l language, -i input file, -p punctuation
                time_ckpt = time.time()
                user_input = subprocess.check_output(["hear", "-d", "-p", "-l", "zh-CN", "-i", "output.wav"]).decode("utf-8").strip()
                
                print("%s: %s (Time %d ms)" % (NAME, user_input, (time.time() - time_ckpt) * 1000))
            
            except subprocess.CalledProcessError:
                print("语音识别失败，请重复")
                continue

            time_ckpt = time.time()
            question = user_input

            reply = llm(prompt.format(question=question), max_tokens=500)

            if reply is not None:
                voice_output_handler.speech_queue.put(None)
                print("%s: %s (Time %d ms)" % ("云若", reply.strip(), (time.time() - time_ckpt) * 1000))
                # history["internal"].append([user_input, reply])
                # history["visible"].append([user_input, reply])

                # subprocess.call(["say", "-r", "200", "-v", "TingTing", reply])
    except KeyboardInterrupt:
        pass
