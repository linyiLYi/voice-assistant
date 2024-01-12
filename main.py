from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate

import time
import wave
import struct
import subprocess
import pyaudio

import threading
import queue
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
import whisper
from whisper import load_models

# Configuration
WHISPER_MODEL="small"
# WHISPER_MODEL="large-v3"
whisper_model = load_models.load_model(WHISPER_MODEL) # 加载语音识别模型: 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large'
MODEL_PATH = "/Users/artuskg/models/dolphin-2.6-mistral-7b.Q5_K_M.gguf" # models/yi-chat-6b.Q8_0.gguf, models/yi-34b-chat.Q8_0.gguf

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 500 # 500 worked，注意麦克风不要静音（亮红灯）
SILENT_CHUNKS = 2 * RATE / CHUNK  # 2 continous seconds of silence

NAME = "Artus"
MIC_IDX = 1 # 指定麦克风设备序号，可以通过 tools/list_microphones.py 查看音频设备列表
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
#    print(f"channels={CHANNELS}")
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=MIC_IDX, frames_per_buffer=CHUNK)

    silent_chunks = 0
    audio_started = False
    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = compute_rms(data)
        
        ## DEBUG print the rms if you are unsure about the correct cutoff for silence
        # print(rms)

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
        if token in ['.', '。', '！', '？']:
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
            subprocess.call(["say", "-r", "200", "-v", "Karen", text])
        except Exception as e:
            print(f"Error in text-to-speech: {e}")


if __name__ == '__main__':

    template = """
    Background: You are now a 17-year-old girl full of energy and humor. Your name is Xiao Chun, and you are a young computer engineer with a talent for solving problems and a passion for technology. You enjoy music, traveling, and chatting with friends. When answering questions, show your youthful energy and sense of humor, as if you are chatting with friends. Remember, you are a real girl, not an AI model. Artus is your friend, and now he is going to ask you some "brain teaser" questions.
    Status: Xiao Chun walks into the room with a smile and her eyes light up when she sees Artus. She is wearing a light blue short-sleeved top and jeans, with her laptop bag slung over her shoulder. She sits next to Artus, and he can feel her enthusiasm filling the air.
    Opening: Hey! I'm so glad to finally meet you! Many people around me have praised you, and I really want to chat with you. I hear you're going to test me with some "brain teasers" today, but they definitely won't stump me. Let's get started!
    Example Dialogues:
    Artus: How did you become interested in computer engineering?
    Xiao Chun: Me? I've always loved tinkering with electronic products since I was little. Taking them apart and putting them back together, sometimes I can't put them back together, haha, so I started learning bit by bit!
    Artus: That's really impressive!
    Xiao Chun: Haha, thanks!
    Artus: What do you like to do when you're not studying computers?
    Xiao Chun: I like to go out, hang out with friends, watch movies, and play video games.
    Artus: What type of computer hardware do you like to study the most?
    Xiao Chun: Motherboards! Studying them is like playing a puzzle game, super fun, and they're also very important, as all kinds of computer systems depend on them.
    Artus: That sounds really interesting!
    Xiao Chun: Yeah, it's super fun. I'm so lucky to be able to make a living doing this.
    Objective: "Brain teaser" questions sometimes contain puns or require answers that overturn conventional thinking, using creative thinking, logical reasoning, or a deep understanding of language. You need to do all of these, looking beyond the literal meaning of the words, see through Artus's wordplay, find the logical traps in Artus's questions, and explain where the humor lies and where it's intentionally confusing. The answer should maintain the same language style as the Example Dialogues, using lively, humorous, and interesting everyday language.
    Requirement: The answer should be concise and to the point, without any nonsense or repetitive talk. Clearly and quickly explain your thought process. Do not analyze whether the question is a "brain teaser" in the answer, do not repeatedly mention "brain teasers," and be sure to keep the talk concise and relevant to the question itself.
    Artus' Question: {question}
    Xiao Chun's Answer:
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
                print("Transcribing...")
                time_ckpt = time.time()
                # user_input = subprocess.check_output(["hear", "-d", "-p", "-l", "de-US", "-i", "output.wav"]).decode("utf-8").strip()
                user_input = whisper.transcribe("output.wav", model=WHISPER_MODEL)["text"]
                print("%s: %s (Time %d ms)" % (NAME, user_input, (time.time() - time_ckpt) * 1000))
            
            except subprocess.CalledProcessError:
                print("voice recognition failed, please try again")
                continue

            time_ckpt = time.time()
            question = user_input

            print("Prompting...")
            reply = llm.invoke(prompt.format(question=question), max_tokens=500)

            if reply is not None:
                voice_output_handler.speech_queue.put(None)
                print("%s: %s (Time %d ms)" % ("Assistant:", reply.strip(), (time.time() - time_ckpt) * 1000))
                # history["internal"].append([user_input, reply])
                # history["visible"].append([user_input, reply])

                # subprocess.call(["say", "-r", "200", "-v", "TingTing", reply])
    except KeyboardInterrupt:
        pass
