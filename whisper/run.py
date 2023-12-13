import whisper

text = whisper.transcribe("/Users/crackerben/me/records/aia-trust.m4a")["text"]
print(text)
with open("result.txt", "w") as file:
    file.write(text)

