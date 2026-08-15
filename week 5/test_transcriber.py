# test_transcriber.py
from modules.transcriber import transcribe_audio

result = transcribe_audio("Recording.m4a")
print(result)