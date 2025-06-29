# -*- coding: utf-8 -*-
# uv add openai-whisper
# uv add yt-dlp
import whisper
import sys
import os
# using async batch translation via googletrans
from googletrans import Translator
import asyncio

# initialize model and translator
model = whisper.load_model("turbo")
translate = Translator()

video = sys.argv[1]  # Get video URL or file path from command line argument
# If input is an HTTPS URL, download audio using yt-dlp
if video.startswith("http://") or video.startswith("https://"):
    print("Downloading audio from URL...")
    if os.path.exists("audio.mp3"):
        os.remove("audio.mp3")  # Remove existing audio file if it exists
    # Download and extract audio to audio.mp3
    os.system(f"uv run yt-dlp --extract-audio --audio-format mp3 -o audio.mp3 {video}")
    video = "audio.mp3"
elif video.endswith(".mp4"):
    ffmpeg_command = f"ffmpeg -i {video} -vn -acodec copy audio.mp3"
    os.system(ffmpeg_command)
    video = "audio.mp3"  # Use the extracted audio file
elif not video:
    print("Please provide a valid .mp4 video or mp3 audio file.")
    sys.exit(1)

audio = whisper.load_audio(video)
audio = whisper.pad_or_trim(audio)
# Use tqdm to display a progress bar during transcription

result = model.transcribe(video, fp16=False, prompt="", task="transcribe", verbose=False)
# Detect language and decide whether to translate
detected_lang = result.get("language", "")
skip_translation = detected_lang.startswith("zh")  # skip if Chinese detected

async def translate_texts(texts, dest='zh-TW'):
    tasks = [translate.translate(text, dest=dest) for text in texts]
    results = await asyncio.gather(*tasks)
    return [res.text for res in results]

# print the recognized text segment by segment
if isinstance(result, dict) and "segments" in result:
    segments = result["segments"]
    texts = [seg['text'] for seg in segments]
    # batch translate if needed
    translations = []
    if not skip_translation:
        translations = asyncio.run(translate_texts(texts))
    with open("transcription.txt", "w", encoding='UTF-8') as f:
        for idx, original in enumerate(texts):
            f.write(f"{original}\n")
            if not skip_translation:
                f.write(f"{translations[idx]}\n")


else:
    # Handle DecodingResult format
    print("Full transcription:")
    print(result.text)