# -*- coding: utf-8 -*-
# uv add openai-whisper
# uv add yt-dlp
# uv add googletrans==4.0.0rc1
import whisper
import sys
import os
import asyncio
import subprocess
from pathlib import Path
from googletrans import Translator
from tqdm import tqdm
import time

from yt_dlp import YoutubeDL

def download_audio(url, output_file="audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_file,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '0',
        }],
        'nocheckcertificate': True,
        'prefer_insecure': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False
def convert_mp4_to_audio(video_path, output_file="audio.mp3"):
    """Convert MP4 to audio using ffmpeg"""
    try:
        cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", "-y", output_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error converting: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

async def translate_text_safe(translator, text, dest='zh-TW', max_retries=3):
    """Safely translate text with retries"""
    for attempt in range(max_retries):
        try:
            result = await translator.translate(text, dest=dest)
            return result.text
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Translation failed after {max_retries} attempts: {e}")
                return text  # Return original text if translation fails
            await asyncio.sleep(1)  # Wait before retry

async def translate_texts_batch(texts, dest='zh-TW', batch_size=10):
    """Translate texts in batches to avoid rate limiting"""
    translator = Translator()
    all_translations = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Translating batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        tasks = [translate_text_safe(translator, text, dest) for text in batch]
        batch_translations = await asyncio.gather(*tasks)
        all_translations.extend(batch_translations)
        
        # Add delay between batches to avoid rate limiting
        if i + batch_size < len(texts):
            await asyncio.sleep(1)
    
    return all_translations

def main():
    # Fix for Windows event loop policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_url_or_file>")
        sys.exit(1)
    
    video = sys.argv[1]
    print(f"Processing: {video}")
    
    # Initialize model
    print("Loading Whisper model...")
    model = whisper.load_model("turbo")
    
    # Handle different input types
    audio_file = None
    
    if video.startswith("http://") or video.startswith("https://"):
        if download_audio(video):
            audio_file = "audio.mp3"
        else:
            print("Failed to download audio")
            sys.exit(1)
    elif video.endswith(".mp4"):
        if convert_mp4_to_audio(video):
            audio_file = "audio.mp3"
        else:
            print("Failed to convert video")
            sys.exit(1)
    elif video.endswith((".mp3", ".wav", ".m4a")):
        audio_file = video
    else:
        print("Please provide a valid video URL, .mp4, .mp3, .wav, or .m4a file.")
        sys.exit(1)
    
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Transcribe audio
    print("Transcribing audio...")
    try:
        result = model.transcribe(audio_file, fp16=False, verbose=False)
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)
    
    # Detect language and decide whether to translate
    detected_lang = result.get("language", "")
    print(f"Detected language: {detected_lang}")
    skip_translation = detected_lang.startswith("zh")
    
    # Process segments
    if isinstance(result, dict) and "segments" in result:
        segments = result["segments"]
        texts = [seg['text'].strip() for seg in segments if seg['text'].strip()]
        
        # Translate if needed
        translations = []
        if not skip_translation and texts:
            print("Translating text to Chinese...")
            try:
                translations = asyncio.run(translate_texts_batch(texts))
            except Exception as e:
                print(f"Translation failed: {e}")
                translations = texts  # Use original text if translation fails
        
        # Write to file
        output_file = "transcription.txt"
        print(f"Writing results to {output_file}...")
        
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(f"Detected Language: {detected_lang}\n")
            f.write("=" * 50 + "\n\n")
            
            for idx, original in enumerate(texts):
                f.write(f"[{idx+1:03d}] {original}\n")
                if not skip_translation and idx < len(translations):
                    f.write(f"[譯] {translations[idx]}\n")
                f.write("\n")
        
        print(f"Transcription complete! Results saved to {output_file}")
        print(f"Processed {len(texts)} segments")
        
    else:
        # Handle full text result
        with open("transcription.txt", "w", encoding='utf-8') as f:
            f.write(f"Detected Language: {detected_lang}\n")
            f.write("=" * 50 + "\n\n")
            f.write(result.get('text', str(result)))
        
        print("Transcription complete! Full text saved to transcription.txt")
    
    # Clean up temporary audio file if it was downloaded/converted
    if audio_file in ["audio.mp3"] and audio_file != video:
        try:
            os.remove(audio_file)
            print("Cleaned up temporary audio file")
        except:
            pass

if __name__ == "__main__":
    main()