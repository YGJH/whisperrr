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
    # 更穩定的 yt-dlp 選項
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': 'audio.%(ext)s',  # 確保有 ext
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '0',
        }],
        'prefer_ffmpeg': True,
        'nocheckcertificate': True,
        'no_warnings': True,
        'quiet': False,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/115.0 Safari/537.36'
        }
    }
    # 如果有 cookie 檔就自動使用
    cookie_path = 'www.youtube.com_cookies.txt'
    if os.path.exists(cookie_path):
        ydl_opts['cookiefile'] = cookie_path

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # 下載後 ffmpeg 會轉成 audio.mp3（postprocessor）
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        try:
            print("Trying fallback with yt-dlp CLI...")
            # CLI fallback 加上常見可解決 SABR/格式問題的參數
            cmd = (
                'uv run yt-dlp --no-playlist --no-check-certificate '
                '--allow-unplayable-formats --output "audio.%(ext)s" '
                '--extract-audio --audio-format mp3 --audio-quality 320K '
                f'{url}'
            )
            subprocess.run(cmd, shell=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Fallback download failed: {e}")
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

def gen_summary(model='openai'):
    # Try OpenAI first, then Gemini -> Ollama -> CPU fallback
    try:
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY2')
        if not OPENAI_API_KEY:
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


        if OPENAI_API_KEY and model=='openai':
            try:
                # Use the modern openai client (openai>=1.0.0)
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)

                with open('transcribe.txt', 'r', encoding='utf-8') as f:
                    text = f.read()

                system_prompt = "你是一個會將轉錄文字詳細總結成中文的助理，請用繁體中文回覆。"
                user_prompt = "請幫我總結以下影片內容，越詳細越好，並且用中文+markdown格式回覆我。\n\n" + text

                # Try progressive model choices
                openai_models = ["gpt-4o" , "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
                resp = None
                for m in openai_models:
                    try:
                        print(f"Trying OpenAI model: {m}")
                        resp = client.chat.completions.create(
                            model=m,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.2,
                            max_tokens=15000
                        )
                        break
                    except Exception as inner_e:
                        print(f"Model {m} failed: {inner_e}")
                        resp = None

                if resp is not None:
                    # New client returns objects with attribute access
                    try:
                        summary_text = resp.choices[0].message.content.strip()
                    except Exception:
                        # Fallback if structure differs
                        summary_text = str(resp)

                    with open('summary.md', 'w', encoding='utf-8') as f:
                        f.write(summary_text)
                    print("Summary generated via OpenAI and saved to summary.md")
                    return True
                else:
                    print("All configured OpenAI models failed, falling back")
            except Exception as e:
                print(f"OpenAI API failed: {e}")
        else:
            print("OPENAI_API_KEY not set, skipping OpenAI")
    except Exception as e:
        print(f"OpenAI block error: {e}")

    # Fallback 1: Gemini
    try:
        print("Trying Gemini fallback...")
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        with open('transcribe.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        prompt = "請幫我總結以下影片內容，越詳細愈好，並且用中文回覆我。\n\n" + text
        response = model.generate_content(prompt)

        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                summary_text = candidate.content.parts[0].text
                with open('summary.md', 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                print("Summary generated with Gemini and saved to summary.md")
                return True
            else:
                raise Exception("No content parts in Gemini response")
        else:
            raise Exception("No candidates in Gemini response")
    except Exception as e:
        print(f"Gemini API failed: {e}")

    # Fallback 2: Ollama (CUDA) or CPU simple summary
    try:
        print("Trying Ollama fallback...")
        import torch
        if torch.cuda.is_available():
            try:
                import ollama
                client = ollama.Client()
                with open('transcribe.txt', 'r', encoding='utf-8') as f:
                    text = f.read()
                prompt = "請幫我總結以下影片內容，越詳細愈好，並且用中文回覆我。\n\n" + text
                response = client.chat(
                    model="qwen3:8b",
                    messages=[{"role": "user", "content": prompt}]
                )
                with open('summary.md', 'w', encoding='utf-8') as f:
                    f.write(response['message']['content'])
                print("Summary generated with Ollama and saved to summary.md")
                return True
            except Exception as e:
                print(f"Ollama (CUDA) attempt failed: {e}")
        else:
            print("CUDA not available for Ollama, will use CPU fallback")
    except Exception as e:
        print(f"Ollama check failed: {e}")

    # Final fallback: simple CPU-based summary from transcription
    try:
        print("Creating basic summary from transcription (CPU fallback)...")
        with open('transcribe.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        summary_lines = []
        summary_lines.append("=== 影片內容摘要 ===")
        summary_lines.append("(由於 API 不可用，這是基於轉錄文字的簡單摘要)")
        summary_lines.append("")

        # intro: first 5 non-empty lines
        for line in lines[:5]:
            summary_lines.append(f"• {line}")

        summary_lines.append("")
        summary_lines.append("=== 主要內容摘錄 ===")

        # middle samples
        if lines:
            mid = len(lines) // 2
            for line in lines[mid: mid + 5]:
                summary_lines.append(f"• {line}")

        # conclusion: last 3 lines
        if len(lines) > 3:
            summary_lines.append("")
            summary_lines.append("=== 結論 ===")
            for line in lines[-3:]:
                summary_lines.append(f"• {line}")

        summary_text = '\n'.join(summary_lines)
        with open('summary.md', 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print("Basic summary generated and saved to summary.md")
        return True
    except Exception as e:
        print(f"All summary generation methods failed: {e}")
        return False

def wait_for_pcloud_mount():
    """等待 pCloud 挂载完成"""
    mount_path = '/home/charles/pCloudDrive'
    target_dir = '/home/charles/pCloudDrive/documents/obsidian/'
    
    print("Starting pCloud...")
    # 使用 Popen 启动 pCloud，但不等待它完成
    process = subprocess.Popen(['/home/charles/Downloads/pcloud'])
    
    # 等待挂载点出现
    max_attempts = 30  # 最多等待60秒 (30 * 2秒)
    for attempt in range(max_attempts):
        if os.path.ismount(mount_path) or os.path.exists(target_dir):
            print(f"pCloud mounted successfully after {attempt * 2} seconds")
            return True
        
        print(f"Waiting for pCloud mount... ({attempt * 2}s)")
        time.sleep(2)
    
    print("Timeout: pCloud failed to mount within expected time")
    return False


def send_telegram(msg: str):
    try:
        BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN_WHISPER']
    except KeyError:
        print("❌ TELEGRAM_BOT_TOKEN_WHISPER not set")
        return

    CHAT_ID   = "6166024220"
    import requests
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg}
    r = requests.post(url, data=payload)
    if not r.ok:
        print("❌ 發送失敗：", r.text)


def main():
    # Fix for Windows event loop policy
    import argparse
    summary = False
    parser = argparse.ArgumentParser(description="Transcribe and translate audio from video files.")
    parser.add_argument('--summary', action='store_true', help="Generate summary of the transcription")
    parser.add_argument('--video', nargs='?', help="Video URL or file path to transcribe")
    parser.add_argument('--copy', action='store_true', help='copy summary to pcloud')
    parser.add_argument('--model', default='openai', help='use specific model for transcription')
    
    args = parser.parse_args()
    model   = args.model
    video = args.video
    summary = args.summary
    copy = args.copy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    print(f"copy mode: {copy}")
    print(f"Summary mode: {summary}")
    print(f"Processing: {video}")
    
    # Initialize model
    print("Loading Whisper model...")
    model = whisper.load_model("turbo")
    print('loading whisper model done')
    # Handle different input types
    audio_file = None
    
    if video.startswith("http://") or video.startswith("https://"):
        # https://www.youtube.com/watch?v=Vz40rDiWnN8&t=150s
        video = video.replace('\\' , '').strip()
        print(video)
        if '?v' in video:
            video = "https://www.youtube.com/watch?v" + video.split('?v')[-1]
        if '&' in video:
            video = video.split('&')[0]
        print(f"Downloading audio from URL: {video}")
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
    print(f"Transcribing audio... {audio_file}")
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
    
    with open('chinese.txt', 'w', encoding='utf-8') as f:
        if skip_translation:
            for idx, original in enumerate(texts):
                f.write(f"{original}\n")
        else:
            for idx, original in enumerate(texts):
                f.write(f"{translations[idx]}\n")

    with open('transcribe.txt', 'w', encoding='utf-8') as f:
        for idx, original in enumerate(texts):
            f.write(f"{original}\n")

    
    if summary:
        print("Generating summary...")
        gen_summary(model=model)
    if copy:
        import platform
        import shutil
        if platform.system() == 'Linux':
            try:
                shutil.copy('summary.md', '/home/charles/pCloudDrive/documents/obsidian/summary.md')
                print("File copied successfully")
            except Exception as e:
                print(f"Initial copy failed: {e}")
                
                if wait_for_pcloud_mount():
                    try:
                        shutil.copy('summary.md', '/home/charles/pCloudDrive/documents/obsidian/summary.md')
                        print("File copied successfully after pCloud mount")
                    except Exception as e2:
                        print(f"Copy failed even after mounting: {e2}")
                else:
                    print("Could not mount pCloud, copy operation failed")
                    
        elif platform.system() == 'Windows':
            shutil.copy('summary.md', 'P:\\documents\\obsidian\\summary.md')
        else:
            print("We don't have this man here")


        
    # Clean up temporary audio file if it was downloaded/converted
    # if audio_file in ["audio.mp3"] and audio_file != video:
    #     try:
    #         os.remove(audio_file)
    #         print("Cleaned up temporary audio file")
    #     except:
    #         pass


    send_telegram('finish whisper')


if __name__ == "__main__":
    main()
