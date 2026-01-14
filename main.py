# -*- coding: utf-8 -*-
"""
Whisper Transcription Service
Transcribes audio/video from YouTube or local files using OpenAI Whisper,
translates to Chinese, and generates AI-powered summaries.
"""

import sys
import os
import asyncio
import subprocess
import time
import json
from pathlib import Path

import whisper
from googletrans import Translator
from yt_dlp import YoutubeDL


# ============================================================================
# DOWNLOAD HELPERS
# ============================================================================


def _guess_latest_downloaded_media_file(since_seconds: int = 15 * 60):
    """Best-effort guess of the newest downloaded media file in the cwd."""
    exts = {'.mp4', '.mkv', '.webm', '.m4a', '.opus', '.mp3', '.wav'}
    now = time.time()
    newest_path = None
    newest_mtime = 0.0

    for p in Path('.').iterdir():
        try:
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            st = p.stat()
            if now - st.st_mtime > since_seconds:
                continue
            if st.st_mtime > newest_mtime:
                newest_mtime = st.st_mtime
                newest_path = str(p)
        except Exception:
            continue

    return newest_path

def _get_ydl_options(cookie_path):
    """Get yt-dlp options with Android client configuration."""
    opts = {
        # Request the best available video + best audio, fall back to best
        # This lets yt-dlp automatically pick the highest resolution available.
        'format': 'bestvideo+bestaudio/best',
        # Save as "<video title>.ext" (merged container will usually be .mp4)
        'outtmpl': '%(title).200B.%(ext)s',
        'noplaylist': True,
        'merge_output_format': 'mp4',
        'windowsfilenames': True,
        'prefer_ffmpeg': True,
        'nocheckcertificate': True,
        'no_warnings': False,
        'quiet': False,
        'ignoreerrors': False,
        'geo_bypass': True,
        # 'extractor_args': {'youtube': {'player_client': ['ios', 'web', 'android']}},
    }
    
    if os.path.exists(cookie_path):
        opts['cookiefile'] = cookie_path
    
    return opts




def _try_python_download(url, ydl_opts):
    """Attempt download using yt-dlp Python API.

    Returns:
        str | None: Downloaded filepath if successful, otherwise None
    """
    try:
        print("Attempting download with yt-dlp...")
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            downloaded_path = None
            if isinstance(info, dict):
                # Prefer post-processed/moved filepath if present
                req = info.get('requested_downloads')
                if isinstance(req, list) and req:
                    for item in req:
                        if isinstance(item, dict) and item.get('filepath'):
                            downloaded_path = item['filepath']
                            break
                if not downloaded_path:
                    downloaded_path = info.get('filepath') or info.get('_filename')

            if not downloaded_path:
                try:
                    downloaded_path = ydl.prepare_filename(info)
                except Exception:
                    downloaded_path = None

        # If merge_output_format is set, the final file often ends with that extension.
        merge_ext = ydl_opts.get('merge_output_format')
        if downloaded_path and merge_ext and not os.path.exists(downloaded_path):
            base, _ext = os.path.splitext(downloaded_path)
            candidate = base + '.' + str(merge_ext)
            if os.path.exists(candidate):
                downloaded_path = candidate

        if downloaded_path and os.path.exists(downloaded_path):
            print(f"Download successful! File: {downloaded_path}")
            return downloaded_path

        guessed = _guess_latest_downloaded_media_file()
        if guessed and os.path.exists(guessed):
            print(f"Download successful! Guessed file: {guessed}")
            return guessed

        print("Download finished but could not determine output filename")
        return None
    except Exception as e:
        print(f"Android client failed: {e}")
        return None


def _try_cli_download(url, cookie_path):
    """Attempt download using yt-dlp CLI.

    Returns:
        str | None: Downloaded filepath if successful, otherwise None
    """
    try:
        print("Trying CLI fallback...")
        cmd = [
            'yt-dlp',
            '--no-playlist',
            '--format', 'bestvideo+bestaudio/best',
            '--merge-output-format', 'mp4',
            '--output', '%(title).200B.%(ext)s',
            '--windows-filenames',
            '--print', 'after_move:filepath',
        ]

        if os.path.exists(cookie_path):
            cmd.extend(['--cookies', cookie_path])

        cmd.append(url)

        print("Running: " + ' '.join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        printed_lines = [ln.strip() for ln in (result.stdout or '').splitlines() if ln.strip()]
        downloaded_path = printed_lines[-1] if printed_lines else None

        # Validate / best-effort fallback
        if downloaded_path and os.path.exists(downloaded_path):
            print(f"Download successful via CLI! File: {downloaded_path}")
            return downloaded_path

        guessed = _guess_latest_downloaded_media_file()
        if guessed and os.path.exists(guessed):
            print(f"Download successful via CLI! Guessed file: {guessed}")
            return guessed

        if os.path.exists('audio.mp4'):
            return 'audio.mp4'
        if os.path.exists('audio.webm'):
            return 'audio.webm'
        print("Download successful via CLI, but could not validate output filename")
        return downloaded_path
    except subprocess.CalledProcessError as e:
        print(f"CLI fallback failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"stderr: {e.stderr}")
        return None


def _try_audio_only_download(url, cookie_path):
    """Attempt audio-only download as last resort.

    Returns:
        str | None: Downloaded filepath if successful, otherwise None
    """
    try:
        print("Final attempt: downloading audio only...")
        cmd = [
            'yt-dlp',
            '--no-playlist',
            '--extractor-args', 'youtube:player_client=android',
            '--format', 'bestaudio',
            '--output', '%(title).200B.%(ext)s',
            '--windows-filenames',
            '--print', 'after_move:filepath',
        ]

        if os.path.exists(cookie_path):
            cmd.extend(['--cookies', cookie_path])

        cmd.append(url)

        print("Running: " + ' '.join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        printed_lines = [ln.strip() for ln in (result.stdout or '').splitlines() if ln.strip()]
        downloaded_path = printed_lines[-1] if printed_lines else None
        if downloaded_path and os.path.exists(downloaded_path):
            print(f"Audio download successful! File: {downloaded_path}")
            return downloaded_path

        guessed = _guess_latest_downloaded_media_file()
        if guessed and os.path.exists(guessed):
            print(f"Audio download successful! Guessed file: {guessed}")
            return guessed

        # Fallback guesses
        for candidate in ('audio.m4a', 'audio.webm', 'audio.opus', 'audio.mp3', 'audio.wav'):
            if os.path.exists(candidate):
                return candidate

        print("Audio download successful, but could not validate output filename")
        return downloaded_path
    except subprocess.CalledProcessError as e:
        print(f"Audio-only download failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"stderr: {e.stderr}")
        return None


def download_audio(url, output_file="audio.mp4"):
    """
    Download audio/video from URL with multiple fallback strategies.
    
    Args:
        url: YouTube or other video URL
        output_file: Output filename
        
    Returns:
        str | None: Downloaded video/audio filepath if successful, otherwise None
    """
    # Filenames are title-based now (see outtmpl), so we generally don't know
    # the final output filename upfront. Only delete when caller explicitly
    # provides a concrete output path.
    if output_file and output_file != "audio.mp4":
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
        except Exception:
            pass

    
    cookie_path = 'www.youtube.com_cookies.txt'
    
    # Strategy 1: Python API with Android client
    ydl_opts = _get_ydl_options(cookie_path)

    # Allow overriding output template when the caller provides a concrete path
    if output_file and output_file != "audio.mp4":
        ydl_opts['outtmpl'] = output_file
    
    downloaded = _try_python_download(url, ydl_opts)
    if downloaded:
        return downloaded
    
    # Strategy 2: CLI fallback
    downloaded = _try_cli_download(url, cookie_path)
    if downloaded:
        return downloaded
    
    # Strategy 3: Audio-only fallback
    downloaded = _try_audio_only_download(url, cookie_path)
    if downloaded:
        return downloaded
    
    # All strategies failed
    print("\nAll download strategies failed. Please check:")
    print("1. Video URL is correct and accessible")
    print("2. Export cookies from your browser to www.youtube.com_cookies.txt")
    print("3. Update yt-dlp: pip install -U yt-dlp")
    return None


def convert_mp4_to_audio(video_path, output_file="audio.mp3"):
    """
    Convert MP4 video to audio using ffmpeg.
    
    Args:
        video_path: Path to input video file
        output_file: Path to output audio file
        
    Returns:
        bool: True if successful, False otherwise
    """
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


# ============================================================================
# TRANSLATION HELPERS
# ============================================================================

async def translate_text_safe(translator, text, dest='zh-TW', max_retries=3):
    """
    Safely translate text with retry logic.
    
    Args:
        translator: Translator instance
        text: Text to translate
        dest: Destination language code
        max_retries: Maximum retry attempts
        
    Returns:
        str: Translated text or original if translation fails
    """
    for attempt in range(max_retries):
        try:
            result = await translator.translate(text, dest=dest)
            return result.text
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Translation failed after {max_retries} attempts: {e}")
                return text
            await asyncio.sleep(1)


async def translate_texts_batch(texts, dest='zh-TW', batch_size=10):
    """
    Translate texts in batches to avoid rate limiting.
    
    Args:
        texts: List of text strings to translate
        dest: Destination language code
        batch_size: Number of texts per batch
        
    Returns:
        list: Translated texts
    """
    translator = Translator()
    all_translations = []
    total = len(texts)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) - 1) // batch_size + 1
        
        print(f"Translating batch {batch_num}/{total_batches}")
        
        tasks = [translate_text_safe(translator, text, dest) for text in batch]
        batch_translations = await asyncio.gather(*tasks)
        all_translations.extend(batch_translations)
        
        # Report progress
        processed = len(all_translations)
        percent = int(processed * 100 / total) if total > 0 else 0
        
        # Delay between batches to avoid rate limiting
        if i + batch_size < len(texts):
            await asyncio.sleep(1)
    
    return all_translations


# ============================================================================
# SUMMARY GENERATION
# ============================================================================

def _try_openai_summary(text, system_prompt):
    """Attempt to generate summary using OpenAI API."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("OPENAI_API_KEY not set, skipping OpenAI")
        return None
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        user_prompt = "請幫我總結以下影片內容，越詳細越好，並且用中文+markdown格式回覆我。\n\n" + text
        
        # Try progressive model choices
        models = ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
        
        for model in models:
            try:
                print(f"Trying OpenAI model: {model}")
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=15000
                )
                
                summary = response.choices[0].message.content.strip()
                print("Summary generated via OpenAI")
                return f"\n{summary}"
            
            except Exception as e:
                print(f"Model {model} failed: {e}")
                continue
        
        print("All OpenAI models failed")
        return None
    
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None


def _try_gemini_summary(text, system_prompt):
    """Attempt to generate summary using Gemini API."""
    try:
        
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_prompt)
        prompt = "請幫我總結以下影片內容，越詳細愈好，並且用中文回覆我。\n\n" + text
        response = model.generate_content(prompt)
        
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                summary = candidate.content.parts[0].text
                print("Summary generated with Gemini")
                return f"\n{summary}"
        
        raise Exception("No valid Gemini response")
    
    except Exception as e:
        print(f"Gemini API failed: {e}")
        return None


def _try_ollama_summary(text):
    """Attempt to generate summary using Ollama (local CUDA)."""
    try:
        print("Trying Ollama fallback...")
        import torch
        import gc

        if not torch.cuda.is_available():
            print("CUDA not available for Ollama")
            return None

        import ollama
        client = ollama.Client()

        prompt = "請幫我總結以下影片內容，越詳細愈好，並且用中文回覆我。\n\n" + text
        response = client.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": prompt}]
        )

        summary = response['message']['content']
        print("Summary generated with Ollama")

        # Attempt best-effort cleanup to free VRAM and related resources.
        try:
            # Close or unload client if the API exposes those methods
            if hasattr(client, 'close'):
                try:
                    client.close()
                except Exception:
                    pass
            if hasattr(client, 'unload_model'):
                try:
                    client.unload_model()
                except Exception:
                    pass
            if hasattr(ollama, 'unload'):
                try:
                    ollama.unload()
                except Exception:
                    pass
        except Exception:
            # Ignore cleanup errors but continue with explicit local cleanup
            pass

        # Delete large objects and force garbage collection + CUDA cache clear
        try:
            del response
            del client
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
        except Exception:
            pass

        return summary
    
    except Exception as e:
        print(f"Ollama failed: {e}")
        return None




def gen_summary(choice='openai', system_prompt=None):
    if system_prompt is None:
        system_prompt = "你是一個會將轉錄文字詳細總結成中文的助理，請用繁體中文回覆。"
    
    
    # Read transcription
    try:
        with open('transcribe.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Failed to read transcription: {e}")
        return False
    
    # Try different providers
    summary = None
    
    if choice == 'openai' and not summary:
        tmp = _try_openai_summary(text, system_prompt)
        if tmp and len(tmp) > 10:
            summary = 'generated by openai\n'
            summary += tmp

    print(f'choice = {choice}')
    if choice != 'ollama' and not summary:
        tmp = _try_gemini_summary(text, system_prompt)
        if tmp and len(tmp) > 20:
            summary = 'generated by gemini\n'
            summary += tmp
    if not summary:        
        tmp = _try_ollama_summary(text)
        if tmp and len(tmp) > 10:
            summary = "generated by ollama\n"
            summary += tmp



    if not summary:
        print("All summary generation methods failed")
        return False
    
    # Write summary to file
    try:
        with open('summary.md', 'w', encoding='utf-8') as f:
            f.write(summary)
        print("Summary saved to summary.md")
        return True
    except Exception as e:
        print(f"Failed to write summary: {e}")
        return False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_timestamp(seconds):
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    millis = int((seconds - int(seconds)) * 1000)
    time_struct = time.gmtime(int(seconds))
    return f"{time_struct.tm_hour:02}:{time_struct.tm_min:02}:{time_struct.tm_sec:02},{millis:03}"


def get_gpu_info():
    """Get available GPU memory in bytes."""
    import torch
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    return total_memory - allocated_memory


def wait_for_gpu_memory(threshold_gb=5):
    """Wait until GPU has enough free memory."""
    import torch
    if not torch.cuda.is_available():
        return
    
    import gc
    threshold_bytes = threshold_gb * 1024 * 1024 * 1024
    
    while True:
        torch.cuda.empty_cache()
        gc.collect()
        if get_gpu_info() > threshold_bytes:
            break
        time.sleep(1)


def copy_to_repository(source_file):
    """Copy file to pCloud Obsidian directory."""
    import platform
    import shutil
    
    if platform.system() != 'Linux':
        if platform.system() == 'Windows':
            shutil.copy(source_file, '..\\notes\\')
        return
    
    target_path = '../notes/summary.md'
    
    try:
        shutil.copy(source_file, target_path)
        print("File copied successfully")
    except Exception as e:
        print(f"Initial copy failed: {e}")
        
        if wait_for_pcloud_mount():
            try:
                shutil.copy(source_file, target_path)
                print("File copied successfully after pCloud mount")
            except Exception as e2:
                print(f"Copy failed even after mounting: {e2}")
        else:
            print("Could not mount pCloud, copy operation failed")


def sanitize_url(url):
    """Clean and normalize YouTube URL."""
    url = url.replace('\\', '').strip()
    
    if '?v' in url:
        url = "https://www.youtube.com/watch?v" + url.split('?v')[-1]
    
    if '&' in url:
        url = url.split('&')[0]
    
    return url


# ============================================================================
# TRANSCRIPTION WORKFLOW
# ============================================================================

def process_video_input(video_arg):
    """
    Process video input and return audio file path.
    
    Args:
        video_arg: URL or file path to video
        
    Returns:
        str: Path to audio file, or None if failed
    """
    # Handle URL downloads
    if video_arg.startswith("http://") or video_arg.startswith("https://"):
        url = sanitize_url(video_arg)
        print(f"Downloading video from URL: {url}")
        
        if download_audio(url):
            return 'audio.mp4'
        else:
            print("Failed to download video")
            return None
    
    # Handle MP4 conversion
    if video_arg.endswith(".mp4"):
        if convert_mp4_to_audio(video_arg):
            return "audio.mp3"
        else:
            print("Failed to convert video")
            return None
    
    # Handle direct audio files
    if video_arg.endswith((".mp3", ".wav", ".m4a")):
        if os.path.exists(video_arg):
            return video_arg
        else:
            print(f"Audio file not found: {video_arg}")
            return None
    
    print("Please provide a valid video URL, .mp4, .mp3, .wav, or .m4a file.")
    return None


def transcribe_audio(model, audio_file):
    """
    Transcribe audio file using Whisper.
    
    Args:
        model: Whisper model instance
        audio_file: Path to audio file
        
    Returns:
        dict: Transcription result or None if failed
    """
    print(f"Transcribing audio: {audio_file}")
    
    try:
        result = model.transcribe(audio_file, fp16=False, verbose=False)
        return result
    except Exception as e:
        print(f"Transcription failed: {e}")
        return None


def save_subtitles(result):
    """
    Save subtitles and transcription to files.
    
    Args:
        result: Whisper transcription result
        
    Returns:
        tuple: (texts, translations, skip_translation)
    """
    detected_lang = result.get("language", "")
    print(f"Detected language: {detected_lang}")
    skip_translation = detected_lang.startswith("zh")
    
    if not isinstance(result, dict) or "segments" not in result:
        # Handle full text result
        with open("transcription.txt", "w", encoding='utf-8') as f:
            f.write(f"Detected Language: {detected_lang}\n")
            f.write("=" * 50 + "\n\n")
            f.write(result.get('text', str(result)))
        print("Full text saved to transcription.txt")
        return [], [], skip_translation
    
    segments = result["segments"]
    texts = [seg['text'].strip() for seg in segments if seg['text'].strip()]
    
    # Save English subtitles
    with open('subtitle.srt', 'w', encoding='utf-8') as f:
        for seg in segments:
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            text = seg['text'].strip()
            f.write(f"{start} --> {end}\n{text}\n\n")
    
    # Translate if needed
    translations = []
    if not skip_translation and texts:
        print("Translating text to Chinese...")
        try:
            translations = asyncio.run(translate_texts_batch(texts))
        except Exception as e:
            print(f"Translation failed: {e}")
            translations = texts
    
    # Save Chinese subtitles
    with open("cn_subtitle.srt", "w", encoding='utf-8') as f:
        for idx, seg in enumerate(segments):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            f.write(f"{start} --> {end}\n")
            if not skip_translation and idx < len(translations):
                f.write(f"{translations[idx]}\n")
            f.write("\n")
    
    # Save plain text files
    with open('transcribe.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(texts))
    
    with open('chinese.txt', 'w', encoding='utf-8') as f:
        if skip_translation:
            f.write('\n'.join(texts))
        else:
            f.write('\n'.join(translations))
    
    print(f"Processed {len(texts)} segments")
    return texts, translations, skip_translation


# ============================================================================
# MAIN
# ============================================================================



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
    
    # Write initial progress immediately
    
    # Unset all proxy environment variables directly in Python
    # (Running a shell script won't affect the current process's environment)
    proxy_vars = [
        'http_proxy', 'https_proxy', 'ftp_proxy', 'socks_proxy', 'all_proxy',
        'HTTP_PROXY', 'HTTPS_PROXY', 'FTP_PROXY', 'SOCKS_PROXY', 'ALL_PROXY',
        'no_proxy', 'NO_PROXY'
    ]
    for var in proxy_vars:
        os.environ.pop(var, None)
    print("All proxy environment variables cleared")
    
    parser = argparse.ArgumentParser(description="Transcribe and translate audio from video files.")
    parser.add_argument('--summary', action='store_true', help="Generate summary of the transcription")
    parser.add_argument('--video', nargs='?', help="Video URL or file path to transcribe")
    parser.add_argument('--copy', action='store_true', help='copy summary to pcloud')
    parser.add_argument('--model', default='openai', help='use specific model for transcription')
    parser.add_argument('--system-prompt', default=None, help='Custom system prompt for AI summary generation')
    wait_for_gpu_memory()


    import torch
    if torch.cuda.is_available():
        import gc
        while 1:
            torch.cuda.empty_cache()
            gc.collect()
            if get_gpu_info() > 5 * 1024 * 1024 * 1024:
                break

    args = parser.parse_args()
    choice   = args.model
    video = args.video
    summary = args.summary
    copy = args.copy
    system_prompt = args.system_prompt
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    print(f"copy mode: {copy}")
    print(f"Summary mode: {summary}")
    print(f"Processing: {video}")
    
    def _convert_video_to_wav(input_path: str, output_path=None) -> str:
        src_path = Path(input_path)
        if output_path is None:
            output_path = str(src_path.with_suffix('.wav'))

        cmd = [
            'ffmpeg',
            '-y',
            '-i', str(src_path),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            output_path,
        ]
        subprocess.run(cmd, check=True)
        return output_path

    VIDEO_EXTS = {'.mp4', '.mkv', '.flv', '.avi', '.mov', '.webm', '.m4v', '.ts'}
    AUDIO_EXTS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}

    # Handle different input types
    if not video:
        print("Please provide --video URL or file path")
        sys.exit(1)

    if video.startswith("http://") or video.startswith("https://"):
        url = sanitize_url(video)
        print(f"Downloading video from URL: {url}")
        downloaded_path = download_audio(url)
        if not downloaded_path:
            print("Failed to download video")
            sys.exit(1)
        # Always convert downloaded video/audio container to .wav
        audio_file = _convert_video_to_wav(downloaded_path, 'audio.wav')
    else:
        src_path = Path(video)
        if not src_path.exists():
            print(f"Input file not found: {video}")
            sys.exit(1)

        ext = src_path.suffix.lower()
        if ext in AUDIO_EXTS:
            audio_file = str(src_path)
        elif ext in VIDEO_EXTS:
            audio_file = _convert_video_to_wav(str(src_path))
        else:
            print("Please provide a valid video URL, or a media file (.mp4/.mkv/.flv/.avi/.mov/.webm/.m4v/.ts/.mp3/.wav/.m4a)")
            sys.exit(1)

    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Initialize model
    print("Loading Whisper model...")
    model = whisper.load_model("turbo")
    print('loading whisper model done')

    # Transcribe audio
    print(f"Transcribing audio... {audio_file}")
    try:
        result = model.transcribe(audio_file, fp16=False, verbose=False)
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)
    

    with open('subtitle.srt', 'w', encoding='utf-8') as f:
        for original in result.get("segments", []):
            start = original['start']
            end = original['end']
            text = original['text'].strip()
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")

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
        output_file = "cn_subtitle.srt"
        print(f"Writing results to {output_file}...")
        
        with open(output_file, "w", encoding='utf-8') as f:
            
            for idx, original in enumerate(result.get('segments', [])):
                f.write(f"{format_timestamp(original['start'])} --> {format_timestamp(original['end'])}\n")
                if not skip_translation and idx < len(translations):
                    f.write(f"{translations[idx]}\n")
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
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("Generating summary...")
        gen_summary(choice=choice, system_prompt=system_prompt)
    if copy:
        copy_to_repository('summary.md')

        
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
