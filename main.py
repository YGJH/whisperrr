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

def download_audio(url, output_file="audio"):
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
        try:
            print("Trying fallback with yt-dlp...")
            cmd = f'uv run yt-dlp --output \"audio.mp3\" --embed-thumbnail --extract-audio --audio-format mp3 --audio-quality 320K {url}'
            # cmd = cmd.split()
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


def gen_summary():
    try:
        print("api key = ", os.getenv('GEMINI_API_KEY'))
        import google.generativeai as genai
        print("import successful")
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        with open('transcribe.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        print("Generating summary...")
        prompt = "請幫我總結以下影片內容，越詳細愈好，並且用中文回覆我。\n\n" + text
        response = model.generate_content(prompt)

        # 新版本的 API 需要檢查 response 是否有效
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                summary_text = candidate.content.parts[0].text
                with open('summary.md', 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                print("Summary generated and saved to summary.md")
                return True
            else:
                raise Exception("No content parts in response")
        else:
            raise Exception("No candidates in response")
            
    except Exception as e:
        print(f"Gemini API failed: {e}")
        try:
            print("Trying Ollama fallback...")
            import torch
            if torch.cuda.is_available():
                import ollama
                client = ollama.Client()
                with open('transcribe.txt', 'r', encoding='utf-8') as f:
                    text = f.read()
                print("Generating summary with Ollama...")
                prompt = "請幫我總結以下影片內容，越詳細愈好，並且用中文回覆我。\n\n" + text
                response = client.chat(
                    model="deepseek-r1:7b",
                    messages=[{"role": "user", "content": prompt}]
                )
                with open('summary.md', 'w', encoding='utf-8') as f:
                    f.write(response['message']['content'])
                print("Summary generated and saved to summary.md")
                return True
            else:
                print("CUDA not available, trying CPU-based summary...")
                # Simple fallback - create a basic summary from the transcription
                with open('transcribe.txt', 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Create a simple summary by taking first few lines and key points
                lines = text.split('\n')
                summary_lines = []
                summary_lines.append("=== 影片內容摘要 ===")
                summary_lines.append("(由於API限制，這是基於轉錄文字的簡單摘要)")
                summary_lines.append("")
                
                # Take first 5 lines as introduction
                for i, line in enumerate(lines[:5]):
                    if line.strip():
                        summary_lines.append(f"• {line.strip()}")
                
                summary_lines.append("")
                summary_lines.append("=== 主要內容 ===")
                
                # Take some key lines from the middle and end
                mid_point = len(lines) // 2
                for i in range(mid_point, min(mid_point + 3, len(lines))):
                    if lines[i].strip():
                        summary_lines.append(f"• {lines[i].strip()}")
                
                if len(lines) > 10:
                    summary_lines.append("")
                    summary_lines.append("=== 結論部分 ===")
                    for line in lines[-3:]:
                        if line.strip():
                            summary_lines.append(f"• {line.strip()}")
                
                summary_text = '\n'.join(summary_lines)
                with open('summary.md', 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                print("Basic summary generated and saved to summary.md")
                return True
        except Exception as e2:
            print(f"Ollama fallback also failed: {e2}")
            print("Creating basic summary from transcription...")
            try:
                with open('transcribe.txt', 'r', encoding='utf-8') as f:
                    text = f.read()
                
                lines = text.split('\n')
                summary_lines = []
                summary_lines.append("=== 影片內容摘要 ===")
                summary_lines.append("(自動生成的基本摘要)")
                summary_lines.append("")
                
                for i, line in enumerate(lines):
                    if line.strip() and i % 3 == 0:  # Take every 3rd line
                        summary_lines.append(f"• {line.strip()}")
                
                summary_text = '\n'.join(summary_lines)
                with open('summary.md', 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                print("Basic summary generated and saved to summary.md")
                return True
            except Exception as e3:
                print(f"All summary generation methods failed: {e3}")
                return False

def main():
    # Fix for Windows event loop policy
    import argparse
    summary = False
    parser = argparse.ArgumentParser(description="Transcribe and translate audio from video files.")
    parser.add_argument('--summary', action='store_true', help="Generate summary of the transcription")
    parser.add_argument('--video', nargs='?', help="Video URL or file path to transcribe")
    parser.add_argument('--copy', action='store_true', help='copy summary to pcloud')
    
    args = parser.parse_args()
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
        gen_summary()
    if copy:
        path = os.getenv('pCloud_Path')
        print(f'copy summary.md to {path}')
        cmd = 'cp summary.md ' + path + 'documents/obsidian/summary.md'  
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    # Clean up temporary audio file if it was downloaded/converted
    # if audio_file in ["audio.mp3"] and audio_file != video:
    #     try:
    #         os.remove(audio_file)
    #         print("Cleaned up temporary audio file")
    #     except:
    #         pass

if __name__ == "__main__":
    main()
