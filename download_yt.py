from yt_dlp import YoutubeDL
import subprocess
import os


def download_audio(url, output_file="audio.mp4"):
    """
    Download audio/video from URL with multiple fallback strategies.
    
    Args:
        url: YouTube or other video URL
        output_file: Output filename
        
    Returns:
        bool: True if successful, False otherwise
    """
    # if os.path.exists(output_file):
    #     os.remove(output_file)

    
    cookie_path = 'www.youtube.com_cookies.txt'
    
    # Strategy 1: Python API with Android client
    ydl_opts = _get_ydl_options(cookie_path)
    
    if _try_python_download(url, ydl_opts):
        return True
    
    # Strategy 2: CLI fallback
    if _try_cli_download(url, cookie_path):
        return True
    
    
    # All strategies failed
    print("\nAll download strategies failed. Please check:")
    print("1. Video URL is correct and accessible")
    print("2. Export cookies from your browser to www.youtube.com_cookies.txt")
    print("3. Update yt-dlp: pip install -U yt-dlp")
    return False



def _get_ydl_options(cookie_path):
    """Get yt-dlp options with Android client configuration."""
    opts = {
        # Request the best available video + best audio, fall back to best
        # This lets yt-dlp automatically pick the highest resolution available.
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': '%(title).10s.%(ext)s',
        'noplaylist': True,
        'merge_output_format': 'mp4',
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
    """Attempt download using yt-dlp Python API."""
    try:
        print("Attempting download with yt-dlp...")
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download successful!")
        return True
    except Exception as e:
        print(f"Android client failed: {e}")
        return False


def _try_cli_download(url, cookie_path):
    """Attempt download using yt-dlp CLI."""
    try:
        print("Trying CLI fallback...")
        cookie_arg = f'--cookies "{cookie_path}"' if os.path.exists(cookie_path) else ''
        
        cmd = (
            f'yt-dlp --no-playlist '
            # f'--extractor-args "youtube:player_client=ios,web,android" '
                f'--format "bestvideo+bestaudio/best" '
            f'--merge-output-format mp4 '
            f'--output "%(title).10s.%(ext)s" '
            f'{cookie_arg} '
            f'"{url}"'
        )
        
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("Download successful via CLI!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"CLI fallback failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def main():
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else input("Enter video URL: ")
    if url.startswith("http://") or url.startswith("https://"):
        url = url.replace('\\' , '').strip()
        print(url)
        if '?v' in url:
            url = "https://www.youtube.com/watch?v" + url.split('?v')[-1]
        if '&' in url:
            url = url.split('&')[0]
        print(f"Downloading video from URL: {url}")


    success = download_audio(url)
    if success:
        print("Audio downloaded successfully.")
    else:
        print("Audio download failed.")


if __name__ == "__main__":
    main()