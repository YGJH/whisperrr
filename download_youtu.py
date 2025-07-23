import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("video")

url = parser.parse_args().video
GREEN = '\033[92m'  # ANSI escape code for bright green


# url = input().strip()
if '\\' in url:
    url = url.replace('\\','')

if '&' in url:
    url = url.split('&')[0]

print(f"{GREEN}downloading: {url}")

subprocess.run("uv run yt-dlp --output \"%(title)s.%(ext)s\" --embed-thumbnail --add-metadata --merge-output-format mp4 " + url , shell=True, check=True)
