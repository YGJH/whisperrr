#!/bin/bash
# 用於執行包含摘要功能的轉錄程式
set -o pipefail
if [ -z "$1" ]; then
    echo "使用方法: ./run_with_summary.sh <video_file_or_url>"
    echo "例如: ./run_with_summary.sh video.mp4"
    echo "或: ./run_with_summary.sh https://www.youtube.com/watch?v=example"
    exit 1
fi

echo "開始處理影片並生成摘要..."
uv run python3 main.py --summary --video "$1" --copy
