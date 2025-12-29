#!/bin/bash
set -o pipefail
cd ~/Documents/whisperrr

for url in "$@"; do
    echo "Downloading $url..."
    uv run download_yt.py "$url"
done
