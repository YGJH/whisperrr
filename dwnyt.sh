#!/bin/bash
set -o pipefail
cd ~/Documents/whisperrr
uv run download_yt.py $1
