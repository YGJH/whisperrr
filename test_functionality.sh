#!/bin/bash
# 測試腳本 - 確認所有功能正常

echo "=== Whisperrr 功能測試 ==="
echo

# 檢查是否有音訊檔案
if ls *.mp3 >/dev/null 2>&1; then
    AUDIO_FILE=$(ls *.mp3 | head -1)
    echo "發現音訊檔案: $AUDIO_FILE"
    echo "開始測試轉錄和摘要功能..."
    echo
    
    # 執行轉錄和摘要
    uv run python3 main.py --summary --video "$AUDIO_FILE"
    
    echo
    echo "=== 測試完成 ==="
    echo "生成的檔案："
    ls -la *.txt 2>/dev/null || echo "沒有找到 .txt 檔案"
    
    if [ -f "summary.txt" ]; then
        echo
        echo "=== 摘要內容預覽 ==="
        head -20 summary.txt
    fi
else
    echo "沒有找到 .mp3 檔案進行測試"
    echo "請先放置一個 .mp3 檔案到此目錄"
fi
