# Whisperrr - 語音轉錄與摘要工具

一個使用 OpenAI Whisper 進行語音轉錄並生成摘要的工具。

## 功能特性

- 🎵 支援多種音訊格式（MP3, WAV, M4A）
- 🎬 支援影片格式（MP4）
- 🌐 支援 YouTube 影片下載
- 🔄 自動翻譯（檢測語言並翻譯成中文）
- 📝 使用 Google Gemini API 生成詳細摘要
- 🔧 多重備援方案（Ollama、基本摘要）

## 安裝需求

```bash
# 使用 uv 安裝依賴
uv add openai-whisper
uv add yt-dlp
uv add googletrans==4.0.0rc1
uv add google-generativeai
```

## 環境設定

設定 Google Gemini API 金鑰：

```bash
export GEMINI_API_KEY="your_api_key_here"
```

## 使用方法

### 基本用法

```bash
# 轉錄音訊檔案
uv run python3 main.py --video audio.mp3

# 轉錄影片檔案
uv run python3 main.py --video video.mp4

# 轉錄 YouTube 影片
uv run python3 main.py --video "https://www.youtube.com/watch?v=example"
```

### 包含摘要功能

```bash
# 轉錄並生成摘要
uv run python3 main.py --summary --video audio.mp3

# 使用便利腳本
./run_with_summary.sh video.mp4
./run_with_summary.sh "https://www.youtube.com/watch?v=example"
```

## 輸出檔案

- `transcription.txt` - 完整的轉錄結果（包含原文和翻譯）
- `chinese.txt` - 中文版本的轉錄內容
- `transcribe.txt` - 純文字轉錄內容
- `summary.txt` - AI 生成的影片摘要（需要 --summary 參數）

## 摘要生成方式

程式會按照以下順序嘗試生成摘要：

1. **Google Gemini API** - 使用 Gemini-1.5-flash 模型
2. **Ollama 備援** - 使用本地 DeepSeek-R1 模型（需要 CUDA）
3. **基本摘要** - 從轉錄文字提取重點

## 故障排除

如果遇到問題：

1. 確認 API 金鑰設定正確
2. 檢查網路連線
3. 確認音訊/影片檔案格式支援
4. 查看終端錯誤訊息

## 範例

```bash
# 完整範例：下載 YouTube 影片、轉錄、翻譯並生成摘要
uv run python3 main.py --summary --video "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

## 注意事項

- 大型檔案可能需要較長處理時間
- 需要網路連線來使用翻譯和摘要功能
- 建議使用 GPU 加速 Whisper 模型（如果可用）