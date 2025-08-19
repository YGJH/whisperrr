import torchaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutomaticSpeechRecognitionPipeline

# 1. Load audio
import os
import sys
if len(sys.argv) < 2:
    print("Usage: python breeze_main.py <audio_file>")
    sys.exit(1) 

audio_path = sys.argv[1]
waveform, sample_rate = torchaudio.load(audio_path)

if audio_path.endswith(".mp3"):
    # Convert MP3 to WAV if necessary
    waveform, sample_rate = torchaudio.load(audio_path, format="mp3")
    audio_path = audio_path.replace(".mp3", ".wav")
    torchaudio.save(audio_path, waveform, sample_rate)
    print(f"Converted {audio_path} to WAV format.")


# 2. Preprocess
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0)                         
waveform = waveform.squeeze().numpy()                        

if sample_rate != 16_000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16_000)
    waveform = resampler(torch.tensor(waveform)).numpy()
    sample_rate = 16_000

# 3. Load Model
processor = WhisperProcessor.from_pretrained("MediaTek-Research/Breeze-ASR-25")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WhisperForConditionalGeneration.from_pretrained("MediaTek-Research/Breeze-ASR-25").to(device).eval()

# 4. Build Pipeline
asr_pipeline = AutomaticSpeechRecognitionPipeline(
    model=model,ㄧ
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=0
)

# 6. Inference
output = asr_pipeline(waveform, return_timestamps=True)  
print("Result:", output["text"])