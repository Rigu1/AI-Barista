import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import soundfile as sf
import numpy as np
import torch
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore")

app = FastAPI()

print("="*50)
print("[FastAPI] 모델 로딩 중... (잠시만 기다려주세요)")

raw_pipe = pipeline("audio-classification", model="dima806/music_genres_classification", device=-1)

try:
    raw_pipe.model = torch.quantization.quantize_dynamic(
        raw_pipe.model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    print("[FastAPI] 모델 로딩 완료")
except Exception as e:
    print(f"[FastAPI] {e}")

print("[FastAPI] 모든 모델 로딩 완료")
print("="*50)

class AnalyzeRequest(BaseModel):
    filepath: str
    start_time: float
    duration: float

@app.post("/analyze")
async def analyze_audio(req: AnalyzeRequest):
    try:
        analyze_duration = min(req.duration, 10.0)
        target_sr = 16000
        
        with sf.SoundFile(req.filepath) as f:
            native_sr = f.samplerate
            start_sample = int(req.start_time * native_sr)
            max_samples = int(analyze_duration * native_sr)
            
            f.seek(start_sample)
            y = f.read(frames=max_samples, dtype='float32')
            
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
                
            if native_sr != target_sr:
                secs = len(y) / native_sr
                num_samples = int(secs * target_sr)
                y = np.interp(np.linspace(0, len(y), num_samples), np.arange(len(y)), y)

        results = raw_pipe(y, top_k=3)
        
        return {res['label']: float(res['score']) for res in results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)