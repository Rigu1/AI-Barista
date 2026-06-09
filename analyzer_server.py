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
print("[FastAPI] 모델 로딩 준비...")
raw_pipe = pipeline("audio-classification", model="dima806/music_genres_classification", device=-1)
try:
    raw_pipe.model = torch.quantization.quantize_dynamic(
        raw_pipe.model, {torch.nn.Linear}, dtype=torch.qint8
    )
except Exception as e:
    pass
print("[FastAPI] 로딩 완료!")
print("="*50)

class AnalyzeRequest(BaseModel):
    filepath: str
    start_time: float
    duration: float

@app.post("/analyze")
async def analyze_audio(req: AnalyzeRequest):
    try:
        target_sr = 16000
        
        with sf.SoundFile(req.filepath) as f:
            native_sr = f.samplerate
            total_frames = f.frames
            total_duration_sec = total_frames / native_sr
            
            if total_duration_sec <= 10.0:
                y = f.read(dtype='float32')
            else:
                f.seek(0)
                y_front = f.read(frames=int(5.0 * native_sr), dtype='float32')
                
                start_back = max(0, total_frames - int(5.0 * native_sr))
                f.seek(start_back)
                y_back = f.read(frames=int(5.0 * native_sr), dtype='float32')
                
                y = np.concatenate((y_front, y_back))
                
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