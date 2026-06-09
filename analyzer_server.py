import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import librosa
import numpy as np
import os
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore")

app = FastAPI()

print("="*50)
print("[FastAPI] 모델 로딩 시작...")
pipe = pipeline("audio-classification", model="dima806/music_genres_classification", device=-1)
print("[FastAPI] 모델 로딩 완료")
print("="*50)

class AnalyzeRequest(BaseModel):
    filepath: str
    start_time: float
    duration: float

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import librosa
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore")

app = FastAPI()

print("="*50)
print("[FastAPI] 모델 로딩 시작...")
pipe = pipeline("audio-classification", model="dima806/music_genres_classification", device=-1)
print("[FastAPI] 모델 로딩 완료")
print("="*50)

class AnalyzeRequest(BaseModel):
    filepath: str
    start_time: float
    duration: float

@app.post("/analyze")
async def analyze_audio(req: AnalyzeRequest):
    try:
        analyze_duration = min(req.duration, 10.0) 
        
        y, sr = librosa.load(req.filepath, sr=16000, offset=req.start_time, duration=analyze_duration)
        
        results = pipe(y, top_k=3)
        
        return {res['label']: float(res['score']) for res in results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)