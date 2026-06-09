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

@app.post("/analyze")
async def analyze_audio(req: AnalyzeRequest):
    try:
        segment_duration = 10.0
        segments = []
        for i in range(3):
            start = req.start_time + (i * segment_duration)
            y, sr = librosa.load(req.filepath, sr=16000, offset=start, duration=segment_duration)
            segments.append(y)
        
        all_results = []
        for seg in segments:
            all_results.append(pipe(seg))
            
        return {"status": "success", "msg": "3구간 분할 분석 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)