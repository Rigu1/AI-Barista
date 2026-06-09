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
        segments = [0.0, req.duration - 5.0]
        
        final_scores = {}
        sr = 16000
        
        for start in segments:
            y, _ = librosa.load(req.filepath, sr=sr, offset=req.start_time + start, duration=5.0)
            
            results = pipe(y, top_k=3)
            
            for res in results:
                final_scores[res['label']] = final_scores.get(res['label'], 0) + res['score']
        
        return {label: round(score / 2.0, 4) for label, score in final_scores.items()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)