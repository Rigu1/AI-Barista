import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import librosa
import soundfile as sf
import os
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore")

app = FastAPI()

print("="*50)
print("[FastAPI] 모델을 메모리로 옮기는 중...")
pipe = pipeline("audio-classification", model="dima806/music_genres_classification")
print("[FastAPI] 로딩 완료")
print("="*50)

class AnalyzeRequest(BaseModel):
    filepath: str
    start_time: float
    duration: float

@app.post("/analyze")
async def analyze_audio(req: AnalyzeRequest):
    try:
        y, sr = librosa.load(req.filepath, sr=16000, offset=req.start_time, duration=req.duration)
        temp_path = f"temp_fastapi_{os.getpid()}.wav"
        sf.write(temp_path, y, sr)
        
        results = pipe(temp_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        output_dict = {res['label']: res['score'] for res in results}
        return output_dict
        
    except Exception as e:
        print(f"[FastAPI]: Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    print(f"✨ [FastAPI] 분석 서버 가동")