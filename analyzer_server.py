import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import soundfile as sf
import numpy as np
import torch
import warnings
from pathlib import Path
from typing import Optional
from transformers import pipeline

warnings.filterwarnings("ignore")

app = FastAPI()
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
TARGET_SR = 16000
MAX_ANALYZE_DURATION = 10.0

class AnalysisError(Exception):
    def __init__(self, status_code: int, message: str, log_message: Optional[str] = None):
        self.status_code = status_code
        self.message = message
        self.log_message = log_message or message
        super().__init__(self.log_message)

@app.exception_handler(AnalysisError)
async def analysis_error_handler(request: Request, exc: AnalysisError):
    print(f"[FastAPI 분석 에러] {exc.log_message}")
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})

@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "분석 요청 처리 중 오류가 발생했습니다."
    print(f"[FastAPI HTTP 에러] {message}")
    return JSONResponse(status_code=exc.status_code, content={"error": message})

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    print(f"[FastAPI 요청 검증 에러] {exc}")
    return JSONResponse(status_code=400, content={"error": "분석 요청 형식이 올바르지 않습니다."})

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

def resolve_audio_path(filepath: str) -> Path:
    if not filepath:
        raise AnalysisError(400, "분석할 오디오 파일 경로가 없습니다.")

    try:
        audio_path = Path(filepath).resolve(strict=True)
    except FileNotFoundError:
        raise AnalysisError(404, "업로드된 오디오 파일을 찾을 수 없습니다.", f"File not found: {filepath}")
    except Exception as e:
        raise AnalysisError(400, "오디오 파일 경로가 올바르지 않습니다.", str(e))

    try:
        audio_path.relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        raise AnalysisError(403, "허용되지 않은 오디오 파일 경로입니다.", f"Path outside upload dir: {audio_path}")

    if not audio_path.is_file():
        raise AnalysisError(400, "분석할 수 있는 오디오 파일이 아닙니다.", f"Path is not a file: {audio_path}")

    return audio_path

def validate_request_values(req: AnalyzeRequest):
    if not np.isfinite(req.start_time) or not np.isfinite(req.duration):
        raise AnalysisError(400, "분석 구간 값이 올바르지 않습니다.")

    if req.start_time < 0 or req.duration <= 0:
        raise AnalysisError(400, "분석 구간 값이 올바르지 않습니다.")

    return min(req.duration, MAX_ANALYZE_DURATION)

def read_audio_segment(audio_path: Path, start_time: float, duration: float):
    try:
        with sf.SoundFile(str(audio_path)) as f:
            native_sr = f.samplerate
            start_sample = int(start_time * native_sr)
            max_samples = int(duration * native_sr)

            if native_sr <= 0 or f.frames <= 0:
                raise AnalysisError(400, "오디오 파일에 분석할 샘플이 없습니다.", f"Invalid audio metadata: sr={native_sr}, frames={f.frames}")

            if start_sample >= f.frames:
                raise AnalysisError(400, "선택한 시작 시간이 오디오 길이를 벗어났습니다.", f"start_sample={start_sample}, frames={f.frames}")

            f.seek(start_sample)
            y = f.read(frames=max_samples, dtype='float32')
    except AnalysisError:
        raise
    except (sf.LibsndfileError, RuntimeError) as e:
        raise AnalysisError(415, "분석 서버가 이 오디오 형식을 읽지 못했습니다.", str(e))
    except Exception as e:
        raise AnalysisError(500, "오디오 파일을 읽는 중 오류가 발생했습니다.", str(e))

    if y.size == 0:
        raise AnalysisError(400, "선택한 구간에 분석할 오디오가 없습니다.")

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    if y.size == 0 or not np.any(np.isfinite(y)):
        raise AnalysisError(400, "선택한 구간에 분석할 수 있는 오디오 신호가 없습니다.")

    return y, native_sr

def resample_audio(y: np.ndarray, native_sr: int):
    if native_sr == TARGET_SR:
        return y

    try:
        secs = len(y) / native_sr
        num_samples = int(secs * TARGET_SR)
        if num_samples <= 0:
            raise AnalysisError(400, "분석할 오디오 구간이 너무 짧습니다.")

        return np.interp(
            np.linspace(0, len(y) - 1, num_samples),
            np.arange(len(y)),
            y
        ).astype('float32')
    except AnalysisError:
        raise
    except Exception as e:
        raise AnalysisError(500, "오디오 리샘플링 중 오류가 발생했습니다.", str(e))

def run_genre_model(y: np.ndarray):
    try:
        results = raw_pipe(y, top_k=3)
    except Exception as e:
        raise AnalysisError(500, "음악 장르 모델 추론에 실패했습니다.", str(e))

    if not isinstance(results, list) or len(results) == 0:
        raise AnalysisError(502, "음악 장르 모델 결과가 비어 있습니다.")

    output = {}
    for item in results:
        label = item.get('label') if isinstance(item, dict) else None
        score = item.get('score') if isinstance(item, dict) else None

        if not isinstance(label, str) or not label:
            raise AnalysisError(502, "음악 장르 모델 결과 형식이 올바르지 않습니다.", f"Invalid label item: {item}")

        try:
            score = float(score)
        except (TypeError, ValueError):
            raise AnalysisError(502, "음악 장르 모델 점수가 올바르지 않습니다.", f"Invalid score item: {item}")

        if not np.isfinite(score):
            raise AnalysisError(502, "음악 장르 모델 점수가 올바르지 않습니다.", f"Non-finite score item: {item}")

        output[label] = score

    return output

@app.post("/analyze")
async def analyze_audio(req: AnalyzeRequest):
    analyze_duration = validate_request_values(req)
    audio_path = resolve_audio_path(req.filepath)
    y, native_sr = read_audio_segment(audio_path, req.start_time, analyze_duration)
    y = resample_audio(y, native_sr)
    return run_genre_model(y)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
