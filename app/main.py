from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
from .diarization_api import ping
import os
import tempfile
import torch
print(torch.version.cuda)    # PyTorch가 빌드된 CUDA 버전
print(torch.backends.cudnn.version())  # cuDNN 버전
print(torch.cuda.is_available())       # GPU 인식 여부
from .utils import ensure_tmp_copy, get_media_duration_sec, is_silent, transcode_to_wav_mono16k
from .asr import transcribe_file
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


app = FastAPI(title="Mina ASR API", version="1.0.0")

# CORS (Electron에선 크게 문제 없지만 유연하게 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 특정 origin으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscribeResponse(BaseModel):
    ok: bool
    text: str
    segments: list[dict]
    language: Optional[str] = None
    language_probability: Optional[float] = None
    duration_sec: Optional[float] = None
    model: Optional[str] = None
    task: Literal["transcribe", "translate"] = "transcribe"
    detail: Optional[str] = None

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(..., description="오디오/영상 파일(mp3, wav, webm, ogg 등)"),
    language: str = Form("auto"),
    task: Literal["transcribe", "translate"] = Form("transcribe"),
):
    """
    멀티파트 입력을 받아 Whisper로 STT 수행.
    - language: "auto" | "ko" | "en" …
    - task: "transcribe"(원문언어 그대로) | "translate"(영어로 번역)
    """
    tmp_in_path = None
    wav_path = None

    try:
        # 1) 업로드 파일 → 임시 저장
        suffix = os.path.splitext(file.filename or "")[1] or ".bin"
        file_bytes = await file.read()
        tmp_in_path = ensure_tmp_copy(None, file_bytes, suffix)

        # 2) Whisper 친화 포맷으로 변환 (mono16k wav)
        ok, wav_path, log = transcode_to_wav_mono16k(tmp_in_path)
        if not ok:
            raise HTTPException(status_code=415, detail=f"ffmpeg 변환 실패: {log[:4000]}")

        # 3) 길이 검사 (녹음 실수/무음 대응)
        raw_dur = get_media_duration_sec(wav_path)  # 변환된 wav 기준
        if raw_dur is not None and raw_dur <= 0.5:
            raise HTTPException(
                status_code=400,
                detail="오디오 길이가 너무 짧습니다(0.5초 이하). 다시 녹음해주세요."
            )

        # 3-1) 무음 검사
        if is_silent(wav_path):
            raise HTTPException(
                status_code=400,
                detail="업로드된 오디오가 무음으로 감지되었습니다. 다시 녹음해주세요."
            )

        # 4) STT
        result = transcribe_file(wav_path, language=language, task=task)

        # 5) 응답
        return TranscribeResponse(
            ok=True,
            text=result["text"],
            segments=result["segments"],
            language=result.get("language"),
            language_probability=result.get("language_probability"),
            duration_sec=result.get("duration"),
            model=result.get("model_name"),
            task=task,
        )

    finally:
        # 임시파일 정리
        try:
            if tmp_in_path and os.path.exists(tmp_in_path):
                os.remove(tmp_in_path)
        except Exception:
            pass
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

@app.get("/healthz")
def healthz():
    return {"ok": True, "msg": "ready"}

@app.get("/ping")
def healthz():
    return ping()
