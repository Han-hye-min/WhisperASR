import os
from typing import Optional, Any, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import torch
from pyannote.audio import Pipeline

from app.utils import save_upload_to_tmp
from app.diarization_utils import build_speaker_map, to_rttm


# ===== 환경설정 =====
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
load_dotenv()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

HF_TOKEN = os.getenv("HF_TOKEN")
print("토큰 값 검증 : ",HF_TOKEN)
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN 환경변수가 필요합니다. (Hugging Face token)")

PIPELINE_NAME = "pyannote/speaker-diarization-3.1"

print(f"[pyannote] Loading pipeline: {PIPELINE_NAME}")
_pipeline = Pipeline.from_pretrained(PIPELINE_NAME, use_auth_token=HF_TOKEN)
_pipeline.to(get_device())
print(f"[pyannote] Loaded to device: {get_device()}")


# ===== FastAPI =====
app = FastAPI(title="Mina ASR + Diarization API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def diarize_file(audio_path: str,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None) -> Dict[str, Any]:
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        kwargs["max_speakers"] = int(max_speakers)

    diarization = _pipeline(audio_path, **kwargs)

    raw_labels = [label for _, _, label in diarization.itertracks(yield_label=True)]
    spk_map = build_speaker_map(raw_labels)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
            "speaker": spk_map[speaker],
        })

    return {
        "num_speakers": len(set([s["speaker"] for s in segments])),
        "segments": segments,
        "rttm": to_rttm(segments)
    }


@app.post("/diarize")
async def diarize_endpoint(
    file: UploadFile = File(...),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None)
):
    tmp_path = None
    try:
        tmp_path = save_upload_to_tmp(file)
        result = diarize_file(tmp_path, min_speakers=min_speakers, max_speakers=max_speakers)
        return {
            "ok": True,
            "device": get_device(),
            "pipeline": PIPELINE_NAME,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"diarization 실패: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def ping():
    return {"pong": True, "device": str(get_device()), "pipeline": PIPELINE_NAME}
