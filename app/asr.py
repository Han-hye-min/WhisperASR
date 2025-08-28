import os
from typing import Literal, Optional

from faster_whisper import WhisperModel

# 환경변수로 모델/정밀도 제어 (없으면 기본값)
MODEL_NAME = os.getenv("ASR_MODEL", "base")     # tiny / base / small / medium / large-v3 …
COMPUTE_TYPE = os.getenv("ASR_COMPUTE", "auto") # "int8", "float16", "float32", "auto"
BEAM_SIZE = int(os.getenv("ASR_BEAM", "5"))

_model: Optional[WhisperModel] = None

def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(model_size_or_path=MODEL_NAME, compute_type=COMPUTE_TYPE)
    return _model

def transcribe_file(
    wav_path: str,
    language: str | Literal["auto"] = "auto",
    task: Literal["transcribe", "translate"] = "transcribe",
):
    """
    faster-whisper 호출 래퍼
    - language="auto"면 언어 감지
    - task="transcribe" | "translate"
    """
    model = get_model()
    segments_gen, info = model.transcribe(
        wav_path,
        beam_size=BEAM_SIZE,
        language=None if language == "auto" else language,
        task=task,
        vad_filter=True,
    )

    segments = []
    for i, seg in enumerate(segments_gen):
        segments.append({
            "id": i,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text,
            "avg_logprob": float(seg.avg_logprob) if seg.avg_logprob is not None else None,
            "no_speech_prob": float(seg.no_speech_prob) if seg.no_speech_prob is not None else None,
            "compression_ratio": float(seg.compression_ratio) if seg.compression_ratio is not None else None,
        })

    out = {
        "language": info.language,
        "language_probability": float(info.language_probability) if info.language_probability is not None else None,
        "duration": float(info.duration) if info.duration is not None else None,
        "segments": segments,
        "text": "".join([s["text"] for s in segments]).strip(),
        "model_name": MODEL_NAME,
    }
    return out
