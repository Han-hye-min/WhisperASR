import os
import tempfile
import shutil
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

# 성능/충돌 회피 옵션 (네 기존 서버에도 비슷한 설정 있던 것 같아)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---- pyannote 로딩 (서버 부팅 시 1회) ----
from pyannote.audio import Pipeline
import torch

def get_device() -> str:
    """CUDA 가능하면 cuda, 아니면 cpu"""
    return "cuda" if torch.cuda.is_available() else "cpu"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN 환경변수가 설정되어야 합니다. (Hugging Face read token)")

# 모델명은 필요에 따라 교체 가능: "pyannote/speaker-diarization" or "pyannote/speaker-diarization-3.1"
PIPELINE_NAME = "pyannote/speaker-diarization-3.1"

print(f"[pyannote] Loading pipeline: {PIPELINE_NAME}")
_pipeline = Pipeline.from_pretrained(PIPELINE_NAME, use_auth_token=HF_TOKEN)
_pipeline.to(get_device())
print(f"[pyannote] Loaded to device: {get_device()}")

# ---- FastAPI ----
app = FastAPI(title="Mina ASR + Diarization API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 좁혀 설정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 유틸 =====
def save_upload_to_tmp(f: UploadFile) -> str:
    """UploadFile을 임시 경로에 저장하고 파일 경로 반환"""
    suffix = os.path.splitext(f.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(prefix="mina_diar_", suffix=suffix)
    os.close(fd)
    with open(path, "wb") as out:
        shutil.copyfileobj(f.file, out)
    return path

def build_speaker_map(labels: List[str]) -> Dict[str, str]:
    """
    pyannote 기본 라벨(ex. 'SPEAKER_00', 'SPEAKER_01' ...)을
    S0, S1, S2 ...로 깔끔하게 매핑
    """
    uniq = []
    for lb in labels:
        if lb not in uniq:
            uniq.append(lb)
    return {orig: f"S{i}" for i, orig in enumerate(uniq)}

def diarize_file(audio_path: str,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None) -> Dict[str, Any]:
    """
    pyannote diarization 실행 후 결과를 JSON friendly 형태로 변환
    """
    kwargs = {}
    # min/max_speakers는 파이프라인 버전에 따라 지원/무시될 수 있음
    if min_speakers is not None:
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        kwargs["max_speakers"] = int(max_speakers)

    diarization = _pipeline(audio_path, **kwargs)

    # 라벨 정규화 매핑
    raw_labels = [label for _, _, label in diarization.itertracks(yield_label=True)]
    spk_map = build_speaker_map(raw_labels)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
            "speaker": spk_map[speaker],
        })

    # RTTM 텍스트도 함께 만들어주면 타툴 연계에 편함
    rttm_lines = []
    # RTTM 형식: SPEAKER <uri> 1 <start> <dur> <ortho> <stype> <name> <conf> <slat>
    # 여기서는 간단히 uri=unknown, 기타 필드는 기본값
    for seg in segments:
        dur = max(0.0, seg["end"] - seg["start"])
        rttm_lines.append(
            f"SPEAKER unknown 1 {seg['start']:.3f} {dur:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>"
        )

    return {
        "num_speakers": len(set([s["speaker"] for s in segments])),
        "segments": segments,        # [{start, end, speaker}, ...]
        "rttm": "\n".join(rttm_lines)
    }

# ===== 라우트 =====

@app.post("/diarize")
async def diarize_endpoint(
    file: UploadFile = File(..., description="오디오 파일 (wav/mp3/m4a 등 ffmpeg가 읽을 수 있으면 OK)"),
    min_speakers: Optional[int] = Form(None, description="예상 최소 화자 수(선택)"),
    max_speakers: Optional[int] = Form(None, description="예상 최대 화자 수(선택)")
):
    """
    화자분리 엔드포인트
    - 업로드된 오디오를 임시 저장 후 pyannote 파이프라인 실행
    - 구간(start/end)과 speaker 라벨(S0,S1,...)을 JSON으로 반환
    - 필요시 min/max_speakers 힌트를 제공할 수 있음
    """
    if file.content_type is None:
        raise HTTPException(status_code=400, detail="파일 content-type이 비어 있습니다.")

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
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# 헬스체크
def ping():
    return {"pong": True, "device": get_device(), "pipeline": PIPELINE_NAME}
