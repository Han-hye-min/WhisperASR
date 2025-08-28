import os
import shutil
import subprocess
import tempfile
from typing import Optional
import soundfile as sf



FFPROBE = "ffprobe"
FFMPEG = "ffmpeg"

def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = proc.communicate()
    return proc.returncode, out, err

def ensure_tmp_copy(upload_path: str | None, file_bytes: bytes | None, suffix: str) -> str:
    """
    - UploadFile을 디스크에 저장해 둔 경로(upload_path) or 메모리 bytes(file_bytes)를 받아
      임시파일 경로를 만들어 반환.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    if upload_path:
        shutil.copyfile(upload_path, tmp_path)
    else:
        with open(tmp_path, "wb") as f:
            f.write(file_bytes or b"")
    return tmp_path

def get_media_duration_sec(path: str) -> Optional[float]:
    """
    ffprobe로 길이(초)를 읽어옴. 실패 시 None.
    """
    cmd = [
        FFPROBE,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    code, out, _ = run_cmd(cmd)
    if code != 0:
        return None
    try:
        return float(out.strip())
    except Exception:
        return None

def transcode_to_wav_mono16k(src_path: str) -> tuple[bool, str, str]:
    """
    Whisper 친화 포맷(PCM 16kHz mono wav)으로 변환.
    반환: (성공여부, 변환된_wav_path, 로그)
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        FFMPEG,
        "-y",
        "-i", src_path,
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        wav_path,
    ]
    code, out, err = run_cmd(cmd)
    ok = (code == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0)
    return ok, wav_path, (out + err)

def is_silent(wav_path, threshold_db=-40.0):
    data, samplerate = sf.read(wav_path)
    # RMS (Root Mean Square) 계산
    rms = (data**2).mean()**0.5
    if rms == 0:
        return True
    import math
    db = 20 * math.log10(rms)
    return db < threshold_db