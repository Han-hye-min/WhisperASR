from typing import List, Dict


def build_speaker_map(labels: List[str]) -> Dict[str, str]:
    """
    pyannote 기본 라벨(SPEAKER_00, SPEAKER_01 ...)을
    S0, S1, S2 ...로 매핑
    """
    uniq = []
    for lb in labels:
        if lb not in uniq:
            uniq.append(lb)
    return {orig: f"S{i}" for i, orig in enumerate(uniq)}


def to_rttm(segments: List[dict]) -> str:
    """
    JSON segments → RTTM 포맷 문자열 변환
    RTTM: "SPEAKER <uri> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>"
    """
    lines = []
    for seg in segments:
        dur = max(0.0, seg["end"] - seg["start"])
        lines.append(
            f"SPEAKER unknown 1 {seg['start']:.3f} {dur:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>"
        )
    return "\n".join(lines)
