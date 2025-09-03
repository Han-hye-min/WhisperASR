# Mina_ASR

FastAPI 기반 Whisper + Pyannote ASR 서버  
로컬 PC GPU 환경과 Google Colab GPU 환경에서 모두 실행할 수 있는 음성 인식 및 화자 분리 서버입니다.  
졸업작품 시연 및 테스트 목적에 맞춰 설계되었습니다.

---

## 설치 방법

### 1. 공통 requirements
모든 환경에서 반드시 필요한 최소 패키지 모음입니다.

```bash
pip install -r requirements.txt

2. Colab 환경

Colab GPU 환경에서 실행할 때 사용하는 패키지 확장입니다.
Torch, Torchaudio, HuggingFace Hub, Transformers 등이 포함되어 있습니다.
또한 ngrok을 통해 외부에서 FastAPI 서버에 접근할 수 있도록 구성되어 있습니다.

pip install -r requirements-colab.txt

3. 로컬 환경

개인 PC 환경에서 개발 및 테스트할 때 사용하는 패키지 확장입니다.
.env 관리 및 개발 편의를 위한 패키지가 포함되어 있습니다.
본인의 환경에 맞게 수정하여 사용 가능합니다.

pip install -r requirements-local.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
