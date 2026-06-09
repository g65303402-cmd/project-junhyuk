## 설치 방법

### 1. 가상환경 생성
conda env create -f environment.yml  \b
conda activate {파이썬버전 3.10.20 가상환경 이름}

### 2. 라이브러리 설치
pip install -r requirements.txt

### 3. 모델 다운로드
hf download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz  \b
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base

## Qwen3-TTS 소스 설치
git clone https://github.com/QwenLM/Qwen3-TTS.git \b
cd Qwen3-TTS  \b
pip install -e .

### 4. 실행
python tts.py
