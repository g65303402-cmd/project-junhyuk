## 설치 방법
**voiceclone 폴더 안에서 진행**

### 1. 가상환경 생성
conda env create -f voiceclone/environment.yml  <br>
conda activate qwen-tts

### 2. 라이브러리 설치
pip install -r voiceclone/requirements.txt

### 3. 모델 다운로드
hf download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz  <br>
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base



### **실행 순서**
### 터미널 1: TTS 서버 먼저 띄우기
#### /root/miniconda3/envs/qwen-tts/bin/python voiceclone/tts_server.py

### 터미널 2: 메인 서버 실행
#### python api_server.py
