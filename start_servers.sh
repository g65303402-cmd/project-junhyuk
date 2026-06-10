#!/bin/bash
cd "/smhrd2/Hyunjin Team/Hyunjin/project"

echo "TTS 서버 시작 중..."
nohup /root/miniforge3/envs/qwen-tts/bin/python voiceclone/tts_server.py > tts_server.log 2>&1 &
TTS_PID=$!
echo "TTS 서버 PID: $TTS_PID"

echo "RAG 서버 시작 중..."
nohup /usr/bin/python3 rag_server.py > rag_server.log 2>&1 &
RAG_PID=$!
echo "RAG 서버 PID: $RAG_PID"

echo "서버 로딩 대기 중 (30초)..."
sleep 30

echo "메인 서버 시작 중..."
nohup env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /opt/conda/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
MAIN_PID=$!
echo "메인 서버 PID: $MAIN_PID"

echo "모든 서버 시작 완료!"
echo "TTS 서버 로그:  tail -f tts_server.log"
echo "RAG 서버 로그:  tail -f rag_server.log"
echo "메인 서버 로그: tail -f server.log"
