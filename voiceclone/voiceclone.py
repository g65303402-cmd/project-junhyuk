from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from dotenv import load_dotenv
import torch
import soundfile as sf
import os
import uuid
from contextlib import asynccontextmanager

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작할 때
    load_model()
    yield
    # 서버 종료할 때 (필요하면 정리 코드 추가)

app = FastAPI(lifespan=lifespan)

# 모델 & 프롬프트 전역으로 한 번만 로드
model = None
prompt = None

# 요청 스키마
class TTSRequest(BaseModel):
    text: str
    output_path: str = None


def load_model(
    model_path: str = os.getenv("MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    ref_audio: str = os.getenv("REF_AUDIO_PATH"),
    ref_text: str = os.getenv("REF_TEXT")
):
    """모델과 참조 오디오를 한 번만 로드"""
    global model, prompt

    model = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    if ref_audio and ref_text:
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text
        )
    print("모델 로드 완료!")


def generate_tts(text: str, output_path: str = None) -> str:
    """텍스트를 음성으로 변환 후 저장"""
    global model, prompt

    if model is None:
        raise RuntimeError("모델이 로드되지 않았습니다.")

    # output_path 없으면 랜덤 파일명 생성
    if output_path is None:
        os.makedirs("./outputs", exist_ok=True)
        output_path = f"./outputs/{uuid.uuid4()}.wav"

    wavs, sample_rate = model.generate_voice_clone(
        text=text,
        voice_clone_prompt=prompt
    )

    sf.write(output_path, wavs[0], sample_rate)
    print(f"음성 저장 완료: {output_path}")
    return output_path





# TTS 엔드포인트
@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    try:
        output_path = generate_tts(
            text=request.text,
            output_path=request.output_path
        )
        return {"output_path": output_path, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 서버 상태 확인
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}


# # 서버 실행
# uvicorn voiceclone:app --host 0.0.0.0 --port 8000