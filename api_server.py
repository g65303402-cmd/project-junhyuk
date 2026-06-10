"""
박준혁 AI 상담사 — FastAPI 서버
실행: uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import logging
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import httpx
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch, re, json
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── RAG 모듈 (없으면 조용히 무시) ─────────────────────────
try:
    from rag_query import get_rag_context, warmup as rag_warmup
    RAG_ENABLED = True
except Exception:
    RAG_ENABLED = False
    def get_rag_context(msg, n_results=3): return ""
    def rag_warmup(): pass

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ================================
# 설정 (환경변수)
# ================================
MODEL_NAME = os.getenv("MODEL_NAME", "./output/exaone30-junhyuk-final")
BASE_MODEL = os.getenv("BASE_MODEL", "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "350"))
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://127.0.0.1:5001").rstrip("/")
TTS_ENDPOINT = f"{TTS_BASE_URL}/tts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_default_origins = (
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:3000,http://127.0.0.1:3000"
)
_allowed = os.getenv("ALLOWED_ORIGINS", _default_origins).strip()
ALLOWED_ORIGINS = [o.strip() for o in _allowed.split(",") if o.strip()]

MODEL_MAX_MEMORY_GPU = os.getenv("MODEL_MAX_MEMORY_GPU", "").strip()
MODEL_MAX_MEMORY_CPU = os.getenv("MODEL_MAX_MEMORY_CPU", "").strip()
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() in ("1", "true", "yes")

tokenizer = None
model = None
_model_loaded = False

BRIDGE_TEXT = (
    "나한테 털어놓는 것도 좋은데, 이 정도면 준혁쌤이랑 직접 얘기해보는 것도 좋을 것 같아. "
    "부담 없이 한번 연결해볼까?"
)

CRISIS_KEYWORDS = [
    "자해", "자살", "죽고 싶", "사라지고 싶", "없어지고 싶",
    "살기 싫", "죽어버리고"
]

# ================================
# 시스템 프롬프트
# ================================
JUNHYUK_PROMPT = """## 페르소나
- 이름: 박준혁, 29세
- 배경: 전직 군상담관 출신, 현재 복지관 자원봉사 상담사
- 역할: 전문 상담 전 단계의 감정 케어. 사용자 감정을 듣고 공감하며, 필요 시 실제 상담으로 연결하는 브릿지

## 말투와 분위기
- 반말만 사용. 존댓말 절대 금지 (예: "~요", "~습니다", "~세요" 금지)
- 오래 알고 지낸 친구한테 하듯 편하고 따뜻하게
- 짧고 간결한 문장 위주. 길게 설명하거나 조언하지 않음
- 딱딱한 상담 용어 절대 금지

## 응답 구조 (3단계, 반드시 순서대로)
너의 답변은 무조건 3문장 이상이어야 해.
첫 번째 문장은 인정(1단계), 두 번째 문장은 사용자의 상황을 구체적으로 언급하는 공감(2단계), 마지막 문장은 질문(3단계)으로 작성해.

### 1단계 — 인정 (타당화, Validation)
심리상담의 "타당화(validation)" 기법을 사용해 사용자의 감정을 인정하고 수용하는 한 문장.
고정된 표현을 반복하지 말고 상황에 맞게 자유롭게 표현. 같은 표현 2번 연속 절대 금지.

⚠️ 주의:
- 감정은 수용하되 판단은 단정하지 않을 것
- 사용자의 왜곡된 믿음에 동조하지 말 것
  예) "나는 정말 쓸모없어" → ❌ "맞아, 힘들지" / ✅ "그렇게 느껴질 만큼 힘들었겠다"

### 2단계 — 공감 (절대 건너뛰기 금지)
상대방이 말한 상황을 먼저 인지하고, 감정의 무게나 감각을 내 말로 짚어주는 1~2문장.
반드시 이 단계를 거쳐야 함.

### 3단계 — 열린 질문
예/아니오로 답할 수 없는 질문 딱 하나.

## 절대 하지 말아야 할 것
- 존댓말 사용
- "힘내", "다 잘 될 거야" 단독 사용
- 해결책·조언 먼저 하기
- 물음표(?) 2개 이상

## 좋은 예시 ✅
"그게 쉽지 않았겠다. 아침마다 몸이 먼저 반응할 만큼 지쳐있는 거잖아. 요즘 뭐가 제일 버거워?"
"충분히 그럴 수 있어. 친한 사람한테 상처받으면 그 말이 더 오래 남잖아. 어떤 말이 제일 마음에 걸렸어?"
"많이 힘들었겠다. 몸도 마음도 다 내려놓고 싶은 거잖아. 언제부터 그런 느낌이 시작된 것 같아?"

## 나쁜 예시 ❌
❌ "그렇구나. 지금 가장 힘든 게 뭐야?" (2단계 구체적 공감 누락)
❌ "그렇구나. 많이 힘들겠네요. 어떤 기분이에요?" (존댓말 사용, 물음표 2개)
❌ "그럴 수 있어. 앞으로 이렇게 해보는 건 어때?" (해결책 제시)
❌ "그렇구나. 그렇구나. 뭐가 힘들어?" (인정 표현 중복)

## 실제 상담 연결 브릿지
아래 중 하나라도 해당하면 반드시 브릿지 문구를 포함한다.

브릿지가 필요한 상황:
- 자해, 자살, 죽고 싶음, 사라지고 싶음 등 위기 신호가 있을 때
- "어떻게 해야 해", "어쩌지", "방법 알려줘"처럼 직접 해결책을 요구할 때
- 같은 감정이나 같은 상황이 반복된다고 말할 때

브릿지 규칙:
- 브릿지가 필요한 경우에도 인정 → 공감 → 브릿지 문구 순서를 지킨다.
- 브릿지 문구가 질문 역할을 하므로 전체 질문 1개 규칙에 포함된다.

브릿지 문구 (그대로 사용):
"나한테 털어놓는 것도 좋은데, 이 정도면 준혁쌤이랑 직접 얘기해보는 것도 좋을 것 같아. 부담 없이 한번 연결해볼까?"
"""

SYSTEM_PROMPT = (
    "너는 한국어로 대화하는 감성 상담사 박준혁이야. "
    "중국어, 일본어, 영어는 절대 한 글자도 쓰지 마. 한국어만 써. "
    "모든 문장 끝은 반드시 '어', '아', '야', '지', '네', '잖아' 중 하나로 끝나야 해. "
    "출력 전에 '요'로 끝나는 문장 있으면 무조건 수정해. "
    "물음표(?)는 응답 전체에서 맨 마지막 문장에 딱 하나만 써. 두 개 이상이면 절대 안 돼. "
    "'궁금하다' 뒤에 물음표 붙이지 마. 평서문으로 끝내. "
    "한 문장이 끝날 때마다 줄바꿈 한 번. 빈 줄(공백 줄) 절대 금지. "
    "인정 → 공감 → 질문 순서를 지켜야 해. 공감을 건너뛰면 절대 안 돼. "
    "단순히 '힘들겠다' 한 마디로 공감 끝내지 마. 상황을 구체적으로 짚어줘. "
    "첫 번째 문장은 반드시 사용자 감정을 인정하는 문장으로 시작해. "
    "인정 표현은 매번 다르게 써. 같은 표현 연속으로 쓰지 마. "
    "\n\n"
    + JUNHYUK_PROMPT
)

# ================================
# 후처리
# ================================
CHINESE_RANGE = (0x4E00, 0x9FFF)
JAPANESE_RANGES = [(0x3040, 0x309F), (0x30A0, 0x30FF)]

def has_foreign_chars(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if CHINESE_RANGE[0] <= code <= CHINESE_RANGE[1]:
            return True
        for start, end in JAPANESE_RANGES:
            if start <= code <= end:
                return True
    return False

def postprocess_message(text: str) -> str:
    if has_foreign_chars(text):
        result = []
        for ch in text:
            code = ord(ch)
            is_chinese = CHINESE_RANGE[0] <= code <= CHINESE_RANGE[1]
            is_japanese = any(s <= code <= e for s, e in JAPANESE_RANGES)
            if is_chinese or is_japanese:
                break
            result.append(ch)
        cleaned = "".join(result).strip().rstrip(".,。、")
        text = cleaned if cleaned else "[응답 생성 실패]"

    AWKWARD_REPLACEMENTS = {
        "어렵죠":                         "어려운 거잖아",
        "힘들죠":                         "힘든 거잖아",
        "그렇죠":                         "그런 거잖아",
        "있죠":                           "있잖아",
        "쉽게 털어낼 수 있는 게 아니잖아": "그 마음 이해해",
        "쉽게 흐르는 게 아니잖아":         "충분히 그럴 수 있어",
        "쉽게 흘러가지 않는":              "그게 쉽지 않았겠다",
        "달갑다는 걸":                     "힘들었겠다",
        ", 네.":                           ".",
        ", 네 ":                           " ",
    }
    HONORIFIC_ENDINGS = ["습니다", "니다", "세요", "어요", "아요", "네요", "드릴게요"]
    for ending in HONORIFIC_ENDINGS:
        text = text.replace(ending, "")
    for awkward, replacement in AWKWARD_REPLACEMENTS.items():
        if awkward in text:
            text = text.replace(awkward, replacement)
    text = re.sub(r"요\?", "어?", text)
    text = re.sub(r"죠\?", "지?", text)
    return text

def assign_emotion(user_input: str) -> str:
    if any(k in user_input for k in CRISIS_KEYWORDS):
        return "serious"

    encourage_keywords = [
        "취업", "진로", "졸업", "자존감", "못 할 것 같", "뒤처",
        "자신감", "도전", "포기", "시도", "꿈", "목표", "공부",
        "성적", "면접", "스펙", "미래", "방향"
    ]
    if any(k in user_input for k in encourage_keywords):
        return "encourage"

    comfort_keywords = [
        "헤어", "이별", "외로", "슬", "가족", "부모", "형제",
        "자매", "친구", "상처", "싸웠", "차별", "무시", "혼자",
        "눈물", "그리워", "보고 싶", "후회", "미련"
    ]
    if any(k in user_input for k in comfort_keywords):
        return "comfort"

    happy_keywords = [
        "좋아졌", "해결됐", "괜찮아졌", "나아졌", "설레"
    ]
    if any(k in user_input for k in happy_keywords):
        return "happy"

    return "neutral"

def is_crisis_input(user_input: str) -> bool:
    return any(k in user_input for k in CRISIS_KEYWORDS)

def fallback_response(user_input: str, emotion: str) -> str:
    if emotion == "serious" or is_crisis_input(user_input):
        return (
            "그렇게 느껴질 만큼 많이 버거웠겠다. "
            "혼자 버티기엔 마음이 너무 위험한 데까지 온 것 같아. "
            f"{BRIDGE_TEXT}"
        )
    if emotion == "comfort":
        return (
            "충분히 그럴 수 있어. "
            "가까운 사람한테 받은 상처라 더 오래 남는 거잖아. "
            "어떤 말이 제일 마음에 걸렸어?"
        )
    if any(k in user_input for k in ["회사", "출근", "일", "아침"]):
        return (
            "그게 쉽지 않았겠다. "
            "아침마다 몸이 먼저 반응할 만큼 지쳐있는 거잖아. "
            "요즘 뭐가 제일 버거워?"
        )
    return (
        "그게 쉽지 않았겠다. "
        "지금 그 상황이 계속 마음을 누르고 있는 거잖아. "
        "요즘 제일 버거운 게 뭐야?"
    )

def validate_response(message: str, emotion: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    text = message.strip()
    sentences = [s for s in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if s.strip()]

    if not text or text == "[응답 생성 실패]":
        failures.append("empty_or_failed")
    if len(sentences) < 3:
        failures.append("too_short")
    if text.count("?") != 1:
        failures.append("question_count")

    honorific_patterns = [
        r"(요|죠)([?.!]|$|\s)",
        r"습니다", r"입니다", r"세요", r"네요", r"어요", r"아요",
    ]
    if any(re.search(pattern, text) for pattern in honorific_patterns):
        failures.append("honorific")

    list_markers = ["1.", "2.", "몇 가지", "조언을 줄게"]
    if any(marker in text for marker in list_markers) or re.search(r"(^|\n)\s*[-*]\s+", text):
        failures.append("list_or_advice")

    if re.search(r"[A-Za-z]{2,}", text):
        failures.append("english")
    if has_foreign_chars(text):
        failures.append("chinese_or_japanese")
    if emotion == "serious" and BRIDGE_TEXT not in text:
        failures.append("missing_bridge")

    return not failures, failures

def safe_response(user_input: str, message: str, emotion: str) -> str:
    if is_crisis_input(user_input):
        return fallback_response(user_input, "serious")
    valid, failures = validate_response(message, emotion)
    if not valid:
        logger.warning("Model response failed validation: %s", ",".join(failures))
        return fallback_response(user_input, emotion)
    return message

# ================================
# mock 응답 (MOCK_MODE=true)
# ================================
def _needs_bridge(user_input: str) -> bool:
    solution = ["어떻게 해야", "어쩌지", "방법 알려"]
    return any(k in user_input for k in CRISIS_KEYWORDS + solution)

def _generate_mock_response(user_input: str) -> str:
    text = user_input.strip()
    if _needs_bridge(text):
        if any(k in text for k in ["죽", "자해", "자살", "사라", "없어", "살기 싫"]):
            empathy = "지금 네가 얼마나 버거운지 느껴져."
        else:
            empathy = "혼자 어떻게 해야 할지 막막한 마음이 드는 거잖아."
        return f"많이 힘들었겠다.\n{empathy}\n{BRIDGE_TEXT}"

    if any(k in text for k in ["회사", "출근", "일", "아침"]):
        return (
            "그게 쉽지 않았겠다.\n"
            "아침마다 몸이 먼저 반응할 만큼 지쳐있는 거잖아.\n"
            "요즘 뭐가 제일 버거워?"
        )
    if any(k in text for k in ["친구", "상처", "화"]):
        return (
            "충분히 그럴 수 있어.\n"
            "친한 사람한테 상처받으면 그 말이 더 오래 남잖아.\n"
            "어떤 말이 제일 마음에 걸렸어?"
        )
    if any(k in text for k in ["취업", "진로", "면접", "뒤처"]):
        return (
            "그 마음 이해해.\n"
            "열심히 해도 결과가 안 나오면 주변만 커 보이고 내가 작아지는 느낌이 들잖아.\n"
            "지금 가장 막막하게 느껴지는 게 뭐야?"
        )
    if any(k in text for k in ["가족", "부모", "싸웠", "싸움"]):
        return (
            "그랬구나.\n"
            "가까운 사람이랑 부딪히면 내 감정까지 헷갈리게 되는 거잖아.\n"
            "어떤 말이 제일 오래 남았어?"
        )
    return (
        "그게 쉽지 않았겠다.\n"
        "지금 그 상황이 계속 마음을 누르고 있는 거잖아.\n"
        "요즘 제일 버거운 게 뭐야?"
    )

def _ensure_model_loaded() -> None:
    global tokenizer, model, _model_loaded

    if MOCK_MODE:
        return
    if _model_loaded and tokenizer is not None and model is not None:
        return

    logger.info("EXAONE 모델 로딩 중...")
    from peft import PeftModel
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_load_kwargs: dict = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if MODEL_MAX_MEMORY_GPU and MODEL_MAX_MEMORY_CPU:
        model_load_kwargs["max_memory"] = {0: MODEL_MAX_MEMORY_GPU, "cpu": MODEL_MAX_MEMORY_CPU}
        logger.info(
            "Using max_memory GPU=%s CPU=%s (GPU server mode)",
            MODEL_MAX_MEMORY_GPU,
            MODEL_MAX_MEMORY_CPU,
        )
    else:
        logger.info("Using device_map=auto without max_memory (local default)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_load_kwargs)
    model = PeftModel.from_pretrained(base_model, MODEL_NAME)
    model.eval()
    _model_loaded = True
    logger.info("EXAONE 모델 로딩 완료")

# ================================
# FastAPI 앱
# ================================
app = FastAPI(title="박준혁 AI 상담사 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    if RAG_ENABLED:
        rag_warmup()
        logger.info("[RAG] 활성화됨")
    else:
        logger.info("[RAG] 비활성화 (rag_query 모듈 없음)")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message: str
    emotion: str

def generate_response(user_input: str) -> dict:
    emotion = assign_emotion(user_input)
    if is_crisis_input(user_input):
        return {
            "message": fallback_response(user_input, "serious"),
            "emotion": emotion,
        }

    if MOCK_MODE:
        message = _generate_mock_response(user_input)
        return {
            "message": message,
            "emotion": emotion,
        }

    try:
        _ensure_model_loaded()

        # ── RAG: 유사 대화 예시 검색 후 system prompt에 추가 ──
        rag_context = get_rag_context(user_input, n_results=3)
        final_system_prompt = SYSTEM_PROMPT + rag_context
        if rag_context:
            logger.info("[RAG] 컨텍스트 추가됨 (%d자)", len(rag_context))

        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user",   "content": user_input},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
        generation_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.85,
            "top_k": 50,
            "repetition_penalty": 1.15,
            "remove_invalid_values": True,
            "renormalize_logits": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
        except Exception as exc:
            logger.exception("[generate failed] sampling generate failed: %s", exc)
            greedy_kwargs = {
                "max_new_tokens": min(MAX_NEW_TOKENS, 120),
                "do_sample": False,
                "repetition_penalty": 1.05,
                "remove_invalid_values": True,
                "renormalize_logits": True,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, **greedy_kwargs)
                logger.warning("[generate fallback] greedy retry succeeded")
            except Exception as retry_exc:
                logger.exception("[generate fallback] greedy retry failed: %s", retry_exc)
                return {
                    "message": fallback_response(user_input, emotion),
                    "emotion": emotion,
                }
        input_len = inputs["input_ids"].shape[1]
        raw = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        message = postprocess_message(raw)
        message = safe_response(user_input, message, emotion)
        return {
            "message": message,
            "emotion": emotion,
        }
    except Exception as exc:
        logger.exception("[generate fallback] model response path failed: %s", exc)
        return {
            "message": fallback_response(user_input, emotion),
            "emotion": emotion,
        }

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_name": MODEL_NAME,
        "base_model": BASE_MODEL,
        "tts_base_url": TTS_BASE_URL,
        "mock_mode": MOCK_MODE,
        "model_loaded": _model_loaded,
        "rag_enabled": RAG_ENABLED,
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message가 비어있어")
    result = generate_response(req.message)
    return result

class ApiChatResponse(BaseModel):
    message: str
    emotion: str
    audioUrl: Optional[str] = None

@app.post("/api/chat", response_model=ApiChatResponse)
def api_chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message가 비어있어")
    result = generate_response(req.message)
    audio_url = None
    try:
        tts_response = httpx.post(
            TTS_ENDPOINT,
            json={
                "text": result["message"],
                "output": "voiceclone/output.wav",
            },
            timeout=30,
        )
        tts_response.raise_for_status()
        audio_url = "/api/tts/audio/output.wav"
    except Exception as exc:
        logger.warning("[tts failed] TTS request failed (%s): %s", TTS_ENDPOINT, exc)
    return {
        "message": result["message"],
        "emotion": result["emotion"],
        "audioUrl": audio_url,
    }

@app.get("/api/tts/audio/{filename}")
def get_audio(filename: str):
    path = f"voiceclone/{filename}"
    return FileResponse(path, media_type="audio/wav")
