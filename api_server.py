"""
박준혁 AI 상담사 — FastAPI 서버
실행: uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch, re, json
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================================
# 설정
# ================================
MODEL_NAME = "./output/exaone30-junhyuk-final"
BASE_MODEL  = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
MAX_NEW_TOKENS = 350
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        "쉽게 털어낼 수 있는 게 아니잖아": "그 마음 이해해",
        "쉽게 흐르는 게 아니잖아":         "충분히 그럴 수 있어",
        "쉽게 흘러가지 않는":              "그게 쉽지 않았겠다",
        "달갑다는 걸":                     "힘들었겠다",
        ", 네.":                           ".",
        ", 네 ":                           " ",
    }
    # 존댓말 후처리
    HONORIFIC_ENDINGS = ["습니다", "니다", "세요", "어요", "아요", "네요", "드릴게요"]
    for ending in HONORIFIC_ENDINGS:
        text = text.replace(ending, "")
    for awkward, replacement in AWKWARD_REPLACEMENTS.items():
        if awkward in text:
            text = text.replace(awkward, replacement)
    return text

def assign_emotion(user_input: str) -> str:
    serious_keywords = [
        "자해", "자살", "죽고 싶", "사라지고 싶", "없어지고 싶",
        "죽고 싶어", "죽어버리고", "살기 싫"
    ]
    if any(k in user_input for k in serious_keywords):
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

# ================================
# 모델 로드 (서버 시작 시 1회)
# ================================
print("모델 로딩 중...")
from peft import PeftModel
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    max_memory={0: "35GiB", "cpu": "10GiB"},
)
model = PeftModel.from_pretrained(base_model, MODEL_NAME)
model.eval()
print("모델 로딩 완료")

# ================================
# FastAPI 앱
# ================================
app = FastAPI(title="박준혁 AI 상담사 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message: str
    emotion: str

def generate_response(user_input: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_input},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    message = postprocess_message(raw)
    return {
        "message": message,
        "emotion": assign_emotion(user_input),
    }

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message가 비어있어")
    result = generate_response(req.message)
    return result