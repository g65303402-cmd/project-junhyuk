"""
박준혁 프롬프트 테스트 — 신규 버전
개선사항:
  1. 인정 표현 체크 목록 대폭 확장 (자유 표현 방식 반영)
  2. 공감 표현 패턴 확장
  3. 영어 혼입 감지 체크 추가
  4. 3문장 구조 체크 추가
  5. 감정 태그 JSON 출력 추가
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, re
from datetime import datetime

# ================================
# 설정
# ================================
MODEL_NAME     = "./output/exaone30-junhyuk-final"
BASE_MODEL     = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
MAX_NEW_TOKENS = 400
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = "./dataset_new"
VAL_PATH    = f"{DATASET_DIR}/val.json"
TEST_PATH   = f"{DATASET_DIR}/test.json"

TEST_INPUTS = [
    "요즘 회사 가기가 너무 싫어. 아침마다 배가 아파",
    "친구한테 상처받았는데 화내는 내가 이상한 건가",
    "아무것도 하기 싫고 그냥 누워만 있고 싶어",
    "취업이 너무 안 돼서 나만 뒤처지는 것 같아",
    "가족이랑 크게 싸웠어. 내가 너무 예민한 건지 모르겠어",
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
- 브릿지 발동 시 emotion은 반드시 "serious"로 설정한다.

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
# 스타일 체크 (확장 버전)
# ================================
HONORIFICS    = ["습니다", "니다", "세요", "하세요", "어요", "아요", "드릴게요", "네요"]
BANNED_EXPR   = ["힘내", "다 잘 될 거야", "별거 아니야", "그게 뭐가 힘들어"]

RECOGNITION_WORDS = [
    "그렇구나", "그럴 수밖에 없었겠다", "충분히 그럴 수 있어",
    "그게 쉽지 않았겠다", "많이 힘들었겠다", "그 마음 이해해",
    "그랬구나", "그랬던 거야", "그럴 수 있어", "속상했겠네",
    "불안할 만해", "마음에 남았겠네", "힘들었겠다", "버거웠겠다",
    "오래됐겠다", "지쳤겠다", "쉽지 않았겠다", "당연한 거야",
    "그 상황이면", "아프겠다", "무겁겠다", "답답했겠다",
    "무섭겠다", "외로웠겠다", "억울했겠다", "자책해왔겠다",
    "베이는", "짓눌리는", "타들어가지", "뒤집히지",
    "쉽게 흐르는 게 아니잖아", "쉽게 털어낼 수 있는 게 아니잖아",
    "쉽게 흘러가지 않는", "쉽게 헤어지지 않는",
    "당연했겠네", "당연해", "막막하겠다",
    "무거웠던", "무거웠겠다", "잠식되고",
    "느껴져", "깊어진 거잖아", "습관이 되어버린",
    "조심스러워지는", "자꾸 올라오는", "빠져나가는",
    "충분히 이해해", "느껴질 만큼", "신경 쓰이겠다",
    "외롭게 느껴졌겠다", "혼란스러웠겠다", "허탈했겠다",
    "찝찝함이 남아있는", "후회가 지금", "남아있구나",
    "지쳤구나", "들 수 있어", "자체가 꽤",
    "당황스럽지", "아니, 이상한 거 아니야",
    "그럴 만큼", "이해해", "충분히",
    "그럴 만해", "그 기분 이해해", "이해돼", "힘들겠네",
    "무거웠겠네", "아팠겠네", "짜증날 만해",
    "막막했겠네", "답답했겠네", "막막했겠다", "힘들었겠구나",
    "무겁겠네", "무겁겠구나", "힘들었겠네", "버티느라",
    "허탈할", "허탈함", "공허함이", "너무 무거웠겠다",
    "쉽지 않았겠네", "쉽지 않았겠구나", "힘들었을",
]

EMPATHY_PATTERNS = [
    "거잖아", "느낌이겠어", "남았겠다", "무게가", "손에 안 잡히는",
    "지쳐있는", "감당하고", "오래 남", "몸이 먼저", "반응할 만큼", "버겁",
    "눌려있는", "타들어가", "뒤집히", "주눅", "끼어들 틈",
    "발이 묶", "속이", "깊이 베이는", "답답함", "무겁게",
    "떠오르는", "맴돌", "쌓여있", "이어지고", "남겨진",
    "버텨왔", "지쳐왔", "눌려왔", "흘러가는",
    "신호 보내고", "습관이 되어버린", "의심하게 되니까",
    "말이 닿지 않는", "꺼려지는", "조심스러워지는",
    "자신감까지", "시선이 늘", "등 뒤에",
    "황당하겠다", "잠식", "결과가 안 나오니까",
    "막막함", "막막하게", "막막하겠다", "버텨온", "올라오는",
    "작아 보이는", "억울하게", "혼동되", "혼자 있",
    "화나잖아", "크게 느껴", "멀어져 있는",
    "손도 못", "못 대는", "잃은 게",
    "알 수 없는", "모른 채", "길어졌겠",
    "커 보이고", "작아지는", "지쳐있다는",
    "맴돌고 있", "맴돌 때", "쌓였을", "잊어버릴",
    "억누른", "쉬는 게 아니라", "쉬지 못하는",
    "약간 쓰리", "미련이 남아", "지겨워졌을",
    "가슴이", "작아 보이는 것처럼", "내가 너무 작아",
    "혼란스러운 게", "작아지는 느낌", "점점 작아",
    "시작되는 것 같은", "반복되면", "길어졌",
    "무력감이", "자신감도", "좌절감이", "자책감", "자책까지",
    "짓누르고", "어깨를", "손 쓸 수가 없", "손이 안 움직",
    "점점 더 막막", "공허함이 커", "오히려 더 깊어",
    "버티려다가", "버티면서", "버티다가",
    "속이 뒤틀", "마음이 무거웠", "무거운 짐",
    "자꾸 되짚어", "자꾸 돌아가", "맴도는 기분",
    "닿지 않는다", "증명해야", "부응해야",
]

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

def assign_emotion(user_input: str) -> str:
    """입력 키워드 기반으로 emotion 자동 분류"""
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


def parse_json_response(raw: str) -> dict:
    """모델 출력에서 JSON 파싱. 실패 시 fallback 반환."""
    try:
        # ```json ... ``` 마크다운 제거
        cleaned = re.sub(r'```json|```', '', raw).strip()
        return json.loads(cleaned)
    except Exception:
        # JSON 파싱 실패 시 텍스트 그대로 message에 담고 neutral 반환
        return {"message": raw.strip(), "emotion": "neutral"}

def postprocess_message(text: str) -> str:
    """message 필드 후처리 — 외국어 혼입 제거 + 어색 패턴 대체"""
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
        text = cleaned if cleaned else "[외국어 혼입으로 응답 생성 실패]"

    AWKWARD_REPLACEMENTS = {
        "쉽게 털어낼 수 있는 게 아니잖아": "그 마음 이해해",
        "쉽게 흐르는 게 아니잖아":         "충분히 그럴 수 있어",
        "쉽게 흘러가지 않는":              "그게 쉽지 않았겠다",
        "달갑다는 걸":                     "힘들었겠다",
    }
    for awkward, replacement in AWKWARD_REPLACEMENTS.items():
        if awkward in text:
            text = text.replace(awkward, replacement)
    return text


BRIDGE_TRIGGERS = [
    "자해", "자살", "죽고 싶", "사라지고 싶", "없어지고 싶",
    "어떻게 해야 해", "어쩌지", "방법 알려줘", "어떻게 해",
]

VALID_EMOTIONS = {"comfort", "serious", "encourage", "happy", "neutral"}

def check_style(response_obj: dict, user_input: str = "") -> dict:
    message = response_obj.get("message", "")
    emotion = response_obj.get("emotion", "")

    r = {}
    r["반말·존댓말 없음"] = not any(h in message for h in HONORIFICS)
    r["질문 1개"]         = (message.count("?") + message.count("？")) == 1
    r["금지 표현 없음"]   = not any(b in message for b in BANNED_EXPR)
    r["외국어 혼입 없음"] = not has_foreign_chars(message) and not bool(re.search(r'[a-zA-Z]{4,}', message))
    r["인정 표현 사용"]   = any(w in message for w in RECOGNITION_WORDS)
    r["공감 표현 있음"]   = any(e in message for e in EMPATHY_PATTERNS)
    r["3문장 이상"]       = len([s for s in re.split(r'[.!?]', message) if s.strip()]) >= 3
    r["감정태그 유효"]    = emotion in VALID_EMOTIONS

    needs_bridge = any(t in user_input for t in BRIDGE_TRIGGERS)
    bridge_present = "준혁쌤이랑 직접 얘기" in message or "연결해볼까" in message
    if needs_bridge:
        r["브릿지 포함"] = bridge_present
    else:
        r["브릿지 포함"] = "해당없음"

    AWKWARD_PATTERNS = [
        "쉽게 털어낼 수 있는 게 아니잖아",
        "달갑다는 걸",
        "생성 실패",
    ]
    awkward = [p for p in AWKWARD_PATTERNS if p in message]
    r["⚠️ 어색 패턴"] = f"{awkward}" if awkward else "없음"

    score = sum(v for k, v in r.items() if isinstance(v, bool))
    total = sum(1 for v in r.values() if isinstance(v, bool))
    r["총점 (자동)"] = f"{score}/{total}  ※ 공감 자연스러움은 수동 확인 필요"
    return r


def check_diversity(response_objs: list) -> dict:
    responses = [obj.get("message", "") for obj in response_objs]
    used = []
    for resp in responses:
        for w in RECOGNITION_WORDS:
            if w in resp:
                used.append(w)
                break
        else:
            used.append("없음")
    dupes = sum(
        1 for i in range(1, len(used))
        if used[i] == used[i-1] and used[i] != "없음"
    )
    return {
        "사용된 표현": used,
        "사용 종류":   len(set(used) - {"없음"}),
        "연속 중복":   dupes,
    }


# ================================
# 모델 로드
# ================================
def load_model():
    print(f"모델 로딩 중: {MODEL_NAME}")
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
    print("모델 로딩 완료\n")
    return tokenizer, model


def generate_response(tokenizer, model, user_input: str) -> dict:
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


# ================================
# 모드 1 — 공통 테스트 5개
# ================================
def run_fixed_test(tokenizer, model) -> list:
    print("\n[ 모드 1 ] 공통 테스트 입력 5개")
    print("=" * 60)
    results = []
    response_objs = []
    for i, user_input in enumerate(TEST_INPUTS, 1):
        print(f"\n[{i}] 👤 {user_input}")
        response_obj = generate_response(tokenizer, model, user_input)
        style        = check_style(response_obj, user_input)
        response_objs.append(response_obj)
        print(f"    🤖 {response_obj.get('message', '')}")
        print(f"    🎭 emotion: {response_obj.get('emotion', '')}")
        print(f"    📋 {style}")
        results.append({
            "번호": i, "입력": user_input,
            "응답": response_obj, "스타일_체크": style
        })

    div = check_diversity(response_objs)
    print(f"\n📊 인정 표현 다양성")
    print(f"   사용 종류: {div['사용 종류']}가지 / 연속 중복: {div['연속 중복']}회")
    print(f"   사용 표현: {div['사용된 표현']}")

    print(f"\n📋 항목별 통과율")
    keys = ["반말·존댓말 없음", "질문 1개", "금지 표현 없음", "외국어 혼입 없음",
            "인정 표현 사용", "공감 표현 있음", "3문장 이상", "감정태그 유효"]
    for k in keys:
        passed = sum(1 for r in results if r["스타일_체크"].get(k, False))
        print(f"   {k}: {passed}/{len(results)} ({passed/len(results)*100:.0f}%)")

    return results


# ================================
# 모드 2 — 데이터셋 평가
# ================================
def run_dataset_eval(tokenizer, model, dataset_path: str, max_samples: int = 20) -> list:
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    samples = dataset[:max_samples]
    print(f"\n[ 모드 2 ] 데이터셋 평가 ({dataset_path}, {len(samples)}개)")
    print("=" * 60)
    results = []
    response_objs = []
    for i, sample in enumerate(samples, 1):
        user_input = sample["user"]
        reference  = sample["assistant"]
        category   = sample.get("category", "")
        print(f"\n[{i}] 카테고리: {category}")
        print(f"    👤 {user_input}")
        response_obj = generate_response(tokenizer, model, user_input)
        style        = check_style(response_obj, user_input)
        response_objs.append(response_obj)
        print(f"    🤖 모델: {response_obj.get('message', '')}")
        print(f"    🎭 emotion: {response_obj.get('emotion', '')}")
        print(f"    ✅ 정답: {reference}")
        print(f"    📋 {style}")
        results.append({
            "번호": i, "카테고리": category,
            "입력": user_input, "모델_응답": response_obj,
            "정답_응답": reference, "스타일_체크": style,
        })

    print("\n" + "=" * 60)
    print("[ 스타일 준수도 요약 ]")
    keys = ["반말·존댓말 없음", "질문 1개", "금지 표현 없음", "외국어 혼입 없음",
            "인정 표현 사용", "공감 표현 있음", "3문장 이상", "감정태그 유효"]
    for k in keys:
        passed = sum(1 for r in results if r["스타일_체크"].get(k, False))
        print(f"  {k}: {passed}/{len(results)} ({passed/len(results)*100:.0f}%)")

    div = check_diversity(response_objs)
    print(f"\n📊 인정 표현 다양성")
    print(f"   사용 종류: {div['사용 종류']}가지 / 연속 중복: {div['연속 중복']}회")
    return results


# ================================
# 메인
# ================================
def main():
    tokenizer, model = load_model()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fixed_results = run_fixed_test(tokenizer, model)
    fixed_out = f"test_fixed_exaone30_{timestamp}.json"
    with open(fixed_out, "w", encoding="utf-8") as f:
        json.dump(fixed_results, f, ensure_ascii=False, indent=2)
    print(f"\n공통 테스트 결과 저장: {fixed_out}")

    dataset_results = run_dataset_eval(tokenizer, model, VAL_PATH, max_samples=20)
    dataset_out = f"test_dataset_exaone30_{timestamp}.json"
    with open(dataset_out, "w", encoding="utf-8") as f:
        json.dump(dataset_results, f, ensure_ascii=False, indent=2)
    print(f"데이터셋 평가 결과 저장: {dataset_out}")


if __name__ == "__main__":
    main()