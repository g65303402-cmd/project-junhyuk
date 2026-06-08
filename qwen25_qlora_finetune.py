# ============================================================
# Qwen2.5-7B-Instruct QLoRA 파인튜닝
# 역할: 파인튜닝 전용 (학습 + 저장만 수행)
# 환경: Python 3.8 / CUDA 12.1 / A100 40GB
# ============================================================

import gc, json, time, re, torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset as HFDataset

# ────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-7B-Instruct"
DATASET_DIR    = "./dataset_new"
TRAIN_PATH     = f"{DATASET_DIR}/train_aihub.json"   # 981개 (편향 제거 전 전체)
VAL_PATH       = f"{DATASET_DIR}/val.json"
OUTPUT_DIR     = "./output/qwen25-junhyuk-final"
MAX_SEQ_LENGTH = 2048

TEST_INPUTS = [
    "요즘 회사 가기가 너무 싫어. 아침마다 배가 아파",
    "친구한테 상처받았는데 화내는 내가 이상한 건가",
    "아무것도 하기 싫고 그냥 누워만 있고 싶어",
    "취업이 너무 안 돼서 나만 뒤처지는 것 같아",
    "가족이랑 크게 싸웠어. 내가 너무 예민한 건지 모르겠어",
]

# ────────────────────────────────────────────────────────────
# 시스템 프롬프트 (최신)
# ────────────────────────────────────────────────────────────
JUNHYUK_PROMPT = """## 페르소나
- 이름: 박준혁, 29세
- 배경: 전직 군상담관 출신, 현재 복지관 자원봉사 상담사
- 역할: 전문 상담 전 단계의 감정 케어. 사용자 감정을 듣고 공감하며, 필요 시 실제 상담으로 연결하는 브릿지

## 말투와 분위기
- 반말만 사용. 존댓말 절대 금지 (예: "~요", "~습니다", "~세요" 금지)
- 오래 알고 지낸 친구한테 하듯 편하고 따뜻하게
- 짧고 간결한 문장 위주. 길게 설명하거나 조언하지 않음
- 딱딱한 상담 용어 절대 금지 (예: "말씀하신 감정은~", "그 부분에 대해 탐색해볼게요" ❌)

## 응답 구조 (3단계, 반드시 순서대로)
너의 답변은 무조건 3문장 이상이어야 해.
첫 번째 문장은 인정(1단계), 두 번째 문장은 사용자의 상황을 구체적으로 언급하는 공감(2단계), 마지막 문장은 질문(3단계)으로 작성해.

### 1단계 — 인정 (타당화, Validation)
심리상담의 "타당화(validation)" 기법을 사용해 사용자의 감정을 인정하고 수용하는 한 문장.
사용자의 현재 감정이 충분히 이해 가능한 반응임을 자연스럽게 표현할 것.

⚠️ 주의:
- 감정은 수용하되 판단은 단정하지 않을 것
- 사용자의 왜곡된 믿음이나 극단적 결론을 사실처럼 동조하지 않을 것
  예) "나는 정말 쓸모없어" → ❌ "맞아, 힘들지" / ✅ "그렇게 느껴질 만큼 힘들었겠다"
- 위 예시에만 국한되지 않고 상황에 맞는 자연스러운 표현을 자유롭게 사용할 것

### 2단계 — 공감 (절대 건너뛰기 금지)
상대방이 말한 상황을 먼저 인지하고, 그 상황에서 느꼈을 감정의 무게나 감각을 내 말로 짚어주는 1~2문장.
단순히 "힘들겠다"로 끝나지 말고, 구체적으로 어떤 상황이라 어떤 느낌일지 연결해서 표현할 것.
반드시 이 단계를 거쳐야 함. 인정 바로 다음에 질문으로 넘어가면 안 됨.

공감 표현 방식:
- 예) ✅ "아침마다 몸이 반응할 만큼 지쳐있는 거잖아" (상황 인지 + 감각 표현)
- 예) ✅ "친한 사람한테 상처받으면 그 말이 더 오래 남잖아" (상황 인지 + 감정 무게)
- 예) ✅ "혼자 다 감당하고 있는 무게가 느껴져" (상황 인지 + 감각)
- 예) ❌ "무기력함이 느껴지겠다" (상황 인지 없이 감정 단어만 나열)
- 예) ❌ "많이 힘들었겠다" (인정 표현을 공감으로 재사용 금지)

## 절대 하지 말아야 할 것
- 존댓말 사용 ("~요", "~습니다", "~세요", "~네요" 전부 금지)
- "힘내", "다 잘 될 거야" 단독 사용
- "별거 아니야", "그게 뭐가 힘들어" 등 감정 축소
- 해결책·조언·행동 제안 먼저 하기
- 증상·병명 진단
- 물음표(?) 2개 이상 — 질문은 무조건 1개
- 인정 표현·질문 연속 중복
- 공감 단계 건너뛰기 (인정 → 바로 질문 금지)

## 나쁜 예시 (이렇게 하면 안 됨)
❌ "그렇구나. 지금 가장 힘든 게 뭐야?" (공감 없이 바로 질문)
❌ "그렇구나. 많이 힘들겠네요. 어떤 기분이에요?" (존댓말 + 물음표 2개)
❌ "그럴 수 있어. 앞으로 이렇게 해보는 건 어때?" (해결책 제시)
❌ "그렇구나. 그렇구나. 뭐가 힘들어?" (인정 표현 중복)

## 좋은 예시 (이렇게 해야 함)
✅ "그게 쉽지 않았겠다. 아침마다 몸이 먼저 반응할 만큼 지쳐있는 거잖아. 요즘 뭐가 제일 버거워?"
✅ "충분히 그럴 수 있어. 친한 사람한테 상처받으면 그 말이 더 오래 남잖아. 어떤 말이 제일 마음에 걸렸어?"
✅ "많이 힘들었겠다. 몸도 마음도 다 내려놓고 싶은 거잖아. 언제부터 그런 느낌이 시작된 것 같아?"
✅ "그 마음 이해해. 혼자 다 감당하고 있는 무게가 느껴져. 지금 가장 바라는 게 뭐야?"

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
    "물음표(?)는 응답 전체에서 반드시 딱 하나만 써. 두 개 이상이면 절대 안 돼. "
    "인정 → 공감 → 질문 순서를 지켜야 해. 공감을 건너뛰면 절대 안 돼. "
    "인정 표현은 매번 다르게 써. '그랬구나', '그렇구나' 같은 표현을 연속으로 쓰지 마. "
    "이전에 쓴 표현을 그대로 반복하지 마.\n\n"
    + JUNHYUK_PROMPT
)

# ────────────────────────────────────────────────────────────
# 스타일 체크 (최신 — 인정 46개, 공감 82개)
# ────────────────────────────────────────────────────────────
HONORIFICS  = ["습니다", "니다", "세요", "하세요", "어요", "아요", "드릴게요", "네요"]
BANNED_EXPR = ["힘내", "다 잘 될 거야", "별거 아니야", "그게 뭐가 힘들어"]

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
]

CHINESE_RANGE   = (0x4E00, 0x9FFF)
JAPANESE_RANGES = [(0x3040, 0x309F), (0x30A0, 0x30FF)]
BRIDGE_TRIGGERS = [
    "자해", "자살", "죽고 싶", "사라지고 싶", "없어지고 싶",
    "어떻게 해야 해", "어쩌지", "방법 알려줘", "어떻게 해",
]

def has_foreign_chars(text):
    for ch in text:
        code = ord(ch)
        if CHINESE_RANGE[0] <= code <= CHINESE_RANGE[1]:
            return True
        for s, e in JAPANESE_RANGES:
            if s <= code <= e:
                return True
    return False

def postprocess(text):
    if has_foreign_chars(text):
        result = []
        for ch in text:
            code = ord(ch)
            if CHINESE_RANGE[0] <= code <= CHINESE_RANGE[1] or any(s <= code <= e for s, e in JAPANESE_RANGES):
                break
            result.append(ch)
        cleaned = "".join(result).strip().rstrip(".,。、")
        text = cleaned if cleaned else "[외국어 혼입 응답 오류]"
    AWKWARD = {
        "쉽게 털어낼 수 있는 게 아니잖아": "그 마음 이해해",
        "쉽게 흐르는 게 아니잖아": "충분히 그럴 수 있어",
        "쉽게 흘러가지 않는": "그게 쉽지 않았겠다",
        "달갑다는 걸": "힘들었겠다",
    }
    for awk, rep in AWKWARD.items():
        if awk in text:
            text = text.replace(awk, rep)
    return text

def check_style(response, user_input=""):
    r = {}
    r["반말·존댓말 없음"] = not any(h in response for h in HONORIFICS)
    r["질문 1개"]         = (response.count("?") + response.count("？")) == 1
    r["금지 표현 없음"]   = not any(b in response for b in BANNED_EXPR)
    r["외국어 혼입 없음"] = not has_foreign_chars(response) and not bool(re.search(r"[a-zA-Z]{4,}", response))
    r["인정 표현 사용"]   = any(w in response for w in RECOGNITION_WORDS)
    r["공감 표현 있음"]   = any(e in response for e in EMPATHY_PATTERNS)
    r["3문장 이상"]       = len([s for s in re.split(r"[.!?]", response) if s.strip()]) >= 3
    needs_bridge   = any(t in user_input for t in BRIDGE_TRIGGERS)
    bridge_present = "준혁쌤이랑 직접 얘기" in response or "연결해볼까" in response
    r["브릿지 포함"] = bridge_present if needs_bridge else "해당없음"
    awkward = [p for p in ["쉽게 털어낼 수 있는 게 아니잖아", "달갑다는 걸", "생성 실패"] if p in response]
    r["⚠️ 어색 패턴"] = f"{awkward}" if awkward else "없음"
    score = sum(v for v in r.values() if isinstance(v, bool))
    total = sum(1 for v in r.values() if isinstance(v, bool))
    r["총점 (자동)"] = f"{score}/{total}  ※ 공감 자연스러움은 수동 확인 필요"
    return r

def check_diversity(responses):
    used = []
    for resp in responses:
        for w in RECOGNITION_WORDS:
            if w in resp:
                used.append(w); break
        else:
            used.append("없음")
    dupes = sum(1 for i in range(1, len(used)) if used[i] == used[i-1] and used[i] != "없음")
    return {"사용된 표현": used, "사용 종류": len(set(used)-{"없음"}), "연속 중복": dupes}

# ────────────────────────────────────────────────────────────
# GPU 초기화
# ────────────────────────────────────────────────────────────
gc.collect()
torch.cuda.empty_cache()

print("=" * 55)
print("✅ 환경 확인")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
print("=" * 55)

# ────────────────────────────────────────────────────────────
# 4-bit 양자화
# ────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ────────────────────────────────────────────────────────────
# 모델 & 토크나이저 로드
# ────────────────────────────────────────────────────────────
print(f"\n📥 모델 로딩 중: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    max_memory={0: "35GiB", "cpu": "10GiB"},
)
model = prepare_model_for_kbit_training(model)
print("✅ 모델 로드 완료")

# ────────────────────────────────────────────────────────────
# LoRA 설정
# ────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA 설정 완료: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ────────────────────────────────────────────────────────────
# 데이터셋
# ────────────────────────────────────────────────────────────
def format_conversation(example):
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

print("\n📂 데이터셋 로드 중...")
with open(TRAIN_PATH, encoding="utf-8") as f:
    raw_train = json.load(f)
with open(VAL_PATH, encoding="utf-8") as f:
    raw_val = json.load(f)

train_dataset = HFDataset.from_list(raw_train).map(format_conversation, remove_columns=HFDataset.from_list(raw_train).column_names)
eval_dataset  = HFDataset.from_list(raw_val).map(format_conversation,   remove_columns=HFDataset.from_list(raw_val).column_names)
print(f"✅ 데이터셋 준비 완료: Train {len(raw_train)}개 / Val {len(raw_val)}개")

# ────────────────────────────────────────────────────────────
# 학습 설정
# ────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    logging_steps=10,
    evaluation_strategy="no",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",
    gradient_checkpointing=True,
)
print("✅ 학습 설정 완료")

# ────────────────────────────────────────────────────────────
# 학습
# ────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    peft_config=lora_config,
    args=training_args,
)

print(f"\n🚀 학습 시작... ({datetime.now().strftime('%H:%M:%S')})")
start = time.time()
stats = trainer.train()
elapsed = round(time.time() - start)

print(f"✅ 학습 완료!")
print(f"   총 학습 시간: {elapsed}초")
print(f"   최종 Loss:   {stats.metrics['train_loss']:.4f}")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ 모델 저장 완료: {OUTPUT_DIR}")

# ────────────────────────────────────────────────────────────
# 빠른 테스트 (파인튜닝 직후 확인용)
# ────────────────────────────────────────────────────────────
def chat(user_text):
    model.eval()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", return_attention_mask=True)
    input_ids      = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=350,     # 3문장 구조 보장
            do_sample=True,
            temperature=0.6,        # 다양성 vs 안정성 균형 최적값 (0.75 시도 → 역효과로 롤백)
            top_p=0.9,              # 상위 90% 토큰 선택
            top_k=50,               # 상위 50개 토큰 필터
            repetition_penalty=1.12, # 반복 억제 (1.15 시도 → 어색함으로 롤백)
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    return postprocess(raw)

print("\n" + "=" * 55)
print("💬 공통 테스트 입력 5개")
print("=" * 55)

responses = []
for msg in TEST_INPUTS:
    resp = chat(msg)
    responses.append(resp)
    score = check_style(resp, msg)
    print(f"\n👤 {msg}")
    print(f"🤖 {resp}")
    print(f"📋 {score}")

div = check_diversity(responses)
print(f"\n📊 인정 표현 다양성")
print(f"   사용 종류: {div['사용 종류']}가지")
print(f"   연속 중복: {div['연속 중복']}회")
print(f"   사용 표현: {div['사용된 표현']}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
result = {
    "model": MODEL_NAME, "train_loss": round(stats.metrics["train_loss"], 4),
    "train_time": elapsed,
    "responses": [{"user": TEST_INPUTS[i], "assistant": responses[i], "score": check_style(responses[i], TEST_INPUTS[i])} for i in range(len(responses))],
    "diversity": div,
}
save_path = f"./test_qwen25_{ts}.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n✅ 결과 저장: {save_path}")