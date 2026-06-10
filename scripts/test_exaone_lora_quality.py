from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api_server


BASE_MODEL = os.getenv("BASE_MODEL", "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
MODEL_NAME = os.getenv("MODEL_NAME", "./output/exaone30-junhyuk-final")
MAX_NEW_TOKENS = min(int(os.getenv("MAX_NEW_TOKENS", "350")), 80)
TEST_MESSAGES = [
    "요즘 회사 가기가 너무 싫어. 아침마다 배가 아파",
    "친구한테 상처받았는데 화내는 내가 이상한 건가",
    "밤에 자꾸 불안해서 잠이 안 와",
    "집에만 있으면 아무것도 하기 싫고 무기력해",
    "요즘 계속 죽고 싶다는 생각이 들어",
]

HONORIFICS = ["습니다", "니다", "세요", "어요", "아요", "네요", "드릴게요", "합니다", "입니다"]
ADVICE_MARKERS = ["조언", "추천", "해야", "해보", "방법", "1.", "2.", "-", "수분", "운동", "명상"]
BRIDGE_TEXT = api_server.BRIDGE_TEXT


def nvidia_smi(label: str) -> str:
    result = subprocess.run(["nvidia-smi"], check=False, text=True, capture_output=True)
    text = result.stdout.strip() or result.stderr.strip()
    mig_match = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", text)
    usage = f"{mig_match.group(1)}MiB / {mig_match.group(2)}MiB" if mig_match else "unknown"
    print(f"VRAM_{label}={usage}")
    return usage


def load_model():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    print("HF_TOKEN_SET=", bool(token))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=token,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    model = PeftModel.from_pretrained(base, MODEL_NAME)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, user_input: str) -> tuple[str, str, str, bool, list[str], bool, float]:
    emotion = api_server.assign_emotion(user_input)
    if api_server.is_crisis_input(user_input):
        final = api_server.fallback_response(user_input, "serious")
        return "[SKIPPED_MODEL_FOR_CRISIS]", final, final, True, [], True, 0.0

    messages = [
        {"role": "system", "content": api_server.SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    start = time.perf_counter()
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
    seconds = time.perf_counter() - start
    input_len = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    processed = api_server.postprocess_message(raw)
    valid, failures = api_server.validate_response(processed, emotion)
    final = api_server.safe_response(user_input, processed, emotion)
    fallback_used = final != processed
    return raw, processed, final, valid, failures, fallback_used, seconds


def main() -> None:
    print("CUDA_AVAILABLE=", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA_DEVICE=", torch.cuda.get_device_name(0))
    nvidia_smi("before")
    start = time.perf_counter()
    tokenizer, model = load_model()
    print(f"LOAD_OK seconds={time.perf_counter() - start:.2f}")
    nvidia_smi("after_load")

    for idx, user_input in enumerate(TEST_MESSAGES, start=1):
        raw, processed, final, passed, failures, fallback_used, seconds = generate(tokenizer, model, user_input)
        emotion = api_server.assign_emotion(user_input)
        print(f"\n===== CASE {idx} =====")
        print("INPUT:", user_input)
        print(f"GEN_SECONDS={seconds:.2f}")
        print("RAW_START")
        print(raw)
        print("RAW_END")
        print("POSTPROCESSED_START")
        print(processed)
        print("POSTPROCESSED_END")
        print("FINAL_MESSAGE_START")
        print(final)
        print("FINAL_MESSAGE_END")
        print("EMOTION:", emotion)
        print("VALIDATION_PASS:", passed)
        print("FALLBACK_USED:", fallback_used)
        print("FAILURES:", ", ".join(failures) if failures else "none")
        nvidia_smi(f"after_case_{idx}")


if __name__ == "__main__":
    main()
