from __future__ import annotations

import os
import subprocess
import time

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
MODEL_NAME = "./output/exaone30-junhyuk-final"
TEST_MESSAGE = "요즘 회사 가기가 너무 싫어. 아침마다 배가 아파"


def print_nvidia_smi(label: str) -> None:
    print(f"\n===== nvidia-smi: {label} =====")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())


def main() -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    print("HF_TOKEN_SET=", bool(token))
    print("CUDA_AVAILABLE=", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA_DEVICE=", torch.cuda.get_device_name(0))
    print_nvidia_smi("before")

    if not os.path.exists(os.path.join(MODEL_NAME, "adapter_config.json")):
        raise FileNotFoundError(f"Missing adapter_config.json under {MODEL_NAME}")
    if not (
        os.path.exists(os.path.join(MODEL_NAME, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(MODEL_NAME, "adapter_model.bin"))
    ):
        raise FileNotFoundError(f"Missing adapter_model.safetensors/bin under {MODEL_NAME}")

    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=token,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
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
    load_seconds = time.perf_counter() - start
    print(f"LOAD_OK seconds={load_seconds:.2f}")
    print_nvidia_smi("after_load")

    messages = [
        {"role": "user", "content": TEST_MESSAGE},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    gen_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_seconds = time.perf_counter() - gen_start
    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    print(f"GENERATE_OK seconds={gen_seconds:.2f}")
    print("GENERATED_TEXT_START")
    print(decoded[:1000])
    print("GENERATED_TEXT_END")
    print_nvidia_smi("after_generate")


if __name__ == "__main__":
    main()
