from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
import torch
import soundfile as sf
import os

model = None
prompt = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ref_audio: str = os.path.join(BASE_DIR, "새로운-녹음.wav"),
    ref_text: str = "그랬구나. 많이 힘들었겠다. 아침마다 몸이 먼저 반응할만큼 지쳐있는 거잖아. 요즘 뭐가 제일 버거워?"
):
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

def generate_tts(test_text: str, output_path: str) -> str:
    global model, prompt
    if model is None:
        raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 실행하세요.")
    wavs, sample_rate = model.generate_voice_clone(
        text=test_text,
        voice_clone_prompt=prompt
    )
    sf.write(output_path, wavs[0], sample_rate)
    print(f"음성 저장 완료: {output_path}")
    return output_path



# # 다른 파일에서 호출하는 방법
# from tts_module import load_model, generate_tts

# # 모델 한 번만 로드
# load_model()

# # 텍스트 받아서 TTS 생성
# text = "여기에 원하는 텍스트"
# output = generate_tts(
#     test_text=text,
#     output_path="voiceclone/output.wav"
# )