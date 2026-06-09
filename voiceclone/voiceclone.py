from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
import torch
import soundfile as sf

# 모델 & 프롬프트 전역으로 한 번만 로드
model = None
prompt = None

def load_model(
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ref_audio: str = None,
    ref_text: str = None
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


def generate_tts(test_text: str, output_path: str) -> str:
    """텍스트를 음성으로 변환 후 저장"""
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


# 실행
if __name__ == "__main__":
    load_model(
        model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ref_audio="/smhrd2/Hyunjin Team/Hahyun/project2/voiceclone/새로운-녹음.wav",
        ref_text="그랬구나. 많이 힘들었겠다. 아침마다 몸이 먼저 반응할만큼 지쳐있는 거잖아. 요즘 뭐가 제일 버거워?"
    )

    generate_tts(
        test_text= "그 스트레스 진짜 크겠다. 성적 자체보다 그 시선이 자꾸 나를 감시하는 것처럼 느껴지니까 더 지치는 거잖아. 그게 얼마나 오랫동안 이어져왔는지 궁금해." ,
        output_path="/smhrd2/Hyunjin Team/Hahyun/project2/voiceclone/output.wav"
    )