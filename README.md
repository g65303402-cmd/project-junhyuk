# Project Junhyuk — AI 감성 상담사

## 모델
- **Base Model**: LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct
- **Fine-tuning**: QLoRA (LoRA r=16, alpha=32)
- **학습 데이터**: 981개 (12개 감정 카테고리)
- **최적 체크포인트**: checkpoint-244 (loss 0.3047)
- **생성 파라미터**: temperature 0.7 / repetition_penalty 1.15 / max_new_tokens 350

## TTS
- **모델**: Qwen3-TTS-12Hz-1.7B-Base
- **방식**: 보이스클로닝 (박준혁 페르소나 목소리)
- **실행**: `voiceclone/tts_server.py` (포트 5001)

## 서버 실행

**한번에 실행 (권장)**
```
bash start_servers.sh
```

**개별 실행**
```
# TTS 서버 먼저
/root/miniforge3/envs/qwen-tts/bin/python voiceclone/tts_server.py

# 메인 서버
/opt/conda/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## API 사용법

### 텍스트만 (기본)
```
POST /chat
Content-Type: application/json
{"message": "요즘 너무 힘들어"}

{"message": "그게 쉽지 않았겠다...", "emotion": "comfort"}
```

### 텍스트 + 음성 (TTS 포함)
```
POST /api/chat
Content-Type: application/json
{"message": "요즘 너무 힘들어"}

{
  "message": "그게 쉽지 않았겠다...",
  "emotion": "comfort",
  "audioUrl": "/api/tts/audio/output.wav"
}
```

### 오디오 파일
```
GET /api/tts/audio/output.wav
```

## 감정 태그
| 태그 | 설명 |
|------|------|
| comfort | 슬픔, 상실, 외로움, 이별, 가족 갈등 |
| serious | 자해, 자살 등 위기 신호 |
| encourage | 진로, 취업, 자존감, 도전 |
| happy | 가벼운 일상 고민, 긍정적 분위기 |
| neutral | 기본값 |

## 파일 구조
```
project/
├── api_server.py          # 메인 API 서버 (포트 8000)
├── start_servers.sh       # 서버 한번에 실행 스크립트
├── exaone30_qlora_finetune.py  # 파인튜닝 코드
├── prompt_test_exaone_new.py   # 모델 테스트 코드
├── aihub_pipeline.py      # 데이터 파이프라인
├── dataset_new/           # 학습 데이터셋
│   ├── train_final.json   # 최종 학습 데이터 (981개)
│   ├── val.json           # 검증 데이터
│   └── test.json          # 테스트 데이터
└── voiceclone/            # TTS 보이스클로닝
    ├── tts_server.py      # TTS 서버 (포트 5001)
    ├── voiceclone.py      # TTS 모델 로드/생성
    └── README.md          # TTS 설치 방법
```
