# Project Junhyuk — AI 감성 상담사

## 모델
- **Base Model**: LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct
- **Fine-tuning**: QLoRA (LoRA r=16, alpha=32)
- **학습 데이터**: 981개 (12개 감정 카테고리)
- **최적 체크포인트**: checkpoint-244 (loss 0.3047)
- **생성 파라미터**: temperature 0.7 / repetition_penalty 1.15 / max_new_tokens 350

## API 사용법

**서버 실행**
```
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**요청**
```
POST /chat
Content-Type: application/json
{"message": "요즘 너무 힘들어"}
```

**응답**
```
{"message": "그게 쉽지 않았겠다...", "emotion": "comfort"}
```

## 감정 태그
| 태그 | 설명 |
|------|------|
| comfort | 슬픔, 상실, 외로움, 이별, 가족 갈등 |
| serious | 자해, 자살 등 위기 신호 |
| encourage | 진로, 취업, 자존감, 도전 |
| happy | 가벼운 일상 고민, 긍정적 분위기 |
| neutral | 기본값 |
