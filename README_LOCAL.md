# README_LOCAL — Windows 로컬 실행 (master 브랜치 기준)

> **작업 브랜치: `master`** (main 아님)  
> 저장소: `https://github.com/g65303402-cmd/project-junhyuk`  
> 프로젝트 루트: 이 README가 있는 디렉터리

---

## 현재 완료 상태 (mock 단계)

| 항목 | 상태 |
|------|------|
| mock 상담 API (`/api/chat`) | ✅ 200, `{ message, emotion, audioUrl }` |
| 프론트 Vite proxy (`5173/api/chat` → `8000/api/chat`) | ✅ 검증 완료 |
| CORS | ✅ 에러 없음 (proxy 경로) |
| `POST /api/tts` | ✅ 호출 없음 (mock 단계) |
| browser TTS fallback | ✅ `VITE_TTS_PROVIDER=browser` |

## 아직 미완료

| 항목 | 필요 작업 |
|------|-----------|
| EXAONE 실모델 | HF 약관 동의, `huggingface-cli login` 또는 `HF_TOKEN`, LoRA `./output/exaone30-junhyuk-final` |
| Qwen3-TTS 실음성 | `voiceclone/` 가중치 다운로드, `qwen_tts` 설치, TTS 서버(:5001) 기동 |

## master에 없는 로컬 폴더 (커밋 대상 아님)

`frontend/`, `backend/`는 **현재 master tracked가 아님**. 팀 합의 전까지 Git 커밋에 포함하지 않는다.  
로컬 mock UI 테스트용으로만 사용한다.

---

## 기존 파일 위치 (master)

| 파일 | 경로 |
|------|------|
| API 서버 | `api_server.py` (루트) |
| TTS 서버 | `voiceclone/tts_server.py` (Flask, port **5001**) |
| 실행 스크립트 (Linux) | `start_servers.sh` |

---

## `api_server.py` 변경 요약

- 환경변수: `MODEL_NAME`, `BASE_MODEL`, `TTS_BASE_URL`, `MAX_NEW_TOKENS`, `ALLOWED_ORIGINS`, `MOCK_MODE`
- **`MOCK_MODE=true`** → EXAONE 로딩 없이 mock `/api/chat`
- **`MOCK_MODE=false`** → lazy-load EXAONE + PEFT (LoRA + HF 필요)
- `/health` 추가, CORS localhost/127.0.0.1, TTS 실패 로깅
- **유지:** `/chat`, `/api/chat`, `/api/tts/audio/{filename}`

### CORS (localhost vs 127.0.0.1)

Vite는 `http://127.0.0.1:5173`, API는 `localhost:8000`처럼 호스트명을 섞을 수 있다.  
`ALLOWED_ORIGINS`에 **localhost와 127.0.0.1 둘 다** 포함했다.  
mock 단계 기본 경로는 **Vite proxy**(`5173/api/chat`)라 CORS 없이 동작한다.

### mock 단계 TTS

master `api_server.py`에는 `/api/tts`가 없다. `.env`에 `VITE_TTS_PROVIDER=browser`로 두고 `audioUrl: null`일 때 **Web Speech API**를 쓴다.

---

## EXAONE 실모델 전환 절차

1. LoRA 어댑터를 `./output/exaone30-junhyuk-final`에 배치
2. [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct) 약관 동의
3. `huggingface-cli login` 또는 `.env`에 `HF_TOKEN=` 설정
4. `.env`에서 `MOCK_MODE=false`로 변경
5. API 서버 재시작: `uvicorn api_server:app --host 127.0.0.1 --port 8000`
6. `GET /health` → `mock_mode: false`, `model_loaded: true` 확인
7. `POST /api/chat` → mock이 아닌 모델 응답 확인

---

## 실행 순서 (master checkout/pull 이후)

### 0. 브랜치 동기화

```powershell
git checkout master
git pull origin master
```

### 1. Python venv

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. 환경변수

```powershell
copy .env.example .env
# MOCK_MODE=true 유지 — LoRA 없이 API/UI 테스트
# VITE_API_BASE_URL은 비워두면 Vite proxy 사용 (권장)
```

### 3. API 서버 (port 8000)

```powershell
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

### 4. 프론트 (로컬 untracked `frontend/`)

```powershell
cd frontend
npm install
npm run dev
```

브라우저: http://127.0.0.1:5173/ — 요청은 `5173/api/chat` → proxy → `8000/api/chat`

### 5. 확인

| 확인 | URL / 방법 |
|------|------------|
| Health | `GET http://127.0.0.1:8000/health` |
| Chat | `POST http://127.0.0.1:8000/api/chat` |

### 6. Qwen3-TTS (실음성 — 미완료)

```powershell
pip install -r voiceclone/requirements.txt
# 가중치 다운로드 후
python voiceclone/tts_server.py
```

---

## 환경변수 요약 (`.env.example`)

```env
MOCK_MODE=true
HF_TOKEN=
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000
# VITE_API_BASE_URL=   # unset = Vite proxy (recommended)
VITE_TTS_PROVIDER=browser
```

커밋 전 검토: `CODEX_REVIEW_CHECKLIST.md`
