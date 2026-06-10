# Production E2E 상태 (체크포인트)

**기록일:** 2026-06-10  
**다음 재검증 조건:** GPU 서버 `MOCK_MODE=false` 실모델 전환 후 (프론트 코드 변경 없이 E2E만 재실행)

---

## 확인된 Production 연동 (성공)

| 항목 | 값 / 결과 |
|------|-----------|
| 프론트 URL (Production alias) | `https://frontend-mu-one-20.vercel.app` |
| API base | `https://retriever-craftily-mauve.ngrok-free.dev` |
| Chat endpoint | `POST /api/chat` → **200** |
| CORS | Production alias origin 허용 완료 |
| Request header | `ngrok-skip-browser-warning: true` |
| Assistant message | 화면 표시 정상 |
| `요청에 실패했어` | **표시 안 됨** |
| `[chat API]` console fatal | **없음** |
| LLM (현재) | GPU `MOCK_MODE=true` mock 응답 |
| TTS `audioUrl` | `/api/tts/audio/output.wav` 수신 |
| WAV fetch | ngrok header + blob fetch **성공** |

### 검증에 사용한 테스트 문장

```
요즘 회사 가기가 너무 싫어. 아침마다 배가 아파
```

### Mock 응답 예시 (2026-06-10)

```json
{
  "message": "그게 쉽지 않았겠다.\n아침마다 몸이 먼저 반응할 만큼 지쳐있는 거잖아.\n요즘 뭐가 제일 버거워?",
  "emotion": "neutral",
  "audioUrl": "/api/tts/audio/output.wav"
}
```

### Vercel Production env (API)

| Name | Value |
|------|-------|
| `VITE_API_BASE_URL` | `https://retriever-craftily-mauve.ngrok-free.dev` |

- 값은 **키 이름 없이 URL만** 넣을 것 (`VITE_API_BASE_URL=https://...` 형태로 통째로 넣으면 안 됨).
- env 변경 후 **Production 재배포** 필수.
- 테스트는 **구 배포 URL** (`frontend-iek95s5s7-...`)이 아니라 **Production alias**로 할 것.

### GPU 서버 측 (참고)

- ngrok URL 변경 시 Vercel `VITE_API_BASE_URL` + GPU `ALLOWED_ORIGINS` 둘 다 갱신.
- `MOCK_MODE=false` 전환 후 동일 alias에서 chat / audioUrl / TTS E2E 재검증.

---

## AI Human 영상 (API 연결과 별개)

### 현재 상태

- `/` 라우트에서 `AiHumanAvatar`는 **렌더됨**.
- 영상 env 미설정 → **`준혁` fallback 아이콘** 표시 (404는 Supabase/영상 URL 미설정).
- **chat API 성공 여부와 무관.**

### fallback이 뜨는 조건

`AiHumanAvatar`는 `fetchAvatarConfig()` 결과로 판단한다.

1. **env에 영상 URL이 없음** (`hasVideos: false`) → fallback
2. **env에 URL은 있으나** `<video>` 로드/재생 실패 (`videoError`) → fallback
3. **활성 상태용 URL이 없음** (`activeVideoUrl` null) → fallback

env 우선순위 (`client.ts` → `avatarFromEnv.ts`):

1. `VITE_SUPABASE_AVATAR_BASE_URL` 이 있으면 → Supabase base + 고정 상대경로로 전체 세트 구성 (**권장, Vercel static**)
2. 없으면 `VITE_AVATAR_IDLE_VIDEO_URL` / `VITE_AVATAR_SPEAKING_VIDEO_URL` 단일 URL
3. 둘 다 없으면 → `/api/avatar/config` 시도 (ngrok API). 실패 시 env fallback (`hasVideos: false`)
4. 위 모두 실패 → **"준혁" 아이콘 fallback**

### AI Human에 필요한 env 변수

#### 방법 A — Supabase base (권장, Vercel Production)

| 변수 | 설명 |
|------|------|
| `VITE_SUPABASE_AVATAR_BASE_URL` | Supabase Storage public base URL (끝 `/` 없이) |

base 아래 **고정 경로** (코드에 정의됨):

```
idle/idle-01.mp4 … idle/idle-03.mp4
speaking/speaking-01.mp4 … speaking/speaking-06.mp4
states/user-typing.mp4
states/counselor-connect.mp4
```

예:

```env
VITE_SUPABASE_AVATAR_BASE_URL=https://<project>.supabase.co/storage/v1/object/public/<bucket>/avatar
```

실제 URL = `{base}/idle/idle-01.mp4` 등.

#### 방법 B — 단일 영상 URL (최소 구성)

| 변수 | 설명 |
|------|------|
| `VITE_AVATAR_IDLE_VIDEO_URL` | idle mp4 전체 URL 1개 |
| `VITE_AVATAR_SPEAKING_VIDEO_URL` | speaking mp4 전체 URL 1개 |

state 영상(user-typing, counselor-connect)은 이 방식에서는 null.

### Supabase 영상 URL 준비 후 설정 위치

1. Supabase Storage에 위 경로 구조로 mp4 업로드 (bucket public 또는 signed URL 정책 확인)
2. **Vercel → Project → Settings → Environment Variables → Production**
   - `VITE_SUPABASE_AVATAR_BASE_URL=<public base>` 추가 또는 수정
3. **Production 재배포** (Vite env는 빌드 시 번들에 포함)
4. `https://frontend-mu-one-20.vercel.app` 에서 hero 영상 재생 확인

로컬: `frontend/.env.local` 또는 프로젝트 루트 `.env` (`vite.config.ts` `envDir: '..'`)

### 코드 수정 없이 env만으로 가능한가?

**예.** `avatarFromEnv.ts` / `AiHumanAvatar`는 이미 env 기반 URL을 지원한다.

- Supabase base 또는 idle/speaking URL env만 설정 + Vercel 재배포면 됨.
- 프론트 코드 변경 불필요.
- GPU `ALLOWED_ORIGINS` / chat API와는 별도 작업.

---

## MOCK_MODE=false 전환 후 재검증 체크리스트

- [ ] `POST https://retriever-craftily-mauve.ngrok-free.dev/api/chat` → 200
- [ ] CORS (Production alias origin)
- [ ] `message` / `emotion` / `audioUrl` JSON
- [ ] 화면 assistant 표시, `요청에 실패했어` 없음
- [ ] `[chat API]` fatal 없음
- [ ] `audioUrl` blob 재생 또는 browser TTS fallback
- [ ] (선택) AI Human env 설정 후 영상 재생
