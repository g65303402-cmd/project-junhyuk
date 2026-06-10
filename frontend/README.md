# React + TypeScript + Vite

## Vercel 배포 (GPU ngrok API 연동)

Vercel 프로젝트 **Settings → Environment Variables**에 다음을 설정하세요.

| Name | Value |
|------|-------|
| `VITE_API_BASE_URL` | `https://retriever-craftily-mauve.ngrok-free.dev` |

- 환경변수는 **빌드 시점**에 번들에 포함됩니다. 값을 바꾼 뒤 **반드시 재배포**하세요.
- 프론트는 ngrok의 `api_server.py`만 호출합니다 (`POST /api/chat`, 응답의 `audioUrl` fetch).
- TTS 서버(5001)에는 직접 연결하지 않습니다.
- ngrok free warning 우회: 모든 API fetch에 `ngrok-skip-browser-warning: true` 헤더가 자동 추가됩니다.
- 오디오는 ngrok URL을 `<audio src>`에 넣지 않고, fetch → blob → `URL.createObjectURL`로 재생합니다.

로컬 개발: `VITE_API_BASE_URL`을 비우거나 `http://localhost:8000`으로 두고 Vite proxy를 사용할 수 있습니다. (`frontend/.env.example` 참고)

**Production E2E 체크포인트:** [`PRODUCTION_STATUS.md`](./PRODUCTION_STATUS.md) (2026-06-10, chat API 연동 성공 기록)

### AI Human 영상 (env만 설정, 코드 변경 불필요)

| 변수 | 용도 |
|------|------|
| `VITE_SUPABASE_AVATAR_BASE_URL` | Supabase public base — idle/speaking/state mp4 전체 세트 (권장) |
| `VITE_AVATAR_IDLE_VIDEO_URL` | 단일 idle mp4 URL (최소 구성) |
| `VITE_AVATAR_SPEAKING_VIDEO_URL` | 단일 speaking mp4 URL (최소 구성) |

미설정 시 **"준혁" fallback 아이콘** (chat API와 무관). 상세: `PRODUCTION_STATUS.md`.

---

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
