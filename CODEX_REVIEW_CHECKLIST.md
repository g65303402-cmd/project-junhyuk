# Codex Review Checklist

After Cursor finishes its patch, review this list before committing.

## Git State

- [ ] Confirm the current branch is `master`.
- [ ] Confirm `.env` is not a commit target in `git status`.
- [ ] Confirm `voiceclone/Qwen3-TTS` is not staged.
- [ ] Confirm `frontend/` and `backend/` are not included in the commit without team agreement.
- [ ] Confirm `api_server.py` was not duplicated or recreated as a second server file.
- [ ] Confirm large model files are not included in Git.

## Mock API

- [ ] Confirm `/health` returns 200 with `MOCK_MODE=true`.
- [ ] Confirm `/api/chat` returns 200 with `MOCK_MODE=true`.
- [ ] Confirm `/api/chat` returns `{ message, emotion, audioUrl }`.
- [ ] Confirm default CORS includes both `http://localhost:5173` and `http://127.0.0.1:5173`.

## Frontend Fallback

- [ ] Confirm the frontend uses browser TTS fallback when `audioUrl: null`.
- [ ] Confirm `/api/tts` 404 does not break the user screen.

## Real Model Mode

- [ ] Test `MOCK_MODE=false` only after HF login, EXAONE terms acceptance, and LoRA path preparation.
- [ ] Confirm GPU-server-only settings such as `max_memory=35GiB` are not left as local defaults.
