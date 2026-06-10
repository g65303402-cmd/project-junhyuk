import type { TtsStatusResponse } from './tts';

export type DetectedBackend =
  | 'gpu-api-server'
  | 'local-mock-backend'
  | 'unreachable'
  | 'unknown';

/** Infer whether /api/tts/status came from GPU api_server.py or local mock backend. */
export function detectBackendFromTtsStatus(
  status: TtsStatusResponse | null,
): DetectedBackend {
  if (!status) return 'unreachable';

  const raw = JSON.stringify(status);

  if (status.mode === 'persistent-subprocess-worker') {
    return 'gpu-api-server';
  }

  if (
    raw.includes('smhrd2') ||
    raw.includes('Qwen3-TTS') && status.engineDirExists
  ) {
    return 'gpu-api-server';
  }

  if (
    raw.includes('C:\\\\Users') ||
    raw.includes('voiceclone') ||
    status.reason === 'model path not found' ||
    status.fallback === 'browser'
  ) {
    return 'local-mock-backend';
  }

  return 'unknown';
}

export function backendDetectLabel(kind: DetectedBackend): string {
  switch (kind) {
    case 'gpu-api-server':
      return 'GPU api_server.py (SSH tunnel 또는 GPU 내부 연결)';
    case 'local-mock-backend':
      return 'Windows 로컬 mock backend — GPU TTS 아님. 로컬 uvicorn 중지 + SSH 터널 필요';
    case 'unreachable':
      return '연결 불가 — SSH 터널 또는 GPU 백엔드 미실행';
    default:
      return '백엔드 종류 불명 — /api/tts/status JSON 확인';
  }
}
