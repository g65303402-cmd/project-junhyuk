import { apiFetch } from './baseUrl';
import { getAvatarConfigFromEnv } from './avatarFromEnv';

export type HealthResponse = {
  ok: boolean;
  service: string;
  mockMode: boolean;
  modelPath?: string;
  tts?: TtsStatusResponse;
};

export type TtsStatusResponse = import('./tts').TtsStatusResponse;

export type { AvatarConfig, AvatarDebugInfo, AvatarVisualState } from '../types/avatar';

export type EvaluationResult = {
  total: number;
  stylePassRate: number;
  questionOneRate: number;
  noHonorificRate: number;
  noForeignRate: number;
  threeSentenceRate: number;
  bridgeCount: number;
  emotionCounts: Record<string, number>;
  items: Array<{
    input: string;
    message: string;
    emotion: string;
    needsBridge: boolean;
    styleCheck: Record<string, boolean | string>;
    debug: Record<string, unknown>;
  }>;
};

export {
  apiUrl,
  getApiBaseUrl,
  getApiBaseUrlLabel,
  getViteProxyTarget,
  resolveMediaUrl,
  usesViteDevProxy,
} from './baseUrl';

async function parseError(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data.detail === 'string') return data.detail;
    return JSON.stringify(data);
  } catch {
    return response.statusText || '요청에 실패했어.';
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await apiFetch('/api/health');
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
}

export async function fetchAvatarConfig(): Promise<import('../types/avatar').AvatarConfig> {
  const envConfig = getAvatarConfigFromEnv();
  if (envConfig.hasVideos) {
    return envConfig;
  }

  try {
    const response = await apiFetch('/api/avatar/config');
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    return response.json();
  } catch {
    return envConfig;
  }
}

export async function fetchTtsStatus(): Promise<TtsStatusResponse> {
  const { fetchTtsStatus: fetchStatus } = await import('./tts');
  return fetchStatus();
}

export { warmupTts } from './tts';

export async function runEvaluation(): Promise<EvaluationResult> {
  const response = await apiFetch('/api/evaluation/run', { method: 'POST' });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
}
