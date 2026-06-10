import { apiFetch } from './baseUrl';
import { parseError } from './parseError';

export type TtsStatusResponse = {
  enabled: boolean;
  provider: string;
  runtime: string;
  available?: boolean;
  ready?: boolean;
  mode?: string;
  busy?: boolean;
  fallbackEnabled?: boolean;
  fallback?: string;
  engineDirExists?: boolean;
  engineScriptExists?: boolean;
  modelDirExists?: boolean;
  tokenizerDirExists?: boolean;
  referenceAudio?: string | null;
  referenceAudioExists?: boolean;
  modelLoaded?: boolean;
  loadError?: string | null;
  outputDir?: string;
  reason?: string | null;
};

export type TtsSynthesizeResponse = {
  audioUrl: string;
  filename?: string;
  mode?: string;
  elapsedSec?: number;
  provider?: string;
  runtime?: string;
};

export async function fetchTtsStatus(): Promise<TtsStatusResponse> {
  const response = await apiFetch('/api/tts/status');
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
}

export async function warmupTts(): Promise<{ ok: boolean; modelLoaded?: boolean; mode?: string }> {
  const response = await apiFetch('/api/tts/warmup', { method: 'POST' });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
}

export async function synthesizeTts(text: string): Promise<TtsSynthesizeResponse> {
  const response = await apiFetch('/api/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: text.trim() }),
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return response.json();
}
