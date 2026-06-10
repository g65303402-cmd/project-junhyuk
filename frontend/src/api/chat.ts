import type { ChatResponse } from '../types/chat';
import { buildChatRequestMeta, logChatApiFetchError } from './apiDebug';
import { apiFetch } from './baseUrl';

type ApiChatResponse = {
  text?: string;
  message?: string;
  audioUrl?: string | null;
  audio_url?: string | null;
  emotion: ChatResponse['emotion'];
  needsBridge?: boolean;
  styleCheck?: ChatResponse['styleCheck'];
  debug?: ChatResponse['debug'];
};

function pickAudioUrl(data: ApiChatResponse): string | null {
  const raw = data.audioUrl ?? data.audio_url ?? null;
  if (!raw || typeof raw !== 'string') return null;
  const trimmed = raw.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function parseErrorFromBody(response: Response, bodyText: string): string {
  try {
    const data = JSON.parse(bodyText) as { detail?: string };
    if (typeof data.detail === 'string') return data.detail;
    return JSON.stringify(data);
  } catch {
    return bodyText.slice(0, 120) || response.statusText || '요청에 실패했어.';
  }
}

function parseJsonBody(bodyText: string, contentType: string): ApiChatResponse {
  if (!contentType.includes('application/json')) {
    throw new Error(
      `API returned non-JSON (${contentType || 'unknown'}). Check ngrok headers. ${bodyText.slice(0, 120)}`,
    );
  }
  return JSON.parse(bodyText) as ApiChatResponse;
}

export async function sendChatMessage(message: string): Promise<ChatResponse> {
  const { requestUrl } = buildChatRequestMeta('/api/chat');

  let response: Response;
  try {
    response = await apiFetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
  } catch (error) {
    logChatApiFetchError({ requestUrl, error });
    throw error instanceof Error ? error : new Error('Failed to fetch');
  }

  const contentType = response.headers.get('content-type') ?? '';
  const responseText = await response.text().catch(() => '');

  if (!response.ok) {
    logChatApiFetchError({
      requestUrl,
      error: new Error(`HTTP ${response.status}`),
      response,
      responseText,
    });
    throw new Error(parseErrorFromBody(response, responseText));
  }

  let data: ApiChatResponse;
  try {
    data = parseJsonBody(responseText, contentType);
  } catch (error) {
    logChatApiFetchError({ requestUrl, error, response, responseText });
    throw error instanceof Error ? error : new Error('Invalid API response');
  }

  const text = (data.text ?? data.message ?? '').trim();
  if (!text) {
    throw new Error('API response missing message text');
  }

  return {
    message: text,
    emotion: data.emotion,
    audioUrl: pickAudioUrl(data),
    needsBridge: data.needsBridge ?? false,
    styleCheck: data.styleCheck,
    debug: data.debug ?? {
      mockMode: false,
      bridgeTriggered: data.needsBridge ?? false,
      responsePath: 'model',
    },
  };
}
