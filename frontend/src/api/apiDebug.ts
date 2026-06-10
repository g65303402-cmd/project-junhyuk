import { getApiBaseUrl, NGROK_SKIP_BROWSER_WARNING } from './baseUrl';

export type ChatApiErrorLog = {
  VITE_API_BASE_URL: string;
  requestUrl: string;
  fetchError?: string;
  responseStatus?: number;
  responseTextPreview?: string;
  ngrokHeaderApplied: boolean;
};

function previewText(text: string, maxLen = 300): string {
  return text.length <= maxLen ? text : `${text.slice(0, maxLen)}…`;
}

/** Dev console diagnostics for chat/API failures (no secrets). */
export function logChatApiError(details: ChatApiErrorLog): void {
  console.error('[chat API]', details);
}

export function buildChatRequestMeta(requestPath = '/api/chat'): {
  baseUrl: string;
  requestUrl: string;
  ngrokHeaderApplied: boolean;
} {
  const baseUrl = getApiBaseUrl();
  const normalized = requestPath.startsWith('/') ? requestPath : `/${requestPath}`;
  const requestUrl = baseUrl ? `${baseUrl}${normalized}` : normalized;
  const ngrokHeaderApplied = baseUrl.toLowerCase().includes('ngrok');
  return { baseUrl, requestUrl, ngrokHeaderApplied };
}

export function logChatApiFetchError(input: {
  requestUrl: string;
  error: unknown;
  response?: Response;
  responseText?: string;
}): void {
  const baseUrl = getApiBaseUrl();
  const ngrokHeaderApplied = baseUrl.toLowerCase().includes('ngrok');

  logChatApiError({
    VITE_API_BASE_URL: baseUrl || '(empty — relative /api, Vite proxy in dev only)',
    requestUrl: input.requestUrl,
    fetchError: input.error instanceof Error ? input.error.message : String(input.error),
    responseStatus: input.response?.status,
    responseTextPreview: input.responseText
      ? previewText(input.responseText)
      : undefined,
    ngrokHeaderApplied,
  });

  if (import.meta.env.DEV && ngrokHeaderApplied) {
    console.info(`[chat API] expected header: ${NGROK_SKIP_BROWSER_WARNING}: true`);
  }
}
