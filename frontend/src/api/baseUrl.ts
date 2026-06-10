/** Backend origin. Empty = same-origin relative `/api/*` (Vite dev proxy in local dev). */
export function getApiBaseUrl(): string {
  const raw = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() ?? '';
  return raw.replace(/\/+$/, '');
}

export const NGROK_SKIP_BROWSER_WARNING = 'ngrok-skip-browser-warning';

function usesNgrokBase(): boolean {
  const base = getApiBaseUrl().toLowerCase();
  return base.includes('ngrok');
}

/** Merge ngrok bypass header when API base is an ngrok URL. */
export function withApiHeaders(initHeaders?: HeadersInit): Headers {
  const headers = new Headers(initHeaders);
  if (usesNgrokBase()) {
    headers.set(NGROK_SKIP_BROWSER_WARNING, 'true');
  }
  return headers;
}

/** Fetch against `apiUrl(path)` with optional ngrok headers. */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  return fetch(apiUrl(path), {
    ...init,
    headers: withApiHeaders(init.headers),
  });
}

/** Human-readable label for /dev diagnostics. */
export function getApiBaseUrlLabel(): string {
  const base = getApiBaseUrl();
  if (base) return base;
  return '(same origin — relative /api/*)';
}

/** Whether requests go through Vite dev proxy (only when base URL is unset). */
export function usesViteDevProxy(): boolean {
  return !getApiBaseUrl() && import.meta.env.DEV;
}

export function getViteProxyTarget(): string {
  return 'http://127.0.0.1:8000';
}

export function apiUrl(path: string): string {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  const base = getApiBaseUrl();
  return base ? `${base}${normalized}` : normalized;
}

/** Resolve server-relative paths such as `/api/tts/audio/foo.wav`. */
export function resolveMediaUrl(url: string): string {
  if (!url) return url;
  if (/^https?:\/\//i.test(url)) return url;
  return apiUrl(url);
}

/** Fetch audio/media via API (blob). Uses ngrok header; never set raw ngrok URL on audio.src. */
export async function fetchMediaBlob(url: string): Promise<Blob> {
  const resolved = resolveMediaUrl(url);
  const response = await fetch(resolved, {
    headers: withApiHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Media fetch failed (${response.status})`);
  }

  const contentType = response.headers.get('content-type') ?? '';
  if (contentType.includes('text/html')) {
    throw new Error('Received HTML instead of audio (ngrok warning page?)');
  }

  return response.blob();
}
