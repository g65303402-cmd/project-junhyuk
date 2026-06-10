import type { SpeakOptions } from './browserTts';
import { synthesizeTts } from '../api/tts';
import { resolveMediaUrl } from '../api/baseUrl';

let currentAudio: HTMLAudioElement | null = null;

export function stopApiSpeech(): void {
  if (!currentAudio) return;
  currentAudio.pause();
  currentAudio.src = '';
  currentAudio = null;
}

/** POST /api/tts on api_server.py → play server-generated audioUrl. */
export async function speakTextViaApi(text: string, options?: SpeakOptions): Promise<void> {
  const trimmed = text.trim();
  if (!trimmed) return;

  stopApiSpeech();

  const data = await synthesizeTts(trimmed);
  const audio = new Audio(resolveMediaUrl(data.audioUrl));
  currentAudio = audio;

  return new Promise((resolve, reject) => {
    audio.onplay = () => options?.onStart?.();
    audio.onended = () => {
      if (currentAudio === audio) currentAudio = null;
      options?.onEnd?.();
      resolve();
    };
    audio.onerror = (event) => {
      if (currentAudio === audio) currentAudio = null;
      reject(event);
    };

    audio.play().catch(reject);
  });
}

export function isApiSpeechSupported(): boolean {
  return typeof window !== 'undefined' && typeof Audio !== 'undefined';
}
