import { fetchMediaBlob } from '../api/baseUrl';
import { speakText as browserSpeakText, type SpeakOptions } from './browserTts';

let currentAudio: HTMLAudioElement | null = null;
let currentObjectUrl: string | null = null;

function cleanupAudio(): void {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.src = '';
    currentAudio = null;
  }
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }
}

export function stopChatAudio(): void {
  cleanupAudio();
}

/**
 * Play TTS from chat response audioUrl (fetch blob + object URL).
 * Falls back to browser Web Speech when audioUrl is null or fetch/play fails.
 */
export async function speakAssistantMessage(
  text: string,
  audioUrl: string | null | undefined,
  options?: SpeakOptions,
): Promise<void> {
  const trimmed = text.trim();
  if (!trimmed) return;

  cleanupAudio();

  if (audioUrl) {
    try {
      const blob = await fetchMediaBlob(audioUrl);
      const objectUrl = URL.createObjectURL(blob);
      currentObjectUrl = objectUrl;

      const audio = new Audio(objectUrl);
      currentAudio = audio;

      await new Promise<void>((resolve, reject) => {
        audio.onplay = () => options?.onStart?.();
        audio.onended = () => {
          cleanupAudio();
          options?.onEnd?.();
          resolve();
        };
        audio.onerror = (event) => {
          cleanupAudio();
          reject(event);
        };
        audio.play().catch(reject);
      });
      return;
    } catch {
      cleanupAudio();
    }
  }

  await browserSpeakText(trimmed, options);
}
