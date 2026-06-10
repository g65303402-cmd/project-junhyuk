import {
  isSpeechSupported as browserIsSpeechSupported,
  speakText as browserSpeakText,
  stopSpeaking as browserStopSpeaking,
  type SpeakOptions,
} from './browserTts';
import { isApiSpeechSupported, speakTextViaApi, stopApiSpeech } from './futureApiTts';

export type TtsProvider = 'browser' | 'api';

const provider = (import.meta.env.VITE_TTS_PROVIDER as TtsProvider | undefined) ?? 'api';
const fallbackProvider =
  (import.meta.env.VITE_TTS_FALLBACK_PROVIDER as TtsProvider | undefined) ?? 'browser';

let lastTtsError: string | null = null;
let lastTtsUsedFallback = false;

export { type SpeakOptions };

export function stopSpeaking(): void {
  browserStopSpeaking();
  stopApiSpeech();
}

/**
 * Speak assistant message.
 * Default: browser Web Speech API (VITE_TTS_PROVIDER=browser).
 * Optional: POST /api/tts with silent browser fallback on failure.
 */
export async function speakText(text: string, options?: SpeakOptions): Promise<void> {
  if (provider === 'api') {
    try {
      lastTtsError = null;
      lastTtsUsedFallback = false;
      return await speakTextViaApi(text, options);
    } catch (error) {
      lastTtsError = error instanceof Error ? error.message : String(error);

      if (fallbackProvider === 'browser' && browserIsSpeechSupported()) {
        lastTtsUsedFallback = true;
        return browserSpeakText(text, options);
      }

      options?.onEnd?.();
      return;
    }
  }

  return browserSpeakText(text, options);
}

export function isSpeechSupported(): boolean {
  if (provider === 'api') {
    return (
      isApiSpeechSupported() ||
      (fallbackProvider === 'browser' && browserIsSpeechSupported())
    );
  }
  return browserIsSpeechSupported();
}

export function getTtsProvider(): TtsProvider {
  return provider;
}

export function getTtsFallbackProvider(): TtsProvider {
  return fallbackProvider;
}

export function getTtsDebugInfo(): {
  provider: TtsProvider;
  fallbackProvider: TtsProvider;
  lastError: string | null;
  usedFallback: boolean;
} {
  return {
    provider,
    fallbackProvider,
    lastError: lastTtsError,
    usedFallback: lastTtsUsedFallback,
  };
}
