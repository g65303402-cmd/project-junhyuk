export type SpeakOptions = {
  lang?: string;
  rate?: number;
  pitch?: number;
  volume?: number;
  onStart?: () => void;
  onEnd?: () => void;
  onError?: (error: unknown) => void;
};

const DEFAULT_LANG = 'ko-KR';
const DEFAULT_RATE = 0.95;
const DEFAULT_PITCH = 1.0;
const DEFAULT_VOLUME = 1.0;

let voicesReadyPromise: Promise<SpeechSynthesisVoice[]> | null = null;

export function isSpeechSupported(): boolean {
  return typeof window !== 'undefined' && 'speechSynthesis' in window;
}

function loadVoices(): Promise<SpeechSynthesisVoice[]> {
  if (!isSpeechSupported()) return Promise.resolve([]);
  if (voicesReadyPromise) return voicesReadyPromise;

  voicesReadyPromise = new Promise((resolve) => {
    const readVoices = () => window.speechSynthesis.getVoices();
    const tryResolve = () => {
      const voices = readVoices();
      if (voices.length > 0) {
        resolve(voices);
        return true;
      }
      return false;
    };

    if (tryResolve()) return;

    window.speechSynthesis.onvoiceschanged = () => {
      tryResolve();
    };

    window.setTimeout(() => resolve(readVoices()), 500);
  });

  return voicesReadyPromise;
}

function selectVoice(voices: SpeechSynthesisVoice[], lang: string): SpeechSynthesisVoice | undefined {
  const normalized = lang.toLowerCase();
  return (
    voices.find((voice) => voice.lang.toLowerCase() === normalized) ??
    voices.find((voice) => voice.lang.toLowerCase().startsWith('ko')) ??
    voices[0]
  );
}

export function stopSpeaking(): void {
  if (!isSpeechSupported()) return;
  window.speechSynthesis.cancel();
}

export async function speakText(text: string, options?: SpeakOptions): Promise<void> {
  if (!isSpeechSupported()) {
    const error = new Error('Speech synthesis is not supported in this browser.');
    options?.onError?.(error);
    throw error;
  }

  const trimmed = text.trim();
  if (!trimmed) return;

  stopSpeaking();

  const voices = await loadVoices();
  const lang = options?.lang ?? DEFAULT_LANG;

  return new Promise((resolve, reject) => {
    const utterance = new SpeechSynthesisUtterance(trimmed);
    const voice = selectVoice(voices, lang);
    if (voice) {
      utterance.voice = voice;
      utterance.lang = voice.lang;
    } else {
      utterance.lang = lang;
    }

    utterance.rate = options?.rate ?? DEFAULT_RATE;
    utterance.pitch = options?.pitch ?? DEFAULT_PITCH;
    utterance.volume = options?.volume ?? DEFAULT_VOLUME;

    utterance.onstart = () => options?.onStart?.();
    utterance.onend = () => {
      options?.onEnd?.();
      resolve();
    };
    utterance.onerror = (event) => {
      options?.onError?.(event);
      reject(event);
    };

    window.speechSynthesis.speak(utterance);
  });
}
