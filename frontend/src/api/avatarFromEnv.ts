import type { AvatarConfig } from '../types/avatar';

const IDLE_PATHS = ['idle/idle-01.mp4', 'idle/idle-02.mp4', 'idle/idle-03.mp4'];
const SPEAKING_PATHS = [
  'speaking/speaking-01.mp4',
  'speaking/speaking-02.mp4',
  'speaking/speaking-03.mp4',
  'speaking/speaking-04.mp4',
  'speaking/speaking-05.mp4',
  'speaking/speaking-06.mp4',
];
const STATE_PATHS = {
  userTyping: 'states/user-typing.mp4',
  counselorConnect: 'states/counselor-connect.mp4',
} as const;

function joinBase(base: string, rel: string): string {
  return `${base.replace(/\/+$/, '')}/${rel.replace(/^\/+/, '')}`;
}

/** Build avatar URLs from Vite env (Vercel/static deploy without backend). */
export function getAvatarConfigFromEnv(): AvatarConfig {
  const base = (import.meta.env.VITE_SUPABASE_AVATAR_BASE_URL as string | undefined)?.trim() ?? '';
  const idleUrl = (import.meta.env.VITE_AVATAR_IDLE_VIDEO_URL as string | undefined)?.trim() ?? '';
  const speakingUrl =
    (import.meta.env.VITE_AVATAR_SPEAKING_VIDEO_URL as string | undefined)?.trim() ?? '';

  if (base) {
    const idleVideos = IDLE_PATHS.map((p) => joinBase(base, p));
    const speakingVideos = SPEAKING_PATHS.map((p) => joinBase(base, p));
    const stateVideos = {
      userTyping: joinBase(base, STATE_PATHS.userTyping),
      counselorConnect: joinBase(base, STATE_PATHS.counselorConnect),
    };
    return {
      source: 'supabase-env',
      hasVideos: true,
      idleVideos,
      speakingVideos,
      stateVideos,
      fallbackMessage: '준혁이 기다리고 있어요',
    };
  }

  const idleVideos = idleUrl ? [idleUrl] : [];
  const speakingVideos = speakingUrl ? [speakingUrl] : [];
  if (idleVideos.length || speakingVideos.length) {
    return {
      source: 'env',
      hasVideos: true,
      idleVideos,
      speakingVideos,
      stateVideos: { userTyping: null, counselorConnect: null },
      fallbackMessage: '준혁이 기다리고 있어요',
    };
  }

  return {
    source: 'not-configured',
    hasVideos: false,
    idleVideos: [],
    speakingVideos: [],
    stateVideos: { userTyping: null, counselorConnect: null },
    fallbackMessage: '준혁이 기다리고 있어요',
  };
}
