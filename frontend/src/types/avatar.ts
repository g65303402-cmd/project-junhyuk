export type AvatarStateVideos = {
  userTyping: string | null;
  counselorConnect: string | null;
};

export type AvatarConfig = {
  source: string;
  hasVideos: boolean;
  idleVideos: string[];
  speakingVideos: string[];
  stateVideos: AvatarStateVideos;
  fallbackMessage: string;
  localPaths?: {
    idle: string;
    speaking: string;
  };
};

export type AvatarVisualState =
  | 'idle'
  | 'userTyping'
  | 'loading'
  | 'speaking'
  | 'counselorConnect';

export type AvatarDebugInfo = {
  visualState: AvatarVisualState;
  activeVideoUrl: string | null;
  videoError: boolean;
  idleCount: number;
  speakingCount: number;
  hasUserTyping: boolean;
  hasCounselorConnect: boolean;
};

export function resolveAvatarVisualState(input: {
  counselorConnect: boolean;
  isSpeaking: boolean;
  isLoading: boolean;
  isUserTyping: boolean;
}): AvatarVisualState {
  if (input.counselorConnect) return 'counselorConnect';
  if (input.isSpeaking) return 'speaking';
  if (input.isLoading) return 'loading';
  if (input.isUserTyping) return 'userTyping';
  return 'idle';
}

export function pickVideoUrl(
  config: AvatarConfig,
  visualState: AvatarVisualState,
  indices: { idle: number; speaking: number },
): string | null {
  if (!config.hasVideos) return null;

  switch (visualState) {
    case 'counselorConnect':
      return config.stateVideos.counselorConnect;
    case 'speaking': {
      if (config.speakingVideos.length === 0) return null;
      const idx = indices.speaking % config.speakingVideos.length;
      return config.speakingVideos[idx] ?? null;
    }
    case 'userTyping':
      return config.stateVideos.userTyping ?? pickIdleUrl(config, indices.idle);
    case 'loading':
    case 'idle':
    default:
      return pickIdleUrl(config, indices.idle);
  }
}

function pickIdleUrl(config: AvatarConfig, idleIndex: number): string | null {
  if (config.idleVideos.length === 0) return null;
  return config.idleVideos[idleIndex % config.idleVideos.length] ?? null;
}
