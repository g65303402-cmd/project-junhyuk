import { getTtsProvider } from '../tts/ttsProvider';

type VoiceControlsProps = {
  variant?: 'user' | 'dev';
  voiceEnabled: boolean;
  autoSpeak?: boolean;
  speechSupported: boolean;
  isSpeaking?: boolean;
  canReplay: boolean;
  onVoiceEnabledChange: (enabled: boolean) => void;
  onAutoSpeakChange?: (enabled: boolean) => void;
  onStop?: () => void;
  onReplay: () => void;
};

export function VoiceControls({
  variant = 'user',
  voiceEnabled,
  autoSpeak = true,
  speechSupported,
  isSpeaking = false,
  canReplay,
  onVoiceEnabledChange,
  onAutoSpeakChange,
  onStop,
  onReplay,
}: VoiceControlsProps) {
  const isDev = variant === 'dev';

  return (
    <section className={`voice-controls voice-controls--${variant}`} aria-label="음성">
      {isDev && <h2>음성 컨트롤</h2>}

      <div className="voice-controls-actions">
        <label className="voice-toggle">
          <input
            type="checkbox"
            checked={voiceEnabled}
            disabled={!speechSupported}
            onChange={(event) => onVoiceEnabledChange(event.target.checked)}
          />
          음성 {voiceEnabled ? '켜짐' : '꺼짐'}
        </label>

        {isDev && onAutoSpeakChange && (
          <label className="voice-toggle">
            <input
              type="checkbox"
              checked={autoSpeak}
              disabled={!speechSupported || !voiceEnabled}
              onChange={(event) => onAutoSpeakChange(event.target.checked)}
            />
            자동 읽기
          </label>
        )}

        {isDev && onStop && (
          <button
            type="button"
            className="voice-action-button"
            disabled={!speechSupported || !isSpeaking}
            onClick={onStop}
          >
            읽기 중지
          </button>
        )}

        <button
          type="button"
          className="voice-action-button voice-action-button--primary"
          disabled={!speechSupported || !voiceEnabled || !canReplay}
          onClick={onReplay}
        >
          다시 듣기
        </button>
      </div>

      {!speechSupported && isDev && (
        <p className="voice-unsupported">
          {getTtsProvider() === 'api'
            ? '보이스클로닝 TTS를 사용할 수 없습니다. 백엔드 TTS 설정을 확인하세요.'
            : '이 브라우저는 Web Speech API를 지원하지 않습니다.'}
        </p>
      )}
    </section>
  );
}
