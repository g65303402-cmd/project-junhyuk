type MiniVoiceControlsProps = {
  voiceEnabled: boolean;
  canReplay: boolean;
  onVoiceToggle: () => void;
  onReplay: () => void;
};

export function MiniVoiceControls({
  voiceEnabled,
  canReplay,
  onVoiceToggle,
  onReplay,
}: MiniVoiceControlsProps) {
  return (
    <div className="chatbot-voice-mini" aria-label="음성">
      <button
        type="button"
        className={`chatbot-voice-mini__btn${voiceEnabled ? ' is-on' : ''}`}
        aria-label={voiceEnabled ? '음성 끄기' : '음성 켜기'}
        aria-pressed={voiceEnabled}
        onClick={onVoiceToggle}
      >
        {voiceEnabled ? '🔊' : '🔇'}
      </button>
      <button
        type="button"
        className="chatbot-voice-mini__btn"
        aria-label="다시 듣기"
        disabled={!canReplay || !voiceEnabled}
        onClick={onReplay}
      >
        ↺
      </button>
    </div>
  );
}
