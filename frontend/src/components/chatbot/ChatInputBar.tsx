import { type FormEvent, type KeyboardEvent, useRef } from 'react';
import { MiniVoiceControls } from './MiniVoiceControls';

type ChatInputBarProps = {
  input: string;
  loading: boolean;
  voiceEnabled: boolean;
  canReplay: boolean;
  onInputChange: (value: string) => void;
  onSubmit: (text: string) => void;
  onFocusChange?: (focused: boolean) => void;
  onMenuClick?: () => void;
  onVoiceToggle: () => void;
  onReplay: () => void;
};

export function ChatInputBar({
  input,
  loading,
  voiceEnabled,
  canReplay,
  onInputChange,
  onSubmit,
  onFocusChange,
  onMenuClick,
  onVoiceToggle,
  onReplay,
}: ChatInputBarProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading) return;
    onSubmit(trimmed);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      const trimmed = input.trim();
      if (!trimmed || loading) return;
      onSubmit(trimmed);
    }
  }

  return (
    <footer className="chatbot-input-bar">
      <form className="chatbot-input-bar__form" onSubmit={handleSubmit}>
        <button
          type="button"
          className="chatbot-input-bar__menu"
          aria-label="메뉴"
          onClick={onMenuClick}
        >
          ▦
        </button>

        <div className="chatbot-input-bar__field">
          <textarea
            ref={textareaRef}
            value={input}
            rows={1}
            placeholder="지금 마음에 걸리는 걸 입력해줘"
            onChange={(event) => onInputChange(event.target.value)}
            onFocus={() => onFocusChange?.(true)}
            onBlur={() => onFocusChange?.(false)}
            onKeyDown={handleKeyDown}
          />
        </div>

        <button
          type="submit"
          className="chatbot-input-bar__send"
          aria-label="보내기"
          disabled={loading || !input.trim()}
        >
          ➜
        </button>
      </form>

      <MiniVoiceControls
        voiceEnabled={voiceEnabled}
        canReplay={canReplay}
        onVoiceToggle={onVoiceToggle}
        onReplay={onReplay}
      />
    </footer>
  );
}
