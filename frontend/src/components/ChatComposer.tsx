import { type FormEvent, type KeyboardEvent } from 'react';

type ChatComposerProps = {
  input: string;
  loading: boolean;
  onInputChange: (value: string) => void;
  onSubmit: (text: string) => void;
  onFocusChange?: (focused: boolean) => void;
  placeholder?: string;
};

export function ChatComposer({
  input,
  loading,
  onInputChange,
  onSubmit,
  onFocusChange,
  placeholder = '지금 마음에 걸리는 걸 적어보세요',
}: ChatComposerProps) {
  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    onSubmit(input);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      onSubmit(input);
    }
  }

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <textarea
        value={input}
        onChange={(event) => onInputChange(event.target.value)}
        onFocus={() => onFocusChange?.(true)}
        onBlur={() => onFocusChange?.(false)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        rows={3}
        disabled={loading}
      />
      <button type="submit" disabled={loading || !input.trim()}>
        {loading ? '전송 중...' : '보내기'}
      </button>
    </form>
  );
}
