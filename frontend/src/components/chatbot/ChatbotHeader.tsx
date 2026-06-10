type ChatbotHeaderProps = {
  title: string;
  onClose?: () => void;
  onSearch?: () => void;
};

export function ChatbotHeader({ title, onClose, onSearch }: ChatbotHeaderProps) {
  return (
    <header className="chatbot-header">
      <button
        type="button"
        className="chatbot-header__icon"
        aria-label="닫기"
        onClick={onClose}
      >
        ✕
      </button>
      <h1 className="chatbot-header__title">{title}</h1>
      <button
        type="button"
        className="chatbot-header__icon"
        aria-label="검색"
        onClick={onSearch}
      >
        🔍
      </button>
    </header>
  );
}
