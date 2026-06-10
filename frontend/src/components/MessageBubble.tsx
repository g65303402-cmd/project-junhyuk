import { BridgeCard } from './BridgeCard';
import { EmotionBadge } from './EmotionBadge';
import { StyleCheckPanel } from './StyleCheckPanel';
import type { ChatMessage } from '../types/chat';

interface MessageBubbleProps {
  message: ChatMessage;
  showDevDetails?: boolean;
  speakerName?: string;
  variant?: 'default' | 'chatbot';
  onConnectBridge?: () => void;
}

export function MessageBubble({
  message,
  showDevDetails = false,
  speakerName = '준혁',
  variant = 'default',
  onConnectBridge,
}: MessageBubbleProps) {
  if (message.role === 'error') {
    return (
      <div className="message-row error">
        <div className="bubble error-bubble" role="alert">
          <p className="message-text">{message.text}</p>
        </div>
      </div>
    );
  }

  if (message.role === 'assistant-loading') {
    const rowClass = `message-row assistant${variant === 'chatbot' ? ' message-row--chatbot' : ''}`;
    const bubbleClass = `bubble assistant-bubble assistant-bubble--loading${variant === 'chatbot' ? ' bubble--chatbot' : ''}`;

    return (
      <div className={rowClass}>
        <div className={bubbleClass} role="status" aria-live="polite">
          <div className="speaker">{speakerName}</div>
          <p className="message-text loading-bubble-text">
            {message.text}
            <span className="typing-dots" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
          </p>
        </div>
      </div>
    );
  }

  const isUser = message.role === 'user';
  const isSerious = message.emotion === 'serious' && message.needsBridge;
  const rowClass = `message-row ${isUser ? 'user' : 'assistant'}${variant === 'chatbot' ? ' message-row--chatbot' : ''}`;
  const bubbleClass = `bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}${isSerious ? ' assistant-bubble--serious' : ''}${variant === 'chatbot' ? ' bubble--chatbot' : ''}`;

  return (
    <div className={rowClass}>
      <div className={bubbleClass}>
        {!isUser && <div className="speaker">{speakerName}</div>}
        <p className="message-text">{message.text}</p>
        {showDevDetails && !isUser && message.emotion && (
          <EmotionBadge emotion={message.emotion} emphasized={isSerious} />
        )}
        {!isUser && (
          <BridgeCard visible={Boolean(message.needsBridge)} onConnect={onConnectBridge} />
        )}
        {showDevDetails && !isUser && (
          <>
            <StyleCheckPanel styleCheck={message.styleCheck} />
            {message.debug && (
              <pre className="debug-json">{JSON.stringify(message.debug, null, 2)}</pre>
            )}
          </>
        )}
      </div>
    </div>
  );
}
