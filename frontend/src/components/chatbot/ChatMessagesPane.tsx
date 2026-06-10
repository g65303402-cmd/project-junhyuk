import type { RefObject } from 'react';
import { MessageBubble } from '../MessageBubble';
import type { ChatMessage } from '../../types/chat';

type ChatMessagesPaneProps = {
  messages: ChatMessage[];
  bottomRef: RefObject<HTMLDivElement | null>;
  onConnectBridge?: () => void;
};

export function ChatMessagesPane({
  messages,
  bottomRef,
  onConnectBridge,
}: ChatMessagesPaneProps) {
  return (
    <div className="chatbot-messages-scroll" aria-live="polite">
      <div className="chatbot-messages">
        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            showDevDetails={false}
            speakerName="준혁"
            variant="chatbot"
            onConnectBridge={onConnectBridge}
          />
        ))}
      </div>
      <div ref={bottomRef} className="chatbot-messages__anchor" />
    </div>
  );
}
