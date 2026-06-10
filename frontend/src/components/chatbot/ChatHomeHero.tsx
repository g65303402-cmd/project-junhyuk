import { AiHumanAvatar } from '../AiHumanAvatar';

type ChatHomeHeroProps = {
  isSpeaking: boolean;
  isLoading: boolean;
  isUserTyping: boolean;
  counselorConnect: boolean;
  showGreeting?: boolean;
};

export function ChatHomeHero({
  isSpeaking,
  isLoading,
  isUserTyping,
  counselorConnect,
  showGreeting = true,
}: ChatHomeHeroProps) {
  return (
    <section className="chatbot-hero" aria-label="상담 시작">
      <AiHumanAvatar
        variant="hero"
        isSpeaking={isSpeaking}
        isLoading={isLoading}
        isUserTyping={isUserTyping}
        counselorConnect={counselorConnect}
        mode="user"
      />
      {showGreeting && (
        <div className="chatbot-hero__text">
          <p className="chatbot-hero__greeting">고객님, 안녕하세요.</p>
          <p className="chatbot-hero__sub">마음에 걸리는 이야기를 편하게 들려줘.</p>
        </div>
      )}
    </section>
  );
}
