import { useMemo, useRef, useState } from 'react';
import { ChatbotHeader } from '../components/chatbot/ChatbotHeader';
import { ChatHomeHero } from '../components/chatbot/ChatHomeHero';
import { CategoryCards } from '../components/chatbot/CategoryCards';
import { ChatInputBar } from '../components/chatbot/ChatInputBar';
import { ChatMessagesPane } from '../components/chatbot/ChatMessagesPane';
import { RecommendChips } from '../components/chatbot/RecommendChips';
import {
  CATEGORIES,
  QUESTIONS_BY_CATEGORY,
  getPromptForQuestion,
  type CategoryId,
} from '../data/recommendQuestions';
import { useChatSession } from '../hooks/useChatSession';

export function UserCounselPage() {
  const [input, setInput] = useState('');
  const [inputFocused, setInputFocused] = useState(false);
  const [counselorConnect, setCounselorConnect] = useState(false);
  const [activeCategory, setActiveCategory] = useState<CategoryId | null>(null);
  const [chipsExpanded, setChipsExpanded] = useState(false);
  const homeRef = useRef<HTMLDivElement>(null);
  const chatContentRef = useRef<HTMLDivElement>(null);

  const {
    messages,
    loading,
    isSpeaking,
    voiceEnabled,
    lastAssistantMessage,
    bottomRef,
    submitMessage,
    replayLast,
    toggleVoice,
    resetSession,
  } = useChatSession({
    autoSpeak: true,
    onNeedsBridge: () => setCounselorConnect(true),
  });

  const isUserTyping = inputFocused && input.trim().length > 0;
  const hasMessages = messages.length > 0;

  const visibleQuestions = useMemo(() => {
    if (activeCategory) {
      return QUESTIONS_BY_CATEGORY[activeCategory];
    }
    return QUESTIONS_BY_CATEGORY.all;
  }, [activeCategory]);

  async function handleSubmit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    setCounselorConnect(false);
    const sent = await submitMessage(trimmed);
    if (sent) setInput('');
  }

  function handleCategorySelect(categoryId: CategoryId) {
    setActiveCategory((current) => (current === categoryId ? null : categoryId));
    setChipsExpanded(false);
    homeRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function handleChipSelect(label: string) {
    setInput(getPromptForQuestion(label));
  }

  function handleClose() {
    if (hasMessages) {
      resetSession();
      setInput('');
      setCounselorConnect(false);
      setActiveCategory(null);
    }
  }

  function handleSearch() {
    const field = document.querySelector<HTMLTextAreaElement>('.chatbot-input-bar__field textarea');
    field?.focus();
  }

  function handleMenuClick() {
    if (hasMessages) {
      chatContentRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
      return;
    }
    setActiveCategory(null);
    setChipsExpanded((value) => !value);
  }

  function handleVoiceToggle() {
    toggleVoice(!voiceEnabled);
  }

  return (
    <div className="chatbot-app">
      <ChatbotHeader title="준혁 상담챗봇" onClose={handleClose} onSearch={handleSearch} />

      <div className="chatbot-shell">
        <aside className="chatbot-video-panel" aria-label="준혁 AI 휴먼">
          <ChatHomeHero
            isSpeaking={isSpeaking}
            isLoading={loading}
            isUserTyping={isUserTyping}
            counselorConnect={counselorConnect}
            showGreeting={!hasMessages}
          />
        </aside>

        <section className="chatbot-chat-panel">
          <div ref={chatContentRef} className="chatbot-chat-panel__content">
            {!hasMessages ? (
              <div ref={homeRef} className="chatbot-home-scroll">
                <CategoryCards
                  categories={CATEGORIES}
                  activeCategory={activeCategory}
                  onSelect={handleCategorySelect}
                />
                <RecommendChips
                  questions={visibleQuestions}
                  expanded={chipsExpanded}
                  onToggleExpand={() => setChipsExpanded((value) => !value)}
                  onSelect={handleChipSelect}
                />
              </div>
            ) : (
              <ChatMessagesPane
                messages={messages}
                bottomRef={bottomRef}
                onConnectBridge={() => setCounselorConnect(true)}
              />
            )}
          </div>

          <ChatInputBar
            input={input}
            loading={loading}
            voiceEnabled={voiceEnabled}
            canReplay={Boolean(lastAssistantMessage.trim())}
            onInputChange={setInput}
            onFocusChange={setInputFocused}
            onSubmit={(text) => void handleSubmit(text)}
            onMenuClick={handleMenuClick}
            onVoiceToggle={handleVoiceToggle}
            onReplay={replayLast}
          />
        </section>
      </div>
    </div>
  );
}
