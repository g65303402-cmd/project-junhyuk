import { useCallback, useEffect, useRef, useState } from 'react';
import { sendChatMessage } from '../api/chat';
import type { ChatMessage } from '../types/chat';
import { isSpeechSupported, stopSpeaking } from '../tts/ttsProvider';
import { speakAssistantMessage, stopChatAudio } from '../tts/chatAudio';
import {
  ASSISTANT_LOADING_STAGE_DELAYS_MS,
  ASSISTANT_LOADING_TEXTS,
} from './loadingStageText';

function createId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function createPendingAssistantMessage(): ChatMessage {
  return {
    id: `pending-assistant-${createId()}`,
    role: 'assistant-loading',
    text: ASSISTANT_LOADING_TEXTS[0],
  };
}

type UseChatSessionOptions = {
  autoSpeak?: boolean;
  onNeedsBridge?: () => void;
};

export function useChatSession(options: UseChatSessionOptions = {}) {
  const { autoSpeak = true, onNeedsBridge } = options;
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [lastAssistantMessage, setLastAssistantMessage] = useState('');
  const speechSupported = isSpeechSupported();
  const bottomRef = useRef<HTMLDivElement>(null);
  const loadingTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const requestSeqRef = useRef(0);

  const clearLoadingTimers = useCallback(() => {
    loadingTimersRef.current.forEach(clearTimeout);
    loadingTimersRef.current = [];
  }, []);

  const updatePendingAssistantText = useCallback((pendingId: string, stageIndex: number) => {
    const text = ASSISTANT_LOADING_TEXTS[stageIndex];
    if (!text) return;
    setMessages((prev) =>
      prev.map((message) =>
        message.id === pendingId ? { ...message, text } : message,
      ),
    );
  }, []);

  const startPendingAssistantTimers = useCallback(
    (pendingId: string) => {
      clearLoadingTimers();
      loadingTimersRef.current = ASSISTANT_LOADING_STAGE_DELAYS_MS.map((delayMs, index) =>
        setTimeout(() => updatePendingAssistantText(pendingId, index + 1), delayMs),
      );
    },
    [clearLoadingTimers, updatePendingAssistantText],
  );

  useEffect(() => {
    const anchor = bottomRef.current;
    if (!anchor) return;
    const frame = requestAnimationFrame(() => {
      anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
    });
    return () => cancelAnimationFrame(frame);
  }, [messages, loading]);

  useEffect(() => {
    return () => {
      clearLoadingTimers();
      stopSpeaking();
      stopChatAudio();
    };
  }, [clearLoadingTimers]);

  const handleSpeakEnd = useCallback(() => {
    setIsSpeaking(false);
  }, []);

  const playAssistantMessage = useCallback(
    async (messageText: string, audioUrl?: string | null) => {
      if (!speechSupported || !voiceEnabled) return;

      setIsSpeaking(true);
      try {
        await speakAssistantMessage(messageText, audioUrl, {
          onEnd: handleSpeakEnd,
          onError: handleSpeakEnd,
        });
      } catch {
        handleSpeakEnd();
      }
    },
    [speechSupported, voiceEnabled, handleSpeakEnd],
  );

  const submitMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || loading) return false;

      const requestSeq = ++requestSeqRef.current;
      const pendingMessage = createPendingAssistantMessage();

      stopSpeaking();
      stopChatAudio();
      setIsSpeaking(false);
      setLoading(true);

      setMessages((prev) => [
        ...prev,
        { id: createId(), role: 'user', text: trimmed },
        pendingMessage,
      ]);
      startPendingAssistantTimers(pendingMessage.id);

      try {
        const response = await sendChatMessage(trimmed);
        if (requestSeq !== requestSeqRef.current) return false;

        const assistantMessage: ChatMessage = {
          id: createId(),
          role: 'assistant',
          text: response.message,
          emotion: response.emotion,
          needsBridge: response.needsBridge,
          styleCheck: response.styleCheck,
          debug: response.debug,
        };

        setLastAssistantMessage(response.message);
        setMessages((prev) => [
          ...prev.filter((message) => message.id !== pendingMessage.id),
          assistantMessage,
        ]);
        clearLoadingTimers();
        setLoading(false);

        if (response.needsBridge) {
          onNeedsBridge?.();
        }

        if (autoSpeak && voiceEnabled) {
          void playAssistantMessage(response.message, response.audioUrl);
        }
        return true;
      } catch (err) {
        if (requestSeq !== requestSeqRef.current) return false;

        setMessages((prev) => [
          ...prev.filter((message) => message.id !== pendingMessage.id),
          {
            id: createId(),
            role: 'error',
            text:
              err instanceof Error
                ? err.message
                : '상담 응답을 가져오지 못했어. 잠시 후 다시 시도해줘.',
          },
        ]);
        clearLoadingTimers();
        setLoading(false);
        return false;
      }
    },
    [
      loading,
      autoSpeak,
      voiceEnabled,
      playAssistantMessage,
      onNeedsBridge,
      clearLoadingTimers,
      startPendingAssistantTimers,
    ],
  );

  const replayLast = useCallback(() => {
    const lastMessage = lastAssistantMessage.trim();
    if (!lastMessage) return;
    void playAssistantMessage(lastMessage, null);
  }, [lastAssistantMessage, playAssistantMessage]);

  const toggleVoice = useCallback((enabled: boolean) => {
    setVoiceEnabled(enabled);
    if (!enabled) {
      stopSpeaking();
      stopChatAudio();
      setIsSpeaking(false);
    }
  }, []);

  const resetSession = useCallback(() => {
    requestSeqRef.current += 1;
    clearLoadingTimers();
    setLoading(false);
    setMessages([]);
    setLastAssistantMessage('');
    stopSpeaking();
    stopChatAudio();
    setIsSpeaking(false);
  }, [clearLoadingTimers]);

  return {
    messages,
    loading,
    isSpeaking,
    voiceEnabled,
    speechSupported,
    lastAssistantMessage,
    bottomRef,
    submitMessage,
    replayLast,
    toggleVoice,
    resetSession,
  };
};
