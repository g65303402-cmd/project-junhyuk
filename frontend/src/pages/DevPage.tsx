import { Link } from 'react-router-dom';
import { useEffect, useState } from 'react';
import {
  checkHealth,
  fetchAvatarConfig,
  fetchTtsStatus,
  warmupTts,
  apiUrl,
  resolveMediaUrl,
  getApiBaseUrl,
  getApiBaseUrlLabel,
  getViteProxyTarget,
  usesViteDevProxy,
  type AvatarConfig,
  type AvatarDebugInfo,
  type HealthResponse,
  type TtsStatusResponse,
} from '../api/client';
import { backendDetectLabel, detectBackendFromTtsStatus } from '../api/backendDetect';
import { synthesizeTts, type TtsSynthesizeResponse } from '../api/tts';
import { sendChatMessage } from '../api/chat';
import { AiHumanAvatar } from '../components/AiHumanAvatar';
import { MessageBubble } from '../components/MessageBubble';
import { VoiceControls } from '../components/VoiceControls';
import { TEST_INPUTS, type ChatMessage } from '../types/chat';
import {
  getTtsFallbackProvider,
  getTtsProvider,
  isSpeechSupported,
  speakText,
  stopSpeaking,
  getTtsDebugInfo,
} from '../tts/ttsProvider';

function createId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function DevPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [avatarConfig, setAvatarConfig] = useState<AvatarConfig | null>(null);
  const [avatarDebug, setAvatarDebug] = useState<AvatarDebugInfo | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [counselorConnect, setCounselorConnect] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [ttsStatus, setTtsStatus] = useState<TtsStatusResponse | null>(null);
  const [ttsDebug, setTtsDebug] = useState(getTtsDebugInfo());
  const [warmupLoading, setWarmupLoading] = useState(false);
  const [warmupError, setWarmupError] = useState<string | null>(null);
  const [ttsTestLoading, setTtsTestLoading] = useState(false);
  const [ttsTestText, setTtsTestText] = useState('그게 쉽지 않았겠다.');
  const [ttsTestResult, setTtsTestResult] = useState<TtsSynthesizeResponse | null>(null);
  const [ttsTestError, setTtsTestError] = useState<string | null>(null);
  const [audioPlayResult, setAudioPlayResult] = useState<string | null>(null);

  const apiConnectionInfo = {
    connectionMode: usesViteDevProxy()
      ? 'SSH tunnel + Vite proxy (127.0.0.1:5173 → /api → 127.0.0.1:8000 → GPU)'
      : getApiBaseUrl()
        ? 'direct VITE_API_BASE_URL'
        : 'relative /api (production or custom proxy)',
    VITE_API_BASE_URL: import.meta.env.VITE_API_BASE_URL ?? null,
    resolvedApiBaseUrl: getApiBaseUrl() || null,
    apiBaseLabel: getApiBaseUrlLabel(),
    usesViteDevProxy: usesViteDevProxy(),
    viteProxyTarget: usesViteDevProxy() ? getViteProxyTarget() : null,
    ttsProvider: getTtsProvider(),
    fallbackProvider: getTtsFallbackProvider(),
    detectedBackend: detectBackendFromTtsStatus(ttsStatus),
    detectedBackendLabel: backendDetectLabel(detectBackendFromTtsStatus(ttsStatus)),
    ttsStatusRequestUrl: apiUrl('/api/tts/status'),
    ttsPostRequestUrl: apiUrl('/api/tts'),
    note: getApiBaseUrl()
      ? 'VITE_API_BASE_URL is set — browser calls that host directly (Vite proxy not used).'
      : usesViteDevProxy()
        ? 'Relative /api/* → Vite proxy → 127.0.0.1:8000. This must be SSH-tunneled to GPU api_server.py. Stop local Windows uvicorn on port 8000.'
        : 'Production build: set VITE_API_BASE_URL or serve frontend behind a reverse proxy to the API.',
  };

  const [lastAssistantMessage, setLastAssistantMessage] = useState('');
  const speechSupported = isSpeechSupported();

  useEffect(() => {
    checkHealth().then(setHealth).catch(() => setHealth(null));
    fetchAvatarConfig().then(setAvatarConfig).catch(() => setAvatarConfig(null));
    fetchTtsStatus().then(setTtsStatus).catch(() => setTtsStatus(null));
  }, []);

  async function runSingleTest(text: string) {
    setLoading(true);
    stopSpeaking();
    setIsSpeaking(false);
    setCounselorConnect(false);

    setMessages((prev) => [...prev, { id: createId(), role: 'user', text }]);

    try {
      const response = await sendChatMessage(text);
      setLastAssistantMessage(response.message);
      setMessages((prev) => [
        ...prev,
        {
          id: createId(),
          role: 'assistant',
          text: response.message,
          emotion: response.emotion,
          needsBridge: response.needsBridge,
          styleCheck: response.styleCheck,
          debug: response.debug,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: createId(),
          role: 'error',
          text: err instanceof Error ? err.message : '요청 실패',
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function replayLast() {
    const text = lastAssistantMessage.trim();
    if (!text || !voiceEnabled) return;
    setIsSpeaking(true);
    try {
      await speakText(text, {
        onEnd: () => {
          setIsSpeaking(false);
          setTtsDebug(getTtsDebugInfo());
        },
        onError: () => setIsSpeaking(false),
      });
      setTtsDebug(getTtsDebugInfo());
    } catch {
      setIsSpeaking(false);
    }
  }

  async function handleWarmup() {
    setWarmupLoading(true);
    setWarmupError(null);
    try {
      await warmupTts();
      const status = await fetchTtsStatus();
      setTtsStatus(status);
    } catch (err) {
      setWarmupError(err instanceof Error ? err.message : 'warmup failed');
    } finally {
      setWarmupLoading(false);
    }
  }

  async function refreshTtsStatus() {
    try {
      setTtsStatus(await fetchTtsStatus());
    } catch {
      setTtsStatus(null);
    }
  }

  async function handleTtsTest() {
    const text = ttsTestText.trim();
    if (!text) return;

    setTtsTestLoading(true);
    setTtsTestError(null);
    setTtsTestResult(null);
    setAudioPlayResult(null);

    try {
      const result = await synthesizeTts(text);
      setTtsTestResult(result);
    } catch (err) {
      setTtsTestError(err instanceof Error ? err.message : 'TTS request failed');
    } finally {
      setTtsTestLoading(false);
    }
  }

  async function handlePlayTestAudio() {
    if (!ttsTestResult?.audioUrl) return;

    setAudioPlayResult(null);
    const resolvedUrl = resolveMediaUrl(ttsTestResult.audioUrl);

    try {
      const audio = new Audio(resolvedUrl);
      await new Promise<void>((resolve, reject) => {
        audio.onended = () => resolve();
        audio.onerror = () => reject(new Error('audio playback failed'));
        audio.play().catch(reject);
      });
      setAudioPlayResult(`재생 성공: ${resolvedUrl}`);
    } catch (err) {
      setAudioPlayResult(
        err instanceof Error ? err.message : `재생 실패: ${resolvedUrl}`,
      );
    }
  }

  return (
    <div className="dev-page app-shell">
      <header className="app-header">
        <div>
          <h1>개발자 테스트</h1>
          <p>Mock Mode, 공통 테스트, API 상태 확인</p>
        </div>
        <div className="dev-header-links">
          {health && (
            <span className="mode-badge">{health.mockMode ? 'Mock Mode' : 'Model Mode'}</span>
          )}
          <Link to="/">사용자 화면</Link>
          <Link to="/evaluation">평가 패널</Link>
        </div>
      </header>

      <section className="dev-info-card">
        <h2>API 연결 (프론트 → 백엔드)</h2>
        <pre>{JSON.stringify(apiConnectionInfo, null, 2)}</pre>
      </section>

      <section className="dev-info-card">
        <h2>API Health</h2>
        <pre>{health ? JSON.stringify(health, null, 2) : '로딩 중...'}</pre>
      </section>

      <section className="dev-info-card">
        <h2>/api/tts/status</h2>
        <p>
          Request URL: <code>{apiUrl('/api/tts/status')}</code>
        </p>
        <p>
          POST URL: <code>{apiUrl('/api/tts')}</code>
        </p>
        <div className="dev-tts-actions">
          <button type="button" disabled={warmupLoading} onClick={() => void handleWarmup()}>
            {warmupLoading ? '모델 로딩 중...' : 'POST /api/tts/warmup'}
          </button>
          <button type="button" disabled={warmupLoading} onClick={() => void refreshTtsStatus()}>
            상태 새로고침
          </button>
        </div>
        {ttsStatus?.modelLoaded && <p>modelLoaded=true — GPU worker 준비됨</p>}
        {warmupError && <p className="error-text">{warmupError}</p>}
        <pre>{ttsStatus ? JSON.stringify(ttsStatus, null, 2) : '로딩 중...'}</pre>
      </section>

      <section className="dev-info-card">
        <h2>POST /api/tts 테스트</h2>
        <p>
          Request URL: <code>{apiUrl('/api/tts')}</code>
        </p>
        <div className="dev-tts-actions">
          <input
            type="text"
            value={ttsTestText}
            onChange={(event) => setTtsTestText(event.target.value)}
            aria-label="TTS test text"
          />
          <button type="button" disabled={ttsTestLoading} onClick={() => void handleTtsTest()}>
            {ttsTestLoading ? '합성 중...' : 'POST /api/tts'}
          </button>
          {ttsTestResult?.audioUrl && (
            <button type="button" onClick={() => void handlePlayTestAudio()}>
              audioUrl 재생 테스트
            </button>
          )}
        </div>
        {ttsTestError && <p className="error-text">{ttsTestError}</p>}
        {ttsTestResult && (
          <pre>{JSON.stringify(ttsTestResult, null, 2)}</pre>
        )}
        {ttsTestResult?.audioUrl && (
          <p>
            resolved audioUrl: <code>{resolveMediaUrl(ttsTestResult.audioUrl)}</code>
          </p>
        )}
        {audioPlayResult && <p>{audioPlayResult}</p>}
      </section>

      <section className="dev-info-card">
        <h2>TTS Client Debug</h2>
        <pre>{JSON.stringify(ttsDebug, null, 2)}</pre>
        {ttsDebug.lastError && (
          <p className="error-text">마지막 /api/tts 실패: {ttsDebug.lastError}</p>
        )}
        {ttsDebug.usedFallback && (
          <p>브라우저 Web Speech API fallback 사용됨</p>
        )}
      </section>

      <section className="dev-info-card">
        <h2>/api/avatar/config</h2>
        <pre>{avatarConfig ? JSON.stringify(avatarConfig, null, 2) : '로딩 중...'}</pre>
      </section>

      <section className="dev-info-card">
        <h2>Avatar Runtime Debug</h2>
        <pre>{avatarDebug ? JSON.stringify(avatarDebug, null, 2) : '아바타 미연결'}</pre>
      </section>

      <div className="avatar-voice-row">
        <AiHumanAvatar
          isSpeaking={isSpeaking}
          isLoading={loading}
          counselorConnect={counselorConnect}
          mode="dev"
          onDebugUpdate={setAvatarDebug}
        />
        <VoiceControls
          variant="dev"
          voiceEnabled={voiceEnabled}
          autoSpeak={autoSpeak}
          speechSupported={speechSupported}
          isSpeaking={isSpeaking}
          canReplay={Boolean(lastAssistantMessage.trim())}
          onVoiceEnabledChange={(enabled) => {
            setVoiceEnabled(enabled);
            if (!enabled) {
              stopSpeaking();
              setIsSpeaking(false);
            }
          }}
          onAutoSpeakChange={setAutoSpeak}
          onStop={() => {
            stopSpeaking();
            setIsSpeaking(false);
          }}
          onReplay={() => void replayLast()}
        />
      </div>

      <section className="quick-tests">
        <p>공통 테스트 입력</p>
        <div className="quick-test-buttons">
          {TEST_INPUTS.map((sample) => (
            <button
              key={sample}
              type="button"
              className="quick-test-button"
              disabled={loading}
              onClick={() => void runSingleTest(sample)}
            >
              {sample}
            </button>
          ))}
        </div>
        <button
          type="button"
          className="batch-test-button"
          disabled={loading}
          onClick={() => setCounselorConnect(true)}
        >
          counselorConnect 상태 테스트
        </button>
      </section>

      <main className="chat-panel">
        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            showDevDetails
            onConnectBridge={() => setCounselorConnect(true)}
          />
        ))}
      </main>
    </div>
  );
}
