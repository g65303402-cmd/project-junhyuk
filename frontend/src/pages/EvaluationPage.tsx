import { Link } from 'react-router-dom';
import { useState } from 'react';
import { runEvaluation, type EvaluationResult } from '../api/client';
import { MessageBubble } from '../components/MessageBubble';
import type { ChatMessage } from '../types/chat';

export function EvaluationPage() {
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setLoading(true);
    setError(null);
    try {
      const data = await runEvaluation();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : '평가 실행 실패');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="evaluation-page app-shell">
      <header className="app-header">
        <div>
          <h1>평가 패널</h1>
          <p>백엔드 `/api/evaluation/run` 결과 (발표/개발용)</p>
        </div>
        <div className="dev-header-links">
          <Link to="/">사용자 화면</Link>
          <Link to="/dev">개발자 도구</Link>
        </div>
      </header>

      <button
        type="button"
        className="batch-test-button"
        disabled={loading}
        onClick={() => void handleRun()}
      >
        {loading ? '평가 실행 중...' : '공통 테스트 5개 일괄 실행'}
      </button>

      {error && <p className="error-text">{error}</p>}

      {result && (
        <>
          <div className="evaluation-grid">
            <div className="evaluation-card">
              <span className="evaluation-label">총 테스트 수</span>
              <strong>{result.total}</strong>
            </div>
            <div className="evaluation-card">
              <span className="evaluation-label">스타일 통과율</span>
              <strong>{result.stylePassRate}%</strong>
            </div>
            <div className="evaluation-card">
              <span className="evaluation-label">질문 1개</span>
              <strong>{result.questionOneRate}%</strong>
            </div>
            <div className="evaluation-card">
              <span className="evaluation-label">존댓말 없음</span>
              <strong>{result.noHonorificRate}%</strong>
            </div>
            <div className="evaluation-card">
              <span className="evaluation-label">외국어 혼입 없음</span>
              <strong>{result.noForeignRate}%</strong>
            </div>
            <div className="evaluation-card">
              <span className="evaluation-label">3문장 이상</span>
              <strong>{result.threeSentenceRate}%</strong>
            </div>
            <div className="evaluation-card">
              <span className="evaluation-label">브릿지 발동</span>
              <strong>{result.bridgeCount}건</strong>
            </div>
          </div>

          <table className="emotion-table">
            <thead>
              <tr>
                <th>emotion</th>
                <th>건수</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.emotionCounts).map(([emotion, count]) => (
                <tr key={emotion}>
                  <td>{emotion}</td>
                  <td>{count}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <section className="evaluation-items">
            <h2>항목별 결과</h2>
            {result.items.map((item) => {
              const message: ChatMessage = {
                id: item.input,
                role: 'assistant',
                text: item.message,
                emotion: item.emotion as ChatMessage['emotion'],
                needsBridge: item.needsBridge,
                styleCheck: item.styleCheck as unknown as ChatMessage['styleCheck'],
                debug: item.debug as unknown as ChatMessage['debug'],
              };
              return (
                <div key={item.input} className="evaluation-item-block">
                  <p className="evaluation-item-input">입력: {item.input}</p>
                  <MessageBubble message={message} showDevDetails />
                </div>
              );
            })}
          </section>
        </>
      )}
    </div>
  );
}
