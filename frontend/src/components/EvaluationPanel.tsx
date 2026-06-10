import { useState } from 'react';
import {
  computeEvaluationStats,
  formatEvaluationStats,
  type ChatMessage,
} from '../types/chat';

interface EvaluationPanelProps {
  messages: ChatMessage[];
  onRunBatchTests: () => void;
  batchRunning: boolean;
}

export function EvaluationPanel({ messages, onRunBatchTests, batchRunning }: EvaluationPanelProps) {
  const [open, setOpen] = useState(true);
  const stats = formatEvaluationStats(computeEvaluationStats(messages));

  return (
    <section className="evaluation-panel">
      <div className="evaluation-header">
        <div>
          <h2>평가 패널</h2>
          <p>현재 세션 응답 기준 (발표/개발용)</p>
        </div>
        <button type="button" className="panel-toggle" onClick={() => setOpen((prev) => !prev)}>
          {open ? '접기' : '펼치기'}
        </button>
      </div>

      {open && (
        <>
          <div className="evaluation-actions">
            <button
              type="button"
              className="batch-test-button"
              disabled={batchRunning}
              onClick={onRunBatchTests}
            >
              {batchRunning ? '5개 테스트 실행 중...' : '공통 테스트 5개 일괄 실행'}
            </button>
          </div>

          <div className="evaluation-grid">
            <div className="evaluation-card">
              <span className="evaluation-label">총 테스트 수</span>
              <strong>{stats.totalTests}</strong>
            </div>
            <div className="evaluation-card">
              <span className="evaluation-label">스타일 통과율</span>
              <strong>{stats.stylePassRate}</strong>
            </div>
            <div className="evaluation-card">
              <span className="evaluation-label">질문 1개</span>
              <strong>{stats.singleQuestionPassRate}</strong>
            </div>
            <div className="evaluation-card">
              <span className="evaluation-label">존댓말 없음</span>
              <strong>{stats.honorificFreeRate}</strong>
            </div>
            <div className="evaluation-card">
              <span className="evaluation-label">외국어 혼입 없음</span>
              <strong>{stats.noForeignRate}</strong>
            </div>
            <div className="evaluation-card">
              <span className="evaluation-label">3문장 이상</span>
              <strong>{stats.minThreeSentencesRate}</strong>
            </div>
            <div className="evaluation-card">
              <span className="evaluation-label">브릿지 발동</span>
              <strong>{stats.bridgeCount}건</strong>
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
              {Object.entries(stats.emotionDistribution).map(([emotion, count]) => (
                <tr key={emotion}>
                  <td>{emotion}</td>
                  <td>{count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
