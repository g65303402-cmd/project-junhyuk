import { useState } from 'react';

interface BridgeCardProps {
  visible: boolean;
  onConnect?: () => void;
}

export function BridgeCard({ visible, onConnect }: BridgeCardProps) {
  const [connectNotice, setConnectNotice] = useState(false);

  if (!visible) return null;

  function handleConnect() {
    onConnect?.();
    setConnectNotice(true);
  }

  return (
    <div className="bridge-card">
      <strong>도움 연결 안내</strong>
      <p>
        나한테 털어놓는 것도 좋은데, 이 정도면 준혁쌤이랑 직접 얘기해보는 것도 좋을 것
        같아. 부담 없이 한번 연결해볼까?
      </p>
      <button type="button" className="bridge-connect-button" onClick={handleConnect}>
        준혁쌤과 연결하기
      </button>
      {connectNotice && (
        <p className="bridge-connect-notice">지금은 시연 단계라 실제 연결은 준비 중이에요.</p>
      )}
    </div>
  );
}
