import { useState } from 'react';
import type { StyleCheck } from '../types/chat';

interface StyleCheckPanelProps {
  styleCheck?: StyleCheck;
}

function formatValue(value: boolean | string): string {
  if (typeof value === 'boolean') {
    return value ? '통과' : '미통과';
  }
  return value;
}

export function StyleCheckPanel({ styleCheck }: StyleCheckPanelProps) {
  const [open, setOpen] = useState(false);

  if (!styleCheck) return null;

  return (
    <div className="style-check-panel">
      <button type="button" className="style-check-toggle" onClick={() => setOpen((prev) => !prev)}>
        스타일 체크 {open ? '접기' : '펼치기'}
      </button>
      {open && (
        <ul className="style-check-list">
          {Object.entries(styleCheck).map(([key, value]) => (
            <li key={key}>
              <span>{key}</span>
              <span className={typeof value === 'boolean' && value ? 'pass' : 'value'}>
                {formatValue(value)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
