import type { EmotionTag } from '../types/chat';
import { EMOTION_LABELS } from '../types/chat';

interface EmotionBadgeProps {
  emotion: EmotionTag;
  emphasized?: boolean;
}

export function EmotionBadge({ emotion, emphasized = false }: EmotionBadgeProps) {
  return (
    <span
      className={`emotion-badge emotion-${emotion}${emphasized ? ' emotion-badge--alert' : ''}`}
    >
      {EMOTION_LABELS[emotion]}
    </span>
  );
}
