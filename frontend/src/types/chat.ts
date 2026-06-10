export type EmotionTag = 'comfort' | 'serious' | 'encourage' | 'happy' | 'neutral';

export type StyleCheckValue = boolean | string;

export interface StyleCheck {
  '반말·존댓말 없음': boolean;
  '질문 1개': boolean;
  '금지 표현 없음': boolean;
  '외국어 혼입 없음': boolean;
  '인정 표현 사용': boolean;
  '공감 표현 있음': boolean;
  '3문장 이상': boolean;
  '감정태그 유효': boolean;
  '브릿지 포함': StyleCheckValue;
  '⚠️ 어색 패턴': string;
}

export interface ChatDebug {
  mockMode: boolean;
  bridgeTriggered: boolean;
  responsePath: 'mock' | 'model';
}

export interface ChatResponse {
  message: string;
  emotion: EmotionTag;
  audioUrl?: string | null;
  needsBridge: boolean;
  styleCheck?: StyleCheck;
  debug: ChatDebug;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'assistant-loading' | 'error';
  text: string;
  emotion?: EmotionTag;
  needsBridge?: boolean;
  styleCheck?: StyleCheck;
  debug?: ChatDebug;
}

export const TEST_INPUTS = [
  '요즘 회사 가기가 너무 싫어. 아침마다 배가 아파',
  '친구한테 상처받았는데 화내는 내가 이상한 건가',
  '아무것도 하기 싫고 그냥 누워만 있고 싶어',
  '취업이 너무 안 돼서 나만 뒤처지는 것 같아',
  '가족이랑 크게 싸웠어. 내가 너무 예민한 건지 모르겠어',
] as const;

export const EMOTION_LABELS: Record<EmotionTag, string> = {
  comfort: '위로',
  serious: '진지',
  encourage: '격려',
  happy: '기쁨',
  neutral: '중립',
};

export interface EvaluationStats {
  totalTests: number;
  stylePassRate: number;
  singleQuestionPassRate: number;
  honorificFreeRate: number;
  noForeignRate: number;
  minThreeSentencesRate: number;
  emotionDistribution: Record<EmotionTag, number>;
  bridgeCount: number;
}

export function computeEvaluationStats(messages: ChatMessage[]): EvaluationStats {
  const assistantMessages = messages.filter(
    (message) => message.role === 'assistant' && message.styleCheck,
  );

  const totalTests = assistantMessages.length;
  if (totalTests === 0) {
    return {
      totalTests: 0,
      stylePassRate: 0,
      singleQuestionPassRate: 0,
      honorificFreeRate: 0,
      noForeignRate: 0,
      minThreeSentencesRate: 0,
      emotionDistribution: {
        comfort: 0,
        serious: 0,
        encourage: 0,
        happy: 0,
        neutral: 0,
      },
      bridgeCount: 0,
    };
  }

  const booleanKeys: Array<keyof StyleCheck> = [
    '반말·존댓말 없음',
    '질문 1개',
    '금지 표현 없음',
    '외국어 혼입 없음',
    '인정 표현 사용',
    '공감 표현 있음',
    '3문장 이상',
    '감정태그 유효',
  ];

  let stylePassSum = 0;
  let stylePassTotal = 0;
  let singleQuestionPass = 0;
  let honorificFree = 0;
  let noForeign = 0;
  let minThreeSentences = 0;
  let bridgeCount = 0;

  const emotionDistribution: Record<EmotionTag, number> = {
    comfort: 0,
    serious: 0,
    encourage: 0,
    happy: 0,
    neutral: 0,
  };

  for (const message of assistantMessages) {
    const style = message.styleCheck!;
    for (const key of booleanKeys) {
      if (typeof style[key] === 'boolean') {
        stylePassSum += style[key] ? 1 : 0;
        stylePassTotal += 1;
      }
    }
    if (style['질문 1개']) singleQuestionPass += 1;
    if (style['반말·존댓말 없음']) honorificFree += 1;
    if (style['외국어 혼입 없음']) noForeign += 1;
    if (style['3문장 이상']) minThreeSentences += 1;
    if (message.needsBridge) bridgeCount += 1;
    if (message.emotion) emotionDistribution[message.emotion] += 1;
  }

  return {
    totalTests,
    stylePassRate: stylePassTotal ? (stylePassSum / stylePassTotal) * 100 : 0,
    singleQuestionPassRate: (singleQuestionPass / totalTests) * 100,
    honorificFreeRate: (honorificFree / totalTests) * 100,
    noForeignRate: (noForeign / totalTests) * 100,
    minThreeSentencesRate: (minThreeSentences / totalTests) * 100,
    emotionDistribution,
    bridgeCount,
  };
}

function pct(value: number): string {
  return `${value.toFixed(0)}%`;
}

export function formatEvaluationStats(stats: EvaluationStats) {
  return {
    totalTests: stats.totalTests,
    stylePassRate: pct(stats.stylePassRate),
    singleQuestionPassRate: pct(stats.singleQuestionPassRate),
    honorificFreeRate: pct(stats.honorificFreeRate),
    noForeignRate: pct(stats.noForeignRate),
    minThreeSentencesRate: pct(stats.minThreeSentencesRate),
    emotionDistribution: stats.emotionDistribution,
    bridgeCount: stats.bridgeCount,
  };
}
