export type CategoryId = 'relationship' | 'career' | 'anxiety' | 'family';

export type CategoryItem = {
  id: CategoryId;
  label: string;
  icon: string;
};

export const CATEGORIES: CategoryItem[] = [
  { id: 'relationship', label: '관계', icon: '💬' },
  { id: 'career', label: '진로/취업', icon: '🧭' },
  { id: 'anxiety', label: '불안', icon: '🌙' },
  { id: 'family', label: '가족', icon: '🏠' },
];

export const QUESTIONS_BY_CATEGORY: Record<CategoryId | 'all', string[]> = {
  all: [
    '회사 가기 싫을 때',
    '친구한테 상처받았을 때',
    '취업이 막막할 때',
    '가족이랑 싸웠을 때',
    '잠이 안 올 때',
  ],
  relationship: [
    '친구한테 상처받았을 때',
    '사람들이 나를 어떻게 보는지 신경 쓰일 때',
    '연락이 뜸해져서 불안할 때',
  ],
  career: [
    '회사 가기 싫을 때',
    '취업이 막막할 때',
    '내가 뒤처지는 것 같을 때',
  ],
  anxiety: [
    '잠이 안 올 때',
    '아무것도 하기 싫을 때',
    '갑자기 숨이 가빠질 때',
  ],
  family: [
    '가족이랑 싸웠을 때',
    '집에 가기가 무거울 때',
    '가족이 나를 이해 못 하는 것 같을 때',
  ],
};

export const QUESTION_PROMPTS: Record<string, string> = {
  '회사 가기 싫을 때': '요즘 회사 가기가 너무 싫어. 아침마다 배가 아파.',
  '친구한테 상처받았을 때': '친구한테 상처받았는데, 화내는 내가 이상한 건지 모르겠어.',
  '취업이 막막할 때': '취업이 너무 안 돼서 나만 뒤처지는 것 같아.',
  '가족이랑 싸웠을 때': '가족이랑 크게 싸웠어. 내가 너무 예민한 건지 모르겠어.',
  '잠이 안 올 때': '잠이 안 와서 밤마다 생각이 너무 많아져.',
  '사람들이 나를 어떻게 보는지 신경 쓰일 때':
    '사람들 시선이 신경 쓰여서 작은 말도 계속 곱씹게 돼.',
  '연락이 뜸해져서 불안할 때': '연락이 뜸해지면 나한테 마음이 식은 건가 불안해져.',
  '내가 뒤처지는 것 같을 때': '주변은 다 잘하는 것 같은데 나만 제자리인 느낌이야.',
  '아무것도 하기 싫을 때': '아무것도 하기 싫고 그냥 누워만 있고 싶어.',
  '갑자기 숨이 가빠질 때': '갑자기 숨이 가빠지고 가슴이 답답해질 때가 있어.',
  '집에 가기가 무거울 때': '집에 가는 길이 항상 무겁게 느껴져.',
  '가족이 나를 이해 못 하는 것 같을 때':
    '가족이 나를 이해 못 하는 것 같아서 말하기가 더 어려워져.',
};

export function getPromptForQuestion(label: string): string {
  return QUESTION_PROMPTS[label] ?? label;
}
