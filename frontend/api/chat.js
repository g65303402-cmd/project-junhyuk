const BRIDGE_TEXT =
  '나한테 털어놓는 것도 좋은데, 이 정도면 준혁쌤이랑 직접 얘기해보는 것도 좋을 것 같아. 부담 없이 한번 연결해볼까?';

const BRIDGE_TRIGGERS = [
  '자해',
  '자살',
  '죽고 싶',
  '사라지고 싶',
  '없어지고 싶',
  '살기 싫',
  '어떻게 해야',
  '어쩌지',
  '방법 알려',
];

const MOCK_RESPONSES = {
  '요즘 회사 가기가 너무 싫어. 아침마다 배가 아파':
    '그게 쉽지 않았겠다.\n아침마다 몸이 먼저 반응할 만큼 지쳐있는 거잖아.\n요즘 뭐가 제일 버거워?',
  '친구한테 상처받았는데 화내는 내가 이상한 건가':
    '충분히 그럴 수 있어.\n친한 사람한테 상처받으면 그 말이 더 오래 남잖아.\n어떤 말이 제일 마음에 걸렸어?',
  '아무것도 하기 싫고 그냥 누워만 있고 싶어':
    '많이 힘들었겠다.\n몸도 마음도 다 내려놓고 싶은 거잖아.\n언제부터 그런 느낌이 시작된 것 같아?',
  '취업이 너무 안 돼서 나만 뒤처지는 것 같아':
    '그 마음 이해해.\n열심히 해도 결과가 안 나오면 주변만 커 보이고 내가 작아지는 느낌이 들잖아.\n지금 가장 막막하게 느껴지는 게 뭐야?',
  '가족이랑 크게 싸웠어. 내가 너무 예민한 건지 모르겠어':
    '그랬구나.\n가까운 사람이랑 부딪히면 내 감정까지 헷갈리게 되는 거잖아.\n싸우기 전에 어떤 말이 제일 마음에 걸렸어?',
};

function needsBridge(text) {
  return BRIDGE_TRIGGERS.some((k) => text.includes(k));
}

function assignEmotion(text) {
  if (['자해', '자살', '죽고 싶', '사라지고 싶', '없어지고 싶', '살기 싫'].some((k) => text.includes(k))) {
    return 'serious';
  }
  if (['취업', '진로', '면접', '뒤처', '자신감'].some((k) => text.includes(k))) {
    return 'encourage';
  }
  if (['친구', '상처', '가족', '외로', '슬', '싸웠'].some((k) => text.includes(k))) {
    return 'comfort';
  }
  return 'neutral';
}

function genericResponse(text, withBridge) {
  if (withBridge) {
    return `많이 힘들었겠다.\n혼자 감당하고 있는 무게가 느껴져.\n${BRIDGE_TEXT}`;
  }
  if (['회사', '출근', '일'].some((k) => text.includes(k))) {
    return '그게 쉽지 않았겠다.\n매일 가기 싫은 마음이 쌓이면 몸도 같이 지치는 거잖아.\n요즘 제일 버거운 순간이 언제야?';
  }
  return '많이 힘들었겠다.\n혼자 감당하고 있는 무게가 느껴져.\n지금 가장 바라는 게 뭐야?';
}

function mockResponse(text, withBridge) {
  const trimmed = text.trim();
  if (MOCK_RESPONSES[trimmed]) {
    const base = MOCK_RESPONSES[trimmed];
    if (withBridge) {
      const lines = base.split('\n');
      return lines.length >= 2 ? `${lines[0]}\n${lines[1]}\n${BRIDGE_TEXT}` : `${base}\n${BRIDGE_TEXT}`;
    }
    return base;
  }
  return genericResponse(trimmed, withBridge);
}

function setCors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

export default function handler(req, res) {
  setCors(res);
  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }
  if (req.method !== 'POST') {
    res.status(405).json({ detail: 'Method not allowed' });
    return;
  }

  const message = typeof req.body?.message === 'string' ? req.body.message.trim() : '';
  if (!message) {
    res.status(400).json({ detail: 'message가 비어있어' });
    return;
  }

  const bridge = needsBridge(message);
  const emotion = bridge ? 'serious' : assignEmotion(message);
  const reply = mockResponse(message, bridge);

  res.status(200).json({
    message: reply,
    emotion,
    needsBridge: bridge,
    styleCheck: {},
    debug: {
      mockMode: true,
      bridgeTriggered: bridge,
      responsePath: 'vercel-mock',
    },
  });
}
