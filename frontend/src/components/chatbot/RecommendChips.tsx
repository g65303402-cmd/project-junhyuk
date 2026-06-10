type RecommendChipsProps = {
  questions: string[];
  expanded: boolean;
  onToggleExpand: () => void;
  onSelect: (label: string) => void;
};

export function RecommendChips({
  questions,
  expanded,
  onToggleExpand,
  onSelect,
}: RecommendChipsProps) {
  const visible = expanded ? questions : questions.slice(0, 5);

  return (
    <section className="chatbot-recommend" aria-label="추천 질문">
      <h2 className="chatbot-recommend__title">추천질문</h2>
      <div className="chatbot-recommend__chips">
        {visible.map((question) => (
          <button
            key={question}
            type="button"
            className="chatbot-chip"
            onClick={() => onSelect(question)}
          >
            {question}
          </button>
        ))}
        {questions.length > 5 && (
          <button type="button" className="chatbot-chip chatbot-chip--more" onClick={onToggleExpand}>
            {expanded ? '접기' : '더보기'}
          </button>
        )}
      </div>
    </section>
  );
}
