import type { CategoryId, CategoryItem } from '../../data/recommendQuestions';

type CategoryCardsProps = {
  categories: CategoryItem[];
  activeCategory: CategoryId | null;
  onSelect: (categoryId: CategoryId) => void;
};

export function CategoryCards({ categories, activeCategory, onSelect }: CategoryCardsProps) {
  return (
    <div className="chatbot-categories" role="group" aria-label="상담 카테고리">
      {categories.map((category) => (
        <button
          key={category.id}
          type="button"
          className={`chatbot-category-card${activeCategory === category.id ? ' is-active' : ''}`}
          onClick={() => onSelect(category.id)}
        >
          <span className="chatbot-category-card__icon" aria-hidden="true">
            {category.icon}
          </span>
          <span className="chatbot-category-card__label">{category.label}</span>
        </button>
      ))}
    </div>
  );
}
