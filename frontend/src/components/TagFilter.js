import React, { useState, useEffect } from 'react';
import { getTags } from '../services/api';
import { getTranslation } from '../services/translations';

const TagFilter = ({ language, selectedTags, onTagsChange }) => {
  const [tags, setTags] = useState([]);
  const [loading, setLoading] = useState(true);
  const t = (key) => getTranslation(language, key);

  useEffect(() => {
    getTags()
      .then((data) => {
        if (data.success) {
          setTags(data.tags);
        }
      })
      .catch((err) => console.error('Failed to load tags:', err))
      .finally(() => setLoading(false));
  }, []);

  const handleToggle = (tagId) => {
    const updated = selectedTags.includes(tagId)
      ? selectedTags.filter((t) => t !== tagId)
      : [...selectedTags, tagId];
    onTagsChange(updated);
  };

  if (loading) return null;
  if (tags.length === 0) return null;

  return (
    <div className="tag-filter">
      <div className="tag-filter-title">{t('filter_by_category')}</div>
      <div className="tag-filter-list">
        {tags.map((tag) => (
          <label key={tag.id} className="tag-filter-item">
            <input
              type="checkbox"
              checked={selectedTags.includes(tag.id)}
              onChange={() => handleToggle(tag.id)}
            />
            <span className="tag-filter-label">
              {language === 'ta' ? tag.tamil : tag.english}
            </span>
            <span className="tag-filter-count">({tag.count})</span>
          </label>
        ))}
      </div>
      {selectedTags.length > 0 && (
        <button
          className="tag-filter-clear"
          onClick={() => onTagsChange([])}
        >
          {t('clear_filters')}
        </button>
      )}
    </div>
  );
};

export default TagFilter;
