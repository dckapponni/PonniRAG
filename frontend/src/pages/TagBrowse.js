import React, { useState, useEffect } from 'react';
import { getTags, getTagArticles } from '../services/api';
import { getTranslation } from '../services/translations';

const TagBrowse = ({ language }) => {
  const [tags, setTags] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedTag, setSelectedTag] = useState(null);
  const [articles, setArticles] = useState([]);
  const [articlesLoading, setArticlesLoading] = useState(false);
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

  const handleTagClick = async (tag) => {
    setSelectedTag(tag);
    setArticlesLoading(true);
    try {
      const data = await getTagArticles(tag.id);
      if (data.success) {
        setArticles(data.articles);
      }
    } catch (err) {
      console.error('Failed to load tag articles:', err);
    } finally {
      setArticlesLoading(false);
    }
  };

  const handleBack = () => {
    setSelectedTag(null);
    setArticles([]);
  };

  if (loading) {
    return (
      <div className="library-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <span>{t('searching')}</span>
        </div>
      </div>
    );
  }

  // Article list view for a selected tag
  if (selectedTag) {
    return (
      <div className="library-container">
        <button className="back-btn" onClick={handleBack}>
          {t('back_to_tags')}
        </button>
        <h1 className="library-title">
          {language === 'ta' ? selectedTag.tamil : selectedTag.english}
        </h1>
        <p className="library-desc">
          {t('articles_in_category')}: {articles.length}
        </p>
        {articlesLoading ? (
          <div className="loading-spinner">
            <div className="spinner"></div>
            <span>{t('searching')}</span>
          </div>
        ) : (
          <div className="tag-articles-list">
            {articles.map((article, idx) => (
              <div key={idx} className="tag-article-card">
                <div className="tag-article-title">
                  {article.title || t('untitled')}
                </div>
                <div className="tag-article-meta">
                  {article.author_name && (
                    <span>{t('author_label')}: {article.author_name}</span>
                  )}
                  {article.doc_issue && (
                    <span>{t('issue_label')}: {article.doc_issue}</span>
                  )}
                  {article.year && <span>{article.year}</span>}
                </div>
                {article.tags && article.tags.length > 0 && (
                  <div className="source-tags">
                    {article.tags.map((tag) => (
                      <span key={tag} className="tag-badge">{tag}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Tag grid view
  return (
    <div className="library-container">
      <h1 className="library-title">{t('browse_tags')}</h1>
      <p className="library-desc">{t('browse_tags_desc')}</p>
      <div className="tags-grid">
        {tags.map((tag) => (
          <div
            key={tag.id}
            className="tag-card"
            onClick={() => handleTagClick(tag)}
          >
            <div className="tag-card-name">
              {language === 'ta' ? tag.tamil : tag.english}
            </div>
            <div className="tag-card-count">
              {tag.count} {t('articles_count')}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TagBrowse;
