import React, { useState, useEffect } from 'react';
import { getTags, getIssueArticles, getArticleContent, getPDFLink, getVolumes, getVolumeIssues } from '../services/api';
import { getTranslation } from '../services/translations';

// ============================================================================
// Component
// ============================================================================

const TagBrowse = ({ language }) => {
  const t = (key) => getTranslation(language, key);

  // State
  const [tags, setTags] = useState([]);
  const [volumes, setVolumes] = useState([]);
  const [volumeIssues, setVolumeIssues] = useState({});
  const [expandedVolume, setExpandedVolume] = useState(null);
  const [selectedIssue, setSelectedIssue] = useState(null);
  const [selectedArticle, setSelectedArticle] = useState(null);
  const [articles, setArticles] = useState([]);
  const [articleContent, setArticleContent] = useState(null);
  const [categoryFilter, setCategoryFilter] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [articlesLoading, setArticlesLoading] = useState(false);
  const [contentLoading, setContentLoading] = useState(false);
  const [pdfUrl, setPdfUrl] = useState(null);

  // Load tags and volumes on mount
  useEffect(() => {
    getTags()
      .then((data) => {
        if (data.success) setTags(data.tags);
      })
      .catch((err) => console.error('Failed to load tags:', err));
    getVolumes()
      .then((data) => setVolumes(data))
      .catch((err) => console.error('Failed to load volumes:', err));
  }, []);

  // Load issues when a volume is expanded
  useEffect(() => {
    if (expandedVolume && !volumeIssues[expandedVolume]) {
      getVolumeIssues(expandedVolume)
        .then((data) => setVolumeIssues((prev) => ({ ...prev, [expandedVolume]: data })))
        .catch((err) => console.error('Failed to load issues:', err));
    }
  }, [expandedVolume, volumeIssues]);

  // Handle issue selection
  const handleSelectIssue = async (volumeId, issueIdx) => {
    setSelectedIssue({ volumeId, issueIdx });
    setSelectedArticle(null);
    setArticleContent(null);
    setArticles([]);
    setArticlesLoading(true);
    setPdfUrl(null);

    try {
      const [articlesData, pdfData] = await Promise.all([
        getIssueArticles(volumeId, issueIdx),
        getPDFLink(volumeId, issueIdx),
      ]);
      if (articlesData.success) setArticles(articlesData.articles);
      if (pdfData.found) setPdfUrl(pdfData.pdf_url);
    } catch (err) {
      console.error('Failed to load issue articles:', err);
    } finally {
      setArticlesLoading(false);
    }
  };

  // Handle article selection
  const handleSelectArticle = async (article) => {
    setSelectedArticle(article);
    setContentLoading(true);
    try {
      const data = await getArticleContent(article.doc_id, article.doc_issue, article.article_no);
      if (data.success) setArticleContent(data);
    } catch (err) {
      console.error('Failed to load article content:', err);
    } finally {
      setContentLoading(false);
    }
  };

  // Resolve tag ID → display name based on language
  const getTagName = (tagId) => {
    const info = tags.find((t) => t.id === tagId);
    if (!info) return tagId;
    return language === 'ta' ? info.tamil : info.english;
  };

  // Filter articles client-side
  const filteredArticles = articles.filter((article) => {
    if (categoryFilter !== 'ALL' && !(article.tags || []).includes(categoryFilter)) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      return (
        (article.title || '').toLowerCase().includes(q) ||
        (article.author_name || '').toLowerCase().includes(q)
      );
    }
    return true;
  });

  // Get cover image URL for an issue from API data
  const getIssueCoverUrl = (volumeId, issueNum) => {
    const issues = volumeIssues[volumeId] || [];
    const issue = issues.find((i) => i.issue_number === issueNum);
    return issue?.cover_image_url || null;
  };

  // Placeholder SVG for missing thumbnails
  const thumbPlaceholder = (label) =>
    `data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="40" height="48" viewBox="0 0 40 48"%3E%3Crect fill="%23f1f5f9" width="40" height="48" rx="4"/%3E%3Ctext fill="%2394a3b8" font-family="Inter,sans-serif" font-size="8" text-anchor="middle" x="20" y="28"%3E${encodeURIComponent(label)}%3C/text%3E%3C/svg%3E`;

  // ---- Render right pane content ----
  const renderContent = () => {
    // State 3: Article detail
    if (selectedArticle) {
      if (contentLoading) {
        return (
          <div className="tags-loading">
            <div className="spinner"></div>
            <span>{t('searching')}</span>
          </div>
        );
      }

      if (!articleContent) {
        return (
          <div className="tags-empty-state">
            <div className="tags-empty-text">{t('tags_no_articles')}</div>
          </div>
        );
      }

      return (
        <div className="tags-content-animate" key="detail">
          <button
            className="tags-detail-back"
            onClick={() => { setSelectedArticle(null); setArticleContent(null); }}
          >
            &larr; {t('tags_back_to_articles')}
          </button>

          {(articleContent.tags || []).length > 0 && (
            <div className="tags-badge-row">
              {articleContent.tags.map((tag) => (
                <span key={tag} className="tag-badge">{getTagName(tag)}</span>
              ))}
            </div>
          )}

          <h1 className="tags-detail-title">
            {articleContent.title || t('untitled')}
          </h1>
          <div className="tags-detail-title-rule" />

          <div className="tags-detail-content">
            {articleContent.content}
          </div>

          <div className="tags-detail-meta">
            {articleContent.author_name && (
              <span>{t('author_label')}: {articleContent.author_name}</span>
            )}
            {articleContent.doc_issue && (
              <span>{t('issue_label')}: {articleContent.doc_issue}</span>
            )}
            {articleContent.year && <span>{articleContent.year}</span>}
          </div>

          <div className="tags-stats-row">
            <div className="tags-stat-card">
              <div className="tags-stat-value">
                {selectedIssue ? selectedIssue.volumeId : '-'}
              </div>
              <div className="tags-stat-label">{t('tags_volume_label')}</div>
            </div>
            <div className="tags-stat-card">
              <div className="tags-stat-value">{t('browse_tags')}</div>
              <div className="tags-stat-label">{t('tags_collection')}</div>
            </div>
            <div className="tags-stat-card">
              <div className="tags-stat-value">{articleContent.word_count || '-'}</div>
              <div className="tags-stat-label">{t('tags_word_count')}</div>
            </div>
          </div>
        </div>
      );
    }

    // State 2: Issue selected — show articles list
    if (selectedIssue) {
      if (articlesLoading) {
        return (
          <div className="tags-loading">
            <div className="spinner"></div>
            <span>{t('searching')}</span>
          </div>
        );
      }

      const coverPath = getIssueCoverUrl(selectedIssue.volumeId, selectedIssue.issueIdx);

      return (
        <div className="tags-content-animate" key={`issue-${selectedIssue.volumeId}-${selectedIssue.issueIdx}`}>
          <div className="tags-badge-row">
            <span className="tags-badge-vol">
              {t('tags_volume_label')} {selectedIssue.volumeId}
            </span>
            <span className="tags-badge-issue">
              {t('issue_label')} {selectedIssue.issueIdx}
            </span>
          </div>

          <div className="tags-issue-header">
            {coverPath && (
              <img
                className="tags-cover-image"
                src={coverPath}
                alt={`${t('issue_label')} ${selectedIssue.issueIdx}`}
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            )}
            <div className="tags-issue-meta-col">
              {pdfUrl && (
                <a
                  href={pdfUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="tags-pdf-btn"
                >
                  {t('tags_read_pdf')}
                </a>
              )}
            </div>
          </div>

          <hr className="tags-divider" />

          {filteredArticles.length === 0 ? (
            <div className="tags-empty-state" style={{ height: 'auto', padding: '2rem 0' }}>
              <div className="tags-empty-text">{t('tags_no_articles')}</div>
            </div>
          ) : (
            <div className="tags-articles-grid">
              {filteredArticles.map((article, idx) => (
                <div
                  key={`${article.doc_id}-${article.doc_issue}-${article.article_no}-${idx}`}
                  className="tags-article-block"
                  style={{ animationDelay: `${Math.min(idx * 0.04, 0.6)}s` }}
                  onClick={() => handleSelectArticle(article)}
                >
                  <div className="tags-article-block-title">
                    {article.title || t('untitled')}
                  </div>
                  {article.author_name && (
                    <div className="tags-article-block-author">
                      {article.author_name}
                    </div>
                  )}
                  {(article.tags || []).length > 0 && (
                    <div className="tags-article-block-tags">
                      {article.tags.map((tag) => (
                        <span key={tag} className="tag-badge">{getTagName(tag)}</span>
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

    // State 1: Empty — no issue selected
    return (
      <div className="tags-empty-state">
        <div className="tags-empty-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
            <line x1="8" y1="7" x2="16" y2="7" />
            <line x1="8" y1="11" x2="13" y2="11" />
          </svg>
        </div>
        <div className="tags-empty-text">{t('tags_select_issue')}</div>
        <div className="tags-empty-hint">
          {language === 'ta'
            ? 'இடது பக்கத்தில் ஒரு தொகுதியை விரிவாக்கி, ஒரு இதழை உருப்படியிடுக'
            : 'Expand a volume on the left and click an issue to browse its articles'}
        </div>
      </div>
    );
  };

  // ---- Main Render ----
  return (
    <div className="tags-page-container">
      {/* Left Sidebar */}
      <div className="tags-sidebar">
        <div className="tags-sidebar-header">
          <div className="tags-sidebar-title">{t('tags_library_nav')}</div>

          {/* Category filter dropdown */}
          <select
            className="tags-sidebar-select"
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
          >
            <option value="ALL">{t('tags_all_categories')}</option>
            {tags.map((tag) => (
              <option key={tag.id} value={tag.id}>
                {language === 'ta' ? tag.tamil : tag.english} ({tag.count})
              </option>
            ))}
          </select>

          {/* Search input */}
          <div className="tags-sidebar-search">
            <span className="tags-sidebar-search-icon">&#128269;</span>
            <input
              type="text"
              placeholder={t('tags_search_placeholder')}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          {/* Results count (only when issue is selected) */}
          {selectedIssue && !selectedArticle && (
            <div className="tags-result-count">
              {filteredArticles.length} {t('tags_results_found')}
            </div>
          )}
        </div>

        {/* Volume accordions */}
        <div className="tags-volumes-list">
          {volumes.map((volume) => {
            const issues = volumeIssues[volume.id] || [];
            const isExpanded = expandedVolume === volume.id;

            return (
              <div key={volume.id} className="tags-volume-accordion">
                <div
                  className={`tags-volume-header ${isExpanded ? 'expanded' : ''}`}
                  onClick={() => setExpandedVolume(isExpanded ? null : volume.id)}
                >
                  <span>
                    <span className="tags-volume-label">
                      {t('tags_volume_label')} {volume.id}
                    </span>
                    <span className="tags-volume-year">{volume.year}</span>
                    <span className="tags-volume-count">
                      &middot; {volume.issue_count} {t('tags_files')}
                    </span>
                  </span>
                  <span className={`tags-volume-chevron ${isExpanded ? 'expanded' : ''}`}>
                    &#9654;
                  </span>
                </div>

                {isExpanded && (
                  <div className="tags-issue-list">
                    {issues.map((issue) => {
                      const isActive =
                        selectedIssue &&
                        selectedIssue.volumeId === volume.id &&
                        selectedIssue.issueIdx === issue.issue_number;

                      return (
                        <div
                          key={issue.issue_number}
                          className={`tags-issue-item ${isActive ? 'active' : ''}`}
                          onClick={() => handleSelectIssue(volume.id, issue.issue_number)}
                        >
                          <img
                            className="tags-issue-thumb"
                            src={issue.cover_image_url || thumbPlaceholder(String(issue.issue_number))}
                            alt={`${t('issue_label')} ${issue.issue_number}`}
                            onError={(e) => {
                              e.target.src = thumbPlaceholder(String(issue.issue_number));
                            }}
                            loading="lazy"
                          />
                          <span className="tags-issue-label">
                            {t('issue_label')} {issue.issue_number}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Right Content Pane */}
      <div className="tags-content">
        {renderContent()}
      </div>
    </div>
  );
};

export default TagBrowse;
