import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { getTranslation } from '../services/translations';

const SourceCard = ({ source, index, language }) => {
  const [expanded, setExpanded] = useState(false);
  const t = (key) => getTranslation(language, key);

  const docIssue = source.doc_issue || source.payload?.metadata?.doc_issue;
  const volume = source.volume || source.payload?.metadata?.volume;
  const heading = source.heading || source.payload?.metadata?.heading;
  const author = source.author_name || source.payload?.metadata?.author_name;
  const content = source.content || source.payload?.content || '';
  const tags = source.tags || source.payload?.metadata?.tags || [];

  const metaParts = [];
  if (docIssue) metaParts.push(`${t('issue_label')}: ${docIssue}`);
  if (volume) metaParts.push(`${t('malar_label')}: ${volume}`);
  if (heading) metaParts.push(`${t('title_label')}: ${heading}`);
  if (author) metaParts.push(`${t('author_label')}: ${author}`);

  const isLongContent = content.length > 300;
  const displayContent = expanded ? content : content.slice(0, 300) + (isLongContent ? '...' : '');

  return (
    <div className="source-card">
      <div className="source-header">
        <span className="source-title">{t('source_title')} {index + 1}</span>
        {tags.length > 0 && (
          <div className="source-tags">
            {tags.map((tag) => (
              <span key={tag} className="tag-badge">{tag}</span>
            ))}
          </div>
        )}
      </div>
      {metaParts.length > 0 && (
        <div className="source-meta">{metaParts.join(' • ')}</div>
      )}
      <div className="source-preview">{displayContent}</div>
      {isLongContent && (
        <button className="read-more-btn" onClick={() => setExpanded(!expanded)}>
          {expanded ? t('show_less') : t('read_more')}
        </button>
      )}
    </div>
  );
};

const Sources = ({ sources, language }) => {
  const [isOpen, setIsOpen] = useState(false);
  const t = (key) => getTranslation(language, key);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources-container">
      <div
        className="sources-header"
        onClick={() => setIsOpen(!isOpen)}
        style={{ cursor: 'pointer', userSelect: 'none' }}
        role="button"
        aria-expanded={isOpen}
      >
        <span>{t('sources_title')} — {sources.length}</span>
        <span>{isOpen ? '▲' : '▼'}</span>
      </div>
      {isOpen && (
        <div className="sources-list">
          {sources.map((source, index) => (
            <SourceCard
              key={index}
              source={source}
              index={index}
              language={language}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const ChatMessage = ({ message, language }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`chat-message ${isUser ? 'user' : 'assistant'}`}>
      <div className={`chat-avatar ${isUser ? 'user' : 'assistant'}`}>
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="chat-message-content">
        {isUser ? (
          <div>{message.content}</div>
        ) : (
          <ReactMarkdown>{message.content}</ReactMarkdown>
        )}
        {!isUser && message.sources && (
          <Sources sources={message.sources} language={language} />
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
