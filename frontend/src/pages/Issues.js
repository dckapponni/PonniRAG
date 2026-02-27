import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getTranslation } from '../services/translations';
import { VOLUME_IMAGES } from '../data/volumeImages';

const VOLUME_YEARS = { 1: '1947', 2: '1948', 3: '1949', 4: '1950', 5: '1951', 6: '1952', 7: '1953', 8: '1954' };

// ============================================================================
// Components
// ============================================================================

const IssueCard = ({ issue, volumeId, language }) => {
  const t = (key) => getTranslation(language, key);

  // Build the image path using exact filename
  const imageSrc = `/images/volume${volumeId}-covers/${issue.filename}`;

  const placeholderSrc = `data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="300" height="280" viewBox="0 0 300 280"%3E%3Crect fill="%23f1f5f9" width="300" height="280"/%3E%3Ctext fill="%2364748b" font-family="Inter,sans-serif" font-size="20" text-anchor="middle" x="150" y="140"%3E${t('issue')} ${issue.issue_num}%3C/text%3E%3C/svg%3E`;

  const [currentSrc, setCurrentSrc] = useState(imageSrc);
  const [imageLoaded, setImageLoaded] = useState(false);

  const handleImageError = () => {
    console.error(`Failed to load: ${imageSrc}`);
    setCurrentSrc(placeholderSrc);
  };

  const handleImageLoad = () => {
    setImageLoaded(true);
  };

  return (
    <Link
      to={`/library/volume/${volumeId}/issue/${issue.issue_num}`}
      className="issue-card"
      style={{ opacity: imageLoaded ? 1 : 0.8, transition: 'opacity 0.3s' }}
    >
      <img
        src={currentSrc}
        alt={`${t('issue')} ${issue.issue_num}`}
        onError={handleImageError}
        onLoad={handleImageLoad}
        loading="lazy"
        style={{ maxWidth: '300px', maxHeight: '400px', objectFit: 'cover' }}
      />
      <div className="issue-card-title">
        {t('issue')} {issue.issue_num}
      </div>
    </Link>
  );
};

const Issues = ({ language }) => {
  const { volumeId } = useParams();
  const [issues, setIssues] = useState([]);
  const t = (key) => getTranslation(language, key);

  useEffect(() => {
    // Get pre-defined images for this volume
    const volumeImages = VOLUME_IMAGES[parseInt(volumeId)] || [];
    setIssues(volumeImages);
  }, [volumeId]);

  return (
    <div className="library-container">
      <div className="library-breadcrumb">
        <Link to="/library">{t('nav_library')}</Link>
        <span className="breadcrumb-sep">/</span>
        <span className="breadcrumb-current">
          {t('lib_vol')} {volumeId} ({VOLUME_YEARS[volumeId] || ''})
        </span>
      </div>

      <div className="library-header">
        <h2 className="library-title">{t('lib_vol')} {volumeId}</h2>
        <div className="library-title-rule" />
      </div>

      {issues.length === 0 ? (
        <div className="library-empty-state">
          <p>{t('no_images')}</p>
        </div>
      ) : (
        <div className="issues-grid">
          {issues.map((issue) => (
            <IssueCard
              key={`vol${volumeId}-issue${issue.issue_num}`}
              issue={issue}
              volumeId={volumeId}
              language={language}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default Issues;
