import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getTranslation } from '../services/translations';
import { getVolumeIssues, getVolumes } from '../services/api';

// ============================================================================
// Components
// ============================================================================

const IssueCard = ({ issue, volumeId, language }) => {
  const t = (key) => getTranslation(language, key);

  const placeholderSrc = `data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="300" height="280" viewBox="0 0 300 280"%3E%3Crect fill="%23f1f5f9" width="300" height="280"/%3E%3Ctext fill="%2364748b" font-family="Inter,sans-serif" font-size="20" text-anchor="middle" x="150" y="140"%3E${t('issue')} ${issue.issue_number}%3C/text%3E%3C/svg%3E`;

  const [currentSrc, setCurrentSrc] = useState(issue.cover_image_url || placeholderSrc);
  const [imageLoaded, setImageLoaded] = useState(false);

  const handleImageError = () => {
    setCurrentSrc(placeholderSrc);
  };

  const handleImageLoad = () => {
    setImageLoaded(true);
  };

  return (
    <Link
      to={`/library/volume/${volumeId}/issue/${issue.issue_number}`}
      className="issue-card"
      style={{ opacity: imageLoaded ? 1 : 0.8, transition: 'opacity 0.3s' }}
    >
      <img
        src={currentSrc}
        alt={`${t('issue')} ${issue.issue_number}`}
        onError={handleImageError}
        onLoad={handleImageLoad}
        loading="lazy"
        style={{ maxWidth: '300px', maxHeight: '400px', objectFit: 'cover' }}
      />
      <div className="issue-card-title">
        {t('issue')} {issue.issue_number}
      </div>
    </Link>
  );
};

const Issues = ({ language }) => {
  const { volumeId } = useParams();
  const [issues, setIssues] = useState([]);
  const [volumeYear, setVolumeYear] = useState('');
  const [loading, setLoading] = useState(true);
  const t = (key) => getTranslation(language, key);

  useEffect(() => {
    setLoading(true);

    // Fetch issues and volume year in parallel
    Promise.all([
      getVolumeIssues(volumeId),
      getVolumes(),
    ])
      .then(([issuesData, volumesData]) => {
        setIssues(issuesData);
        const vol = volumesData.find((v) => v.id === parseInt(volumeId));
        if (vol) setVolumeYear(vol.year);
      })
      .catch((err) => console.error('Failed to load issues:', err))
      .finally(() => setLoading(false));
  }, [volumeId]);

  return (
    <div className="library-container">
      <div className="library-breadcrumb">
        <Link to="/library">{t('nav_library')}</Link>
        <span className="breadcrumb-sep">/</span>
        <span className="breadcrumb-current">
          {t('lib_vol')} {volumeId} ({volumeYear})
        </span>
      </div>

      <div className="library-header">
        <h2 className="library-title">{t('lib_vol')} {volumeId}</h2>
        <div className="library-title-rule" />
      </div>

      {loading ? (
        <div className="library-empty-state"><p>Loading...</p></div>
      ) : issues.length === 0 ? (
        <div className="library-empty-state">
          <p>{t('no_images')}</p>
        </div>
      ) : (
        <div className="issues-grid">
          {issues.map((issue) => (
            <IssueCard
              key={`vol${volumeId}-issue${issue.issue_number}`}
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
