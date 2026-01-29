import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { getTranslation } from '../services/translations';
import { getVolumeIssues } from '../services/api';

const IssueCard = ({ issue, volumeId, language }) => {
  const t = (key) => getTranslation(language, key);

  // Try to load image from public folder
  const imageSrc = `/images/volume ${volumeId} cover images/issue${issue.issue_number}.jpg`;
  const placeholderSrc = `data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="300" height="280" viewBox="0 0 300 280"%3E%3Crect fill="%23f1f5f9" width="300" height="280"/%3E%3Ctext fill="%2364748b" font-family="Inter,sans-serif" font-size="20" text-anchor="middle" x="150" y="140"%3E${t('issue')} ${issue.issue_number}%3C/text%3E%3C/svg%3E`;

  return (
    <Link to={`/library/volume/${volumeId}/issue/${issue.issue_number}`} className="issue-card">
      <img
        src={imageSrc}
        alt={`${t('issue')} ${issue.issue_number}`}
        onError={(e) => {
          e.target.onerror = null;
          e.target.src = placeholderSrc;
        }}
      />
      <div className="issue-card-title">
        {t('issue')} {issue.issue_number}
      </div>
    </Link>
  );
};

const Issues = ({ language }) => {
  const { volumeId } = useParams();
  const navigate = useNavigate();
  const [issues, setIssues] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const t = (key) => getTranslation(language, key);

  useEffect(() => {
    const fetchIssues = async () => {
      try {
        setLoading(true);
        const data = await getVolumeIssues(volumeId);
        setIssues(data);
      } catch (err) {
        console.error('Error fetching issues:', err);
        setError(err.message);
        // Fallback to mock data if API fails
        const mockIssues = Array.from({ length: 10 }, (_, i) => ({
          issue_number: i + 1,
          has_pdf: true,
        }));
        setIssues(mockIssues);
      } finally {
        setLoading(false);
      }
    };

    fetchIssues();
  }, [volumeId]);

  return (
    <div className="library-container">
      <button className="back-btn" onClick={() => navigate('/library')}>
        ← {t('lib_back')}
      </button>
      <h2 className="library-title">{t('lib_vol')} {volumeId}</h2>

      {loading ? (
        <div className="loading-spinner">
          <div className="spinner"></div>
          <span>Loading...</span>
        </div>
      ) : error ? (
        <p>Using fallback data</p>
      ) : null}

      <div className="issues-grid">
        {issues.map((issue) => (
          <IssueCard
            key={issue.issue_number}
            issue={issue}
            volumeId={volumeId}
            language={language}
          />
        ))}
      </div>
    </div>
  );
};

export default Issues;
