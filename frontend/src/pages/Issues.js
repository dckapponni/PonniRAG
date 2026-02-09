import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { getTranslation } from '../services/translations';

// ============================================================================
// MANUAL IMAGE MAPPING - Based on actual file structure
// ============================================================================
const VOLUME_IMAGES = {
  1: [
    { issue_num: 1, filename: 'இதழ் 1.jpg' },
    { issue_num: 7, filename: 'இதழ் 7.jpg' },
    { issue_num: 8, filename: 'இதழ் 8.jpg' },
    { issue_num: 9, filename: 'இதழ் 9.jpg' },
    { issue_num: 10, filename: 'இதழ் 10.jpg' },
    { issue_num: 11, filename: 'இதழ் 11.jpg' },
    { issue_num: 12, filename: 'இதழ் 12.jpg' },
    // { issue_num: 'special', filename: 'பொங்கல் இதழ்.jpg' }, // Special Pongal issue
  ],
  2: [
    { issue_num: 1, filename: 'இதழ் 1.png' },
    { issue_num: 2, filename: 'இதழ்.2.png' },
    { issue_num: 3, filename: 'இதழ் 3.png' },
    { issue_num: 4, filename: 'இதழ் 4.png' },
    { issue_num: 5, filename: 'இதழ் 5.png' },
    { issue_num: 6, filename: 'இதழ் 6.png' },
    { issue_num: 7, filename: 'இதழ் 7.png' },
    { issue_num: 8, filename: 'இதழ் 8.png' },
    { issue_num: 9, filename: 'இதழ் .9.png' },
    { issue_num: 10, filename: 'இதழ் 10.png' },
    { issue_num: 11, filename: 'இதழ் 11.png' },
    { issue_num: 12, filename: 'இதழ் 12.png' },
    { issue_num: 13, filename: 'இதழ் 13.png' },
    { issue_num: 14, filename: 'இதழ்.14.png' },
    { issue_num: 15, filename: 'இதழ் 15.png' },
    { issue_num: 16, filename: 'இதழ் 16.png' },
    { issue_num: 17, filename: 'இதழ் 17.png' },
  ],
  3: [
    { issue_num: 1, filename: 'vol 3 இதழ் 1.png' },
    { issue_num: 2, filename: 'vol 3  இதழ் 2.png' },      // Note: double space
    { issue_num: 3, filename: 'vol 3 இதழ் 3.png' },
    { issue_num: 4, filename: 'vol 3 இதழ் 4.png' },
    { issue_num: 5, filename: 'vol 3 இதழ் 5.png' },
    { issue_num: 6, filename: 'vol 3இதழ் 6.png' },        // Note: no space between 3 and இ
    { issue_num: 7, filename: 'vol 3 இதழ் 7.png' },
    { issue_num: 8, filename: 'vol 3 இதழ் 8.png' },
    { issue_num: 9, filename: 'vol 3 இதழ் 9.png' },
    { issue_num: 10, filename: 'vol 3 இதழ் 10.png' },
    { issue_num: 11, filename: 'vol 3 இதழ் 11.png' },
    { issue_num: 12, filename: 'vol 3 இதழ் 12.png' },
    { issue_num: 13, filename: 'vol 3 இதழ் 13.png' },
    { issue_num: 14, filename: 'vol 3 இதழ்14.png' },      // Note: no space before 14
    { issue_num: 15, filename: 'vol 3 இதழ் 15.png' },
    { issue_num: 16, filename: 'vol 3 இதழ் 16.png' },
    { issue_num: 17, filename: 'vol 3 இதழ் 17.png' },
    { issue_num: 18, filename: 'vol 3 இதழ் 18.png' },
    { issue_num: 19, filename: 'vol 3 இதழ் 19_.png' },    // Note: underscore before .png
    { issue_num: 20, filename: 'vol 3 இதழ் 20.png' },
    { issue_num: 21, filename: 'vol 3 இதழ் 21.png' },
    { issue_num: 22, filename: 'vol 3 இதழ் 22.png' },
    { issue_num: 23, filename: 'vol 3 இதழ் 23.png' },
  ],
  4: [
    { issue_num: 1, filename: 'இதழ் 1 .png' },
    { issue_num: 4, filename: 'இதழ் 4.png' },
    { issue_num: 5, filename: 'இதழ் 5.png' },
    { issue_num: 6, filename: 'இதழ் 6.png' },
    { issue_num: 7, filename: 'இதழ் 7.png' },
    { issue_num: 8, filename: 'இதழ் 8.png' },
    { issue_num: 9, filename: 'இதழ் 9.png' },
    { issue_num: 10, filename: 'இதழ் 10.png' },
    { issue_num: 11, filename: 'இதழ் 11.png' },
  ],
  5: [
    { issue_num: 1, filename: 'இதழ் 1.png' },
    { issue_num: 2, filename: 'இதழ் 2.png' },
    { issue_num: 3, filename: 'இதழ் 3.png' },
    { issue_num: 4, filename: 'இதழ்  4.png' },
    { issue_num: 5, filename: 'இதழ்  5.png' },
    { issue_num: 6, filename: 'இதழ்  6.png' },
    { issue_num: 7, filename: 'இதழ்  7.png' },
    { issue_num: 8, filename: 'இதழ்  8.png' },
    { issue_num: 9, filename: 'இதழ் 9 .png' },
    { issue_num: 10, filename: 'இதழ் 10.png' },
    { issue_num: 11, filename: 'இதழ் 11.png' },
    { issue_num: 12, filename: 'இதழ் 12.png' },
    { issue_num: 13, filename: 'இதழ் 13.png' },
    { issue_num: 14, filename: 'இதழ் 14.png' },
    { issue_num: 15, filename: 'இதழ் 15.png' },
    { issue_num: 16, filename: 'இதழ் 16.png' },
    { issue_num: 17, filename: 'இதழ் 17.png' },
    { issue_num: 18, filename: 'இதழ் 18.png' },
    { issue_num: 19, filename: 'இதழ் 19.png' },
    { issue_num: 20, filename: 'இதழ் 20.png' },
    { issue_num: 21, filename: 'இதழ் 21.png' },
  ],
  6: [
    { issue_num: 1, filename: 'vol6 -  இதழ் 1_.jpg' },    // Note: double space, underscore
    { issue_num: 2, filename: 'vol6- இதழ் 2.jpg' },       // Note: no space after 6
    { issue_num: 3, filename: 'vol6 - இதழ் 3_.jpg' },     // Note: underscore
    { issue_num: 4, filename: 'vol6 - இதழ் 4.jpg' },
    { issue_num: 5, filename: 'vol6 -இதழ் 5.jpg' },       // Note: no space after dash
    { issue_num: 6, filename: 'vol6 - இதழ் 6.jpg' },
    { issue_num: 7, filename: 'vol6 - இதழ் 7.jpg' },
    { issue_num: 8, filename: 'vol6 - இதழ் 8.jpg' },
    { issue_num: 9, filename: 'vol6 - இதழ் 9.jpg' },
    { issue_num: 10, filename: 'vol6 - இதழ் 10.jpg' },
    { issue_num: 11, filename: 'vol6 - இதழ் 11_.jpg' },   // Note: underscore
    { issue_num: 12, filename: 'vol6 இதழ் 12.jpg' },      // Note: no dash
    { issue_num: 14, filename: 'vol6 - இதழ் 14.jpg' },    // Note: issue 13 missing
    { issue_num: 17, filename: 'vol6 -  இதழ் 17.jpg' },   // Note: double space
    { issue_num: 19, filename: 'vol6 - இதழ் 19.jpg' },
    { issue_num: 20, filename: 'vol6 - இதழ் 20.jpg' },
    { issue_num: 21, filename: 'vol6 - இதழ் 21.jpg' },
    // { issue_num: 'pongal', filename: 'vol6 - pongal  இதழ்_.jpg' }, // Special Pongal issue
  ],
  7: [
    { issue_num: 1, filename: 'இதழ் 1.jpg' },
  ],
  8: [
    { issue_num: 1, filename: 'இதழ் 1.jpg' },
    { issue_num: 2, filename: 'இதழ் 2.jpg' },
    { issue_num: 3, filename: 'இதழ் 3.jpg' },
    { issue_num: 4, filename: 'இதழ் 4.jpg' },
    { issue_num: 6, filename: 'இதழ் 6.jpg' },
    { issue_num: 7, filename: 'இதழ் 7.jpg' },
    { issue_num: 8, filename: 'இதழ் 8.jpg' },
    { issue_num: 10, filename: 'இதழ் 10.jpg' },
    { issue_num: 11, filename: 'இதழ் 11.jpg' },
    { issue_num: 12, filename: 'இதழ் 12.jpg' },
  ],
};

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
  const navigate = useNavigate();
  const [issues, setIssues] = useState([]);
  const t = (key) => getTranslation(language, key);

  useEffect(() => {
    // Get pre-defined images for this volume
    const volumeImages = VOLUME_IMAGES[parseInt(volumeId)] || [];
    setIssues(volumeImages);
  }, [volumeId]);

  return (
    <div className="library-container">
      <button className="back-btn" onClick={() => navigate('/library')}>
        ← {t('lib_back')}
      </button>
      <h2 className="library-title">{t('lib_vol')} {volumeId}</h2>

      {issues.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
          <p>No images configured for this volume</p>
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