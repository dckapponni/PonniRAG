import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { getTranslation } from '../services/translations';
import { getVolumes } from '../services/api';

const VolumeCard = ({ volume, language }) => {
  const t = (key) => getTranslation(language, key);

  const placeholderSrc = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="250" height="375" viewBox="0 0 250 375"%3E%3Crect fill="%23f1f5f9" width="250" height="375"/%3E%3Ctext fill="%2364748b" font-family="Inter,sans-serif" font-size="24" text-anchor="middle" x="125" y="187"%3EVolume ' + volume.id + '%3C/text%3E%3C/svg%3E';

  return (
    <Link to={`/library/volume/${volume.id}`} className="volume-card">
      <div className="volume-card-image-wrap">
        <img
          src={volume.cover_image_url || placeholderSrc}
          alt={`${t('lib_vol')} ${volume.id}`}
          loading="lazy"
          width={250}
          height={375}
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = placeholderSrc;
          }}
        />
      </div>
      <div className="volume-card-info">
        <div className="volume-card-title">
          {t('lib_vol')} {volume.id}
        </div>
        <div className="volume-card-year">{volume.year}</div>
      </div>
    </Link>
  );
};

const Library = ({ language }) => {
  const t = (key) => getTranslation(language, key);
  const [volumes, setVolumes] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getVolumes()
      .then((data) => setVolumes(data))
      .catch((err) => console.error('Failed to load volumes:', err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="library-container">
      <div className="library-header">
        <h2 className="library-title">{t('lib_title')}</h2>
        <div className="library-title-rule" />
        <p className="library-desc">{t('lib_desc')}</p>
      </div>
      {loading ? (
        <div className="library-empty-state"><p>Loading...</p></div>
      ) : (
        <div className="volumes-grid">
          {volumes.map((volume) => (
            <VolumeCard key={volume.id} volume={volume} language={language} />
          ))}
        </div>
      )}
    </div>
  );
};

export default Library;
