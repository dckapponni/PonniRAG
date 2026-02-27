import React from 'react';
import { Link } from 'react-router-dom';
import { getTranslation } from '../services/translations';

// Volume data with years
const volumes = [
  { id: 1, year: '1947', image: 'Volume1.jpg' },
  { id: 2, year: '1948', image: 'Volume2.jpg' },
  { id: 3, year: '1949', image: 'Volume3.jpg' },
  { id: 4, year: '1950', image: 'Volume4.jpg' },
  { id: 5, year: '1951', image: 'Volume5.jpg' },
  { id: 6, year: '1952', image: 'Volume6.jpg' },
  { id: 7, year: '1953', image: 'Volume7.jpg' },
  { id: 8, year: '1954', image: 'Volume8.jpg' },
];

const VolumeCard = ({ volume, language }) => {
  const t = (key) => getTranslation(language, key);

  // Try to load image from public folder
  const imageSrc = `/images/${volume.image}`;
  const placeholderSrc = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="250" height="375" viewBox="0 0 250 375"%3E%3Crect fill="%23f1f5f9" width="250" height="375"/%3E%3Ctext fill="%2364748b" font-family="Inter,sans-serif" font-size="24" text-anchor="middle" x="125" y="187"%3EVolume ' + volume.id + '%3C/text%3E%3C/svg%3E';

  return (
    <Link to={`/library/volume/${volume.id}`} className="volume-card">
      <div className="volume-card-image-wrap">
        <img
          src={imageSrc}
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

  return (
    <div className="library-container">
      <div className="library-header">
        <h2 className="library-title">{t('lib_title')}</h2>
        <div className="library-title-rule" />
        <p className="library-desc">{t('lib_desc')}</p>
      </div>
      <div className="volumes-grid">
        {volumes.map((volume) => (
          <VolumeCard key={volume.id} volume={volume} language={language} />
        ))}
      </div>
    </div>
  );
};

export default Library;
