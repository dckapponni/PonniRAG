import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { getTranslation } from '../services/translations';

const Navigation = ({ language, onToggleLanguage }) => {
  const location = useLocation();
  const t = (key) => getTranslation(language, key);

  const isActive = (path) => {
    if (path === '/' && location.pathname === '/') return true;
    if (path !== '/' && location.pathname.startsWith(path)) return true;
    return false;
  };

  return (
    <div className="nav-container">
      <Link to="/" className="logo">
        {t('app_title')}
      </Link>
      <div className="nav-links">
        <Link
          to="/"
          className={`nav-link ${isActive('/') && !location.pathname.startsWith('/library') && !location.pathname.startsWith('/about') ? 'active' : ''}`}
        >
          {t('nav_ask_ai')}
        </Link>
        <Link
          to="/library"
          className={`nav-link ${isActive('/library') ? 'active' : ''}`}
        >
          {t('nav_library')}
        </Link>
        <Link
          to="/about"
          className={`nav-link ${isActive('/about') ? 'active' : ''}`}
        >
          {t('nav_about')}
        </Link>
        <button className="lang-toggle" onClick={onToggleLanguage}>
          {t('nav_toggle')}
        </button>
      </div>
    </div>
  );
};

export default Navigation;
