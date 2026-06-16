import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { getTranslation } from '../services/translations';

const Navigation = ({ language, onToggleLanguage }) => {
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);
  const t = (key) => getTranslation(language, key);

  const isActive = (path) => {
    if (path === '/' && location.pathname === '/') return true;
    if (path !== '/' && location.pathname.startsWith(path)) return true;
    return false;
  };

  const handleNavClick = () => setMenuOpen(false);

  return (
    <div className="nav-container">
      <Link to="/" className="logo" onClick={handleNavClick}>
        {t('app_title')}
      </Link>
      <button
        className="nav-hamburger"
        onClick={() => setMenuOpen(!menuOpen)}
        aria-label="Toggle menu"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          {menuOpen ? (
            <>
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </>
          ) : (
            <>
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </>
          )}
        </svg>
      </button>
      <div className={`nav-links ${menuOpen ? 'open' : ''}`}>
        <Link
          to="/"
          className={`nav-link ${isActive('/') && !location.pathname.startsWith('/library') && !location.pathname.startsWith('/tags') && !location.pathname.startsWith('/history') && !location.pathname.startsWith('/about') ? 'active' : ''}`}
          onClick={handleNavClick}
        >
          {t('nav_ask_ai')}
        </Link>
        <Link
          to="/library"
          className={`nav-link ${isActive('/library') ? 'active' : ''}`}
          onClick={handleNavClick}
        >
          {t('nav_library')}
        </Link>
        <Link
          to="/tags"
          className={`nav-link ${isActive('/tags') ? 'active' : ''}`}
          onClick={handleNavClick}
        >
          {t('nav_tags')}
        </Link>
        <Link
          to="/history"
          className={`nav-link ${isActive('/history') ? 'active' : ''}`}
          onClick={handleNavClick}
        >
          {t('nav_history')}
        </Link>
        <Link
          to="/about"
          className={`nav-link ${isActive('/about') ? 'active' : ''}`}
          onClick={handleNavClick}
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
