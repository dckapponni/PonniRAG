import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getTranslation } from '../services/translations';
import { getPDFLink } from '../services/api';

const PDFViewer = ({ language }) => {
  const { volumeId, issueId } = useParams();
  const [pdfData, setPdfData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const t = (key) => getTranslation(language, key);

  useEffect(() => {
    const fetchPDFLink = async () => {
      try {
        setLoading(true);
        const data = await getPDFLink(volumeId, issueId);
        setPdfData(data);
      } catch (err) {
        console.error('Error fetching PDF link:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPDFLink();
  }, [volumeId, issueId]);

  return (
    <div className="library-container">
      <div className="library-breadcrumb">
        <Link to="/library">{t('nav_library')}</Link>
        <span className="breadcrumb-sep">/</span>
        <Link to={`/library/volume/${volumeId}`}>{t('lib_vol')} {volumeId}</Link>
        <span className="breadcrumb-sep">/</span>
        <span className="breadcrumb-current">{t('issue')} {issueId}</span>
      </div>

      <div className="library-header">
        <h2 className="library-title">
          {t('lib_vol')} {volumeId} &mdash; {t('issue')} {issueId}
        </h2>
        <div className="library-title-rule" />
      </div>

      {loading ? (
        <div className="loading-spinner">
          <div className="spinner"></div>
          <span>{t('loading_pdf')}</span>
        </div>
      ) : error ? (
        <p style={{ color: '#64748b', padding: '1rem 0' }}>{t('error_loading_pdf')}: {error}</p>
      ) : pdfData && pdfData.found ? (
        <>
          <div className="pdf-container">
            <iframe
              src={pdfData.embed_url}
              title={`Volume ${volumeId} Issue ${issueId}`}
              allow="autoplay"
            />
          </div>
          <div className="pdf-actions">
            <a
              href={pdfData.pdf_url}
              target="_blank"
              rel="noopener noreferrer"
              className="pdf-link"
            >
              {t('open_pdf')}
            </a>
          </div>
        </>
      ) : (
        <p style={{ color: '#64748b', padding: '1rem 0' }}>{t('pdf_not_available')}</p>
      )}
    </div>
  );
};

export default PDFViewer;
