import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getTranslation } from '../services/translations';
import { getPDFLink } from '../services/api';

const PDFViewer = ({ language }) => {
  const { volumeId, issueId } = useParams();
  const navigate = useNavigate();
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

  const handleBack = () => {
    navigate(`/library/volume/${volumeId}`);
  };

  return (
    <div className="library-container">
      <button className="back-btn" onClick={handleBack}>
        ← {t('lib_back_issues')}
      </button>
      <h2 className="library-title">
        {t('lib_vol')} {volumeId} - {t('issue')} {issueId}
      </h2>

      {loading ? (
        <div className="loading-spinner">
          <div className="spinner"></div>
          <span>{t('loading_pdf')}</span>
        </div>
      ) : error ? (
        <p>{t('error_loading_pdf')}: {error}</p>
      ) : pdfData && pdfData.found ? (
        <>
          <div className="pdf-container">
            <iframe
              src={pdfData.embed_url}
              title={`Volume ${volumeId} Issue ${issueId}`}
              allow="autoplay"
            />
          </div>
          <a
            href={pdfData.pdf_url}
            target="_blank"
            rel="noopener noreferrer"
            className="pdf-link"
          >
            {t('open_pdf')}
          </a>
        </>
      ) : (
        <p>{t('pdf_not_available')}</p>
      )}
    </div>
  );
};

export default PDFViewer;
