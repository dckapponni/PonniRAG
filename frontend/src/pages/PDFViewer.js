import React, { useState, useEffect, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Viewer, Worker } from '@react-pdf-viewer/core';
import { defaultLayoutPlugin } from '@react-pdf-viewer/default-layout';
import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/default-layout/lib/styles/index.css';
import { getTranslation } from '../services/translations';
import { getPDFLink } from '../services/api';

const PDFViewer = ({ language }) => {
  const { volumeId, issueId } = useParams();
  const [pdfData, setPdfData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const t = (key) => getTranslation(language, key);

  const defaultLayoutPluginInstance = useMemo(() => defaultLayoutPlugin(), []);
  const apiBase = process.env.REACT_APP_API_URL || '';
  const fileUrl = pdfData?.proxy_url
    ? (pdfData.proxy_url.startsWith('http') ? pdfData.proxy_url : apiBase + pdfData.proxy_url)
    : null;

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
      ) : pdfData && pdfData.found && fileUrl ? (
        <>
          <div className="pdf-container">
            <Worker workerUrl="/pdf.worker.min.js">
              <Viewer fileUrl={fileUrl} plugins={[defaultLayoutPluginInstance]} />
            </Worker>
          </div>
          {pdfData.pdf_url && (
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
          )}
        </>
      ) : pdfData && pdfData.found && pdfData.embed_url ? (
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
