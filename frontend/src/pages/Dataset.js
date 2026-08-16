import React, { useEffect, useState } from 'react';
import {
  datasetOpenDownloadUrl,
  getDatasetInfo,
  requestDatasetAccess,
} from '../services/api';
import { getTranslation } from '../services/translations';

const CITATION = `@inproceedings{ponnirag2026,
  title     = {Ponni RAG: A Hybrid Retrieval-Augmented Question Answering
               System and Corpus for a Tamil Literary Archive},
  author    = {Mahendiran, Abinaya and Suresh, Abinaya and Karunakaran, S. and
               Srija, S. and Chidambaram, Karthik},
  booktitle = {Forum for Information Retrieval Evaluation (FIRE)},
  year      = {2026}
}`;

const EMPTY_FORM = {
  name: '',
  email: '',
  affiliation: '',
  intended_use: '',
  license_accepted: false,
};

const Dataset = ({ language }) => {
  const ta = language === 'ta';
  const t = (key) => getTranslation(language, key);

  const [info, setInfo] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    getDatasetInfo()
      .then((data) => {
        if (!cancelled) setInfo(data);
      })
      .catch(() => {
        if (!cancelled) setInfo(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleChange = (field) => (event) => {
    const value =
      event.target.type === 'checkbox' ? event.target.checked : event.target.value;
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const data = await requestDatasetAccess(form);
      setResult(data);
      setForm(EMPTY_FORM);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || t('ds_error'));
    } finally {
      setSubmitting(false);
    }
  };

  const contactEmail = info?.contact_email || 'contact@ponniarchive.com';
  const title = (file) => (ta ? file.title_ta : file.title_en);
  const description = (file) => (ta ? file.description_ta : file.description_en);

  return (
    <div className="dataset-container">
      <div className="dataset-header">
        <h2 className="dataset-heading">{t('ds_title')}</h2>
        <p className="dataset-subtitle">{t('ds_subtitle')}</p>
        {info && (
          <p className="dataset-version">
            {t('ds_version')}: <code>{info.corpus_version}</code>
          </p>
        )}
      </div>

      <section className="dataset-section">
        <h3 className="dataset-section-heading">{t('ds_open_heading')}</h3>
        <p className="dataset-note">{t('ds_open_note')}</p>
        <div className="dataset-file-list">
          {(info?.open_files || []).map((file) => (
            <div className="dataset-file" key={file.id}>
              <div className="dataset-file-text">
                <div className="dataset-file-title">{title(file)}</div>
                <div className="dataset-file-desc">{description(file)}</div>
                <div className="dataset-file-license">{file.license}</div>
              </div>
              <a
                className="dataset-download-btn"
                href={datasetOpenDownloadUrl(file.id)}
              >
                {t('ds_download')}
              </a>
            </div>
          ))}
        </div>
      </section>

      <section className="dataset-section">
        <h3 className="dataset-section-heading">{t('ds_gated_heading')}</h3>
        <p className="dataset-note">{t('ds_gated_note')}</p>
        {(info?.gated_files || []).map((file) => (
          <div className="dataset-file" key={file.id}>
            <div className="dataset-file-text">
              <div className="dataset-file-title">{title(file)}</div>
              <div className="dataset-file-desc">{description(file)}</div>
              <div className="dataset-file-license">
                {file.license}
                {file.version ? ` · ${file.version}` : ''}
              </div>
            </div>
          </div>
        ))}

        <h4 className="dataset-form-heading">{t('ds_form_heading')}</h4>
        <p className="dataset-note">{t('ds_form_intro')}</p>

        {result ? (
          <div className="dataset-success">
            <p>{result.message}</p>
            <p className="dataset-request-id">
              {t('ds_request_id')}: <code>{result.request_id}</code>
            </p>
            <p>
              {t('ds_contact_note')}{' '}
              <a href={`mailto:${result.contact_email}`}>{result.contact_email}</a>
            </p>
          </div>
        ) : (
          <form className="dataset-form" onSubmit={handleSubmit}>
            <label className="dataset-field">
              <span>{t('ds_name')}</span>
              <input
                type="text"
                value={form.name}
                onChange={handleChange('name')}
                maxLength={200}
                required
              />
            </label>

            <label className="dataset-field">
              <span>{t('ds_email')}</span>
              <input
                type="email"
                value={form.email}
                onChange={handleChange('email')}
                maxLength={254}
                required
              />
            </label>

            <label className="dataset-field">
              <span>{t('ds_affiliation')}</span>
              <input
                type="text"
                value={form.affiliation}
                onChange={handleChange('affiliation')}
                placeholder={t('ds_affiliation_ph')}
                maxLength={300}
                required
              />
            </label>

            <label className="dataset-field">
              <span>{t('ds_intended_use')}</span>
              <textarea
                rows={4}
                value={form.intended_use}
                onChange={handleChange('intended_use')}
                placeholder={t('ds_intended_use_ph')}
                minLength={20}
                maxLength={2000}
                required
              />
            </label>

            <label className="dataset-checkbox">
              <input
                type="checkbox"
                checked={form.license_accepted}
                onChange={handleChange('license_accepted')}
                required
              />
              <span>
                {t('ds_accept_license')} —{' '}
                <a
                  href={datasetOpenDownloadUrl('license')}
                  target="_blank"
                  rel="noreferrer"
                >
                  {t('ds_read_license')}
                </a>
              </span>
            </label>

            {error && <div className="dataset-error">{error}</div>}

            <button
              className="dataset-submit-btn"
              type="submit"
              disabled={submitting || !form.license_accepted}
            >
              {submitting ? t('ds_submitting') : t('ds_submit')}
            </button>
          </form>
        )}
      </section>

      <section className="dataset-section">
        <h3 className="dataset-section-heading">{t('ds_cite_heading')}</h3>
        <pre className="dataset-citation">{CITATION}</pre>
        <p className="dataset-note">
          {t('ds_contact_note')}{' '}
          <a href={`mailto:${contactEmail}`}>{contactEmail}</a>
        </p>
      </section>
    </div>
  );
};

export default Dataset;
