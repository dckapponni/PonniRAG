# Datasheet — Ponni Archive Tamil Corpus (v1)

Following the datasheet framework of Gebru et al. (2021). Licence terms are in
`DATA_LICENSE.md`; access instructions are in §6.

Fields marked **[pending]** are filled in at release time from the distribution
manifest and are not yet final.

---

## 1. Motivation

**For what purpose was the dataset created?**
To make the Tamil literary magazine *Ponni* (1947–1955) searchable, citable, and
computationally reusable. *Ponni* is a significant but under-studied Dravidian-
movement periodical: it published rationalist and self-respect writing, carried
Bharathidasan and the poets it labelled "the Bharathidasan lineage", and
introduced Tamil readers to Chinese, Persian, Russian, Kannada, Urdu, Sanskrit,
and Telugu literatures. No machine-readable edition existed. The corpus was
built as the retrieval substrate for the Ponni RAG system, and is released
separately so it can be reused independently of that system.

**Who created it and who funded it?**
Created by the Ponni Archive project. Digitisation, OCR, and proofreading were
carried out by the Department of Tamil, Madras Christian College. System and
pipeline development by the Ponni RAG team (DCKAP). Source material and OCR
support from the Roja Muthiah Research Library (RMRL), Chennai; original print
preservation by Perry Alagappan. Funded by DCKAP.

**What gap does it fill?**
Tamil is under-resourced for information retrieval, and historical Tamil
periodicals more so. Existing Tamil NLP corpora are predominantly contemporary
text. This corpus supplies proofread mid-20th-century Tamil prose and poetry
with article-level structure and author attribution.

---

## 2. Composition

**What do the instances represent?**
Two nested units:

1. **Issue-level documents** — the proofread full text of one magazine issue.
2. **Article-level records** — a single article, with title, author, volume,
   issue, year, category tag, and body text, produced by automatic article
   boundary detection over the issue text.

**How many instances are there?**

| Quantity | Count |
|---|---|
| Volumes | 8 |
| Issues in the registry | 108 |
| Article metadata rows (`summary.csv`) | 1,702 |
| Extracted articles used for taxonomy derivation | 1,667 |
| Category labels | 15 |
| Years covered | 1947–1953, 1955 |

Issues per volume, and metadata rows per volume:

| Volume | Year | Issues | Metadata rows |
|---|---|---|---|
| 1 | 1947 | 8 | 166 |
| 2 | 1948 | 18 | 331 |
| 3 | 1949 | 23 | 356 |
| 4 | 1950 | 9 | 141 |
| 5 | 1951 | 21 | 264 |
| 6 | 1952 | 18 | 255 |
| 7 | 1953 | 1 | 13 |
| 8 | 1955 | 10 | 175 |

One metadata row carries `NA` for volume. Issue identifiers are strings, not
integers: `PONGAL` is a valid issue name for special festival numbers.

Token and character counts, per-file sizes, and checksums: **[pending]** —
generated from the release manifest.

**Is it a sample or the complete set?**
It is the complete set of issues that the project has been able to source and
proofread — not the complete published run of *Ponni*. Volume 7 (1953) is
represented by a single issue, and no 1954 volume is present, because those
issues have not been located. Coverage per volume above is the coverage that
exists, and the gap is a property of surviving material and sourcing, not of
sampling.

**What data does each instance consist of?**
Unicode Tamil text (proofread), plus structural metadata. No images are included
in the text corpus; scanned page PDFs are served separately through the archive
website and are not part of this distribution.

**Is there a label or target?**
Each article carries one of 15 taxonomy categories, assigned by a rule-based
classifier with a statistical fallback. The taxonomy was derived from
title-frequency analysis of 1,667 extracted articles. Category labels are
system-assigned, not human-verified in bulk, and should be treated as weak
labels.

**Is any information missing?**

- Volume 1 issue 1 has a scanned PDF but no proofread text — it was never
  digitised to text.
- Volume 1 issue 7 has 15 articles in the metadata CSV and a proofread source
  document is present, but the document is anomalously large relative to its
  siblings and its extraction has historically been incomplete; see the
  known-gaps table in `data_pipeline.md`.
- Author fields are empty for unsigned or pseudonymous articles, which were
  common in the period.
- Article boundary detection is imperfect on issues with heavy multi-column or
  interleaved layouts; a small number of articles are merged or split.

**Are there errors, noise, or redundancies?**
Yes, and they are inherent to the source. The text originates in OCR of ageing
print, then human proofreading. Residual issues include: occasional
mis-segmented articles; orthographic variation from historical Tamil printing
conventions, which differ from modern normalisation; and inconsistent
transliteration of non-Tamil proper nouns. Serialised fiction appears across
multiple issues under numbered titles and is intentionally retained as separate
articles.

**Does it contain confidential or sensitive data?**
No. All content is previously published material from a public periodical.

**Does it contain data that might be offensive or distressing?**
The corpus is a historical political and literary periodical and reflects the
polemics of its time. It contains sharp criticism of caste practice and
religious orthodoxy, and period-typical language about caste, gender, and
religion that will not match contemporary usage. Material is preserved as
published, without redaction, because altering it would destroy its value as a
historical record. Users building generative systems on it should account for
this.

**Does it identify individuals?**
It names authors, editors, and public figures, as published. No private personal
data was added.

---

## 3. Collection process

**How was the data acquired?**
Physical and scanned copies of *Ponni* issues were assembled from the archive
preserved by Perry Alagappan and from soft copies supplied by the Roja Muthiah
Research Library. Pages were scanned, OCRed, and then manually proofread against
the scans by a team at the Department of Tamil, Madras Christian College.

**What was the sampling strategy?**
None — the aim was exhaustive coverage of every issue obtainable. See §2 for the
coverage that resulted.

**Who was involved and how were they compensated?**
Proofreading and e-book production were carried out by a coordinated student and
staff team at Madras Christian College, acknowledged individually in the
resource paper. Work was carried out as part of a funded collaboration.

**Over what timeframe was the data collected?**
The source material was published 1947–1955. Digitisation and proofreading were
carried out by the project through 2025–2026.

**Were ethical review processes conducted?**
No institutional review was required: the material is previously published,
non-personal, historical text. Rights concerns are handled through the takedown
process in `DATA_LICENSE.md` §7.

---

## 4. Preprocessing, cleaning, labelling

**What preprocessing was done?**
The pipeline, all of which is released as open source, is:

1. **Proofread source documents** (DOCX) held in
   `s3://ponni-dev/Raw_Proof_Read_Content/`.
2. **Text extraction** — `src/data_extraction/text_extraction.py` produces plain
   text.
3. **Article separation** — `src/data_extraction/article_seperation.py` detects
   article boundaries using title patterns and the metadata CSV, producing one
   structured JSON record per article.
4. **Category tagging** — `src/db/article_tagger.py` assigns one of 15
   categories by rule matching, with a statistical fallback for unmatched
   articles; multi-part fiction is detected via title numbering.
5. **Indexing** (not part of the corpus release) — chunking and embedding with
   `intfloat/multilingual-e5-large` into Qdrant.

Tamil-specific normalisation is applied at query and index time in the retrieval
system, **not** baked into the released text. The distributed corpus preserves
the proofread text as-is, so downstream users can apply their own normalisation.

**Was the raw data saved?**
Yes. Proofread source documents and every intermediate stage are retained in S3,
so any stage is reproducible and re-derivable.

**Is the preprocessing software available?**
Yes — <https://github.com/dckapponni/PonniRAG>.

---

## 5. Uses

**What tasks has the dataset been used for?**
Hybrid dense–sparse retrieval with cross-encoder reranking, and retrieval-
augmented question answering over historical Tamil text, as reported in the
resource paper. It also backs a public archive interface at
<https://ponniarchive.com/>.

**What other tasks could it be used for?**
Historical Tamil language modelling and domain adaptation; OCR post-correction
research; author attribution and stylometry; diachronic lexical study of
mid-20th-century Tamil; Dravidian-movement historiography and periodical
studies; Tamil document layout and article segmentation benchmarks.

**Is there anything about the composition that could cause unfair treatment?**
Two points. First, the corpus is ideologically non-neutral by construction: it
is one movement's periodical, and a model trained on it will reproduce that
stance. Second, coverage is uneven across years (§2), so time-series claims
drawn from it will be skewed toward 1948–1952 unless corrected.

**Are there tasks for which it should not be used?**
It should not be used as a source of factual claims about present-day people,
communities, or institutions; as a general-purpose Tamil language corpus without
noting its period and register; or to generate text presented as contemporary
Tamil. Commercial use requires permission (`DATA_LICENSE.md` §3.2).

---

## 6. Distribution

**How is it distributed?**

| Layer | Access | Licence |
|---|---|---|
| Metadata and annotations | Direct download, and in the GitHub repository | CC BY 4.0 |
| Evaluation resources (50 QA pairs, graded judgments, judgment pool, metrics) | Direct download, and in the GitHub repository | CC BY 4.0 |
| Full proofread corpus | Request form at <https://ponniarchive.com/dataset> or <contact@ponniarchive.com> | Ponni Archive Data Use Agreement |

Requests to the form are logged and a time-limited download link is emailed to
the requesting address. The request step exists to keep a record of use and to
attach the use agreement, not to filter research access.

**When was it distributed, and under what identifier?**
Version 1, 2026. Archival DOI: **[pending]**.

**Are there IP-based or other restrictions?**
Yes — see `DATA_LICENSE.md` §1 and §3, in particular the reason the full corpus
is not under a Creative Commons licence.

**Have export controls or regulatory restrictions been applied?**
No.

---

## 7. Maintenance

**Who maintains it and how can they be contacted?**
The Ponni Archive project — <contact@ponniarchive.com>.

**Will it be updated?**
Yes. Newly sourced and proofread issues are added as they become available,
which will close some of the gaps in §2. Transcription corrections are folded
into subsequent versions.

**How are updates communicated?**
Through versioned releases (`ponni-corpus-vN`), the archival record, and a
change log in this file. Superseded versions remain retrievable on request so
that published results stay reproducible.

**Will older versions be supported?**
Older versions are retained and available on request; only the latest version
receives corrections.

**Can others extend or contribute?**
Yes. Corrections and additional issue scans are welcome by email or as GitHub
issues. Contributed material is reviewed against the scans before inclusion.

---

## Change log

| Version | Date | Change |
|---|---|---|
| v1 | 2026 | Initial release: 8 volumes, 108 issues, 1,702 metadata rows. |
