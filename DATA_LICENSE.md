# Ponni Archive — Data Licence and Use Terms

_Version 1.0 — 15 August 2026_

This document covers the **data** released by the Ponni Archive project. The
**software** in this repository is licensed separately under the Apache License
2.0 (see `LICENSE` and `NOTICE`); nothing here changes that.

---

## 1. Three layers, two licences

The release is deliberately split, because the project does not hold the same
rights over every layer.

| Layer | What it is | Licence |
|---|---|---|
| **A. Metadata and annotations** | Article-level metadata (`src/data/summary.csv`: year, volume, issue, title, author), the 15-category taxonomy and per-article category assignments, corpus statistics | **CC BY 4.0** |
| **B. Evaluation resources** | 50 Tamil question–answer pairs with expert reference answers, TREC-style graded relevance judgments over 41 scored queries, the judgment pool, and metric implementations | **CC BY 4.0** |
| **C. Full proofread OCR corpus** | The OCR-extracted, manually proofread Tamil text of the digitised *Ponni* issues, and the derived per-article JSON | **Ponni Archive Data Use Agreement (§3)** — request required |

Layers A and B are openly downloadable with no request step. Layer C requires an
accepted request (see §4).

### Why layer C is not under a Creative Commons licence

*Ponni* was published between 1947 and 1955. Under the Indian Copyright Act,
literary works are protected for the life of the author plus 60 years, so the
copyright status of individual contributions varies: many contributors' works
have entered the public domain, while others may not have. The Ponni Archive
project holds rights in its own **transcription, proofreading, segmentation and
annotation work**, but applying a Creative Commons licence to the underlying
article text would assert an ownership the project does not uniformly hold. A
data use agreement is a permission grant rather than an ownership claim, and is
therefore the accurate instrument for layer C.

If you are a rights holder in any *Ponni* contribution and object to its
inclusion, contact <contact@ponniarchive.com> and the material will be removed
from the distributed corpus (see §7).

---

## 2. Licence text for layers A and B

Layers A and B are made available under the
**Creative Commons Attribution 4.0 International licence (CC BY 4.0)**:
<https://creativecommons.org/licenses/by/4.0/>

You are free to share and adapt this material, including commercially, provided
you give appropriate credit (see §6 for the citation).

---

## 3. Ponni Archive Data Use Agreement (layer C)

By requesting and downloading the full proofread corpus, you agree to the
following terms.

**3.1 Permitted use.** You may use the corpus for non-commercial research,
scholarship, teaching, and personal study. This includes computational analysis,
training and evaluating retrieval or language models for research purposes, and
publishing research results derived from the corpus.

**3.2 Prohibited use.** You may not:

- (a) use the corpus, or models trained principally on it, in a commercial
  product or service without prior written permission;
- (b) redistribute the corpus, in whole or in substantial part, on any public
  platform, mirror, dataset hub, or file-sharing service — direct others to
  <https://ponniarchive.com/dataset> instead;
- (c) present the material as your own work, or remove provenance information;
- (d) republish the material in a form that competes with the archive's own
  public presentation of the magazine, such as a reprinted edition offered for
  sale.

**3.3 Attribution.** Any publication, model card, or public artefact derived
from the corpus must cite the resource as set out in §6, and must state the
corpus version used.

**3.4 Derived works.** You may publish derived artefacts (annotations, model
weights, statistics, embeddings, evaluation results, quoted excerpts sufficient
to support your findings) under a licence of your choosing, provided the derived
artefact does not permit reconstruction of the substantial whole of the corpus
text. Excerpts quoted in scholarly publications for analysis, criticism, or
review are expressly permitted.

**3.5 No warranty.** The corpus is provided "as is". The text is derived from
OCR of ageing print material followed by human proofreading; residual errors
remain (see `DATASHEET.md`). The project makes no warranty of accuracy,
completeness, or fitness for any purpose, and accepts no liability arising from
use of the data.

**3.6 Third-party rights.** This agreement grants only the permissions the Ponni
Archive project is able to grant. It does not grant, and cannot grant, rights
held by third parties in individual contributions. You are responsible for your
own compliance with applicable copyright law in your jurisdiction, including any
determination of whether a specific use requires additional permission.

**3.7 Term and termination.** Permission continues until terminated. It
terminates automatically if you breach these terms. On termination you must stop
distributing any copies of the corpus in your possession; you may retain copies
needed to verify already-published research results.

**3.8 Governing law.** This agreement is governed by the laws of India, with
courts at Chennai, Tamil Nadu having jurisdiction.

---

## 4. How to request the full corpus

Submit the request form at <https://ponniarchive.com/dataset>, or email
<contact@ponniarchive.com> with your name, affiliation, and intended use.
Requests are logged, and a download link is issued to the address you supply.
Download links are single-dataset and time-limited; request a new one at any
time.

You will be asked to confirm that you accept §3 of this document.

---

## 5. Versioning

The corpus is versioned (`ponni-corpus-vN`). Each version has a fixed content
manifest with per-file checksums. Superseded versions remain retrievable on
request so that published results stay reproducible. Version history and
per-version changes are recorded in `DATASHEET.md`.

---

## 6. How to cite

If you use any layer of this release, cite the resource paper:

```bibtex
@misc{ponnirag2026,
  title  = {Hybrid Retrieval-Augmented Generation System for the Tamil
            {Ponni} Archive},
  author = {Sankar, Honika and Suresh, Abinaya and Chidambaram, Karthik and
            Karunakaran, S. and Srija, S. and Mahendiran, Abinaya},
  year   = {2026},
  note   = {Under review at the Forum for Information Retrieval Evaluation
            (FIRE) 2026}
}
```

Once the paper is accepted this is replaced with the published
`@inproceedings` record.

Please also state the corpus version, for example: "Ponni corpus v1, obtained
under the Ponni Archive Data Use Agreement."

---

## 7. Takedown and corrections

- **Rights concerns:** <contact@ponniarchive.com>. Material subject to a
  credible rights objection is withdrawn from the distributed corpus pending
  review, and the outcome recorded in `DATASHEET.md`.
- **Transcription errors:** corrections are welcome at the same address, or as a
  GitHub issue, and are folded into the next corpus version.

---

## 8. Acknowledgement of sources

Digitisation, OCR, and proofreading were carried out by the Department of Tamil,
Madras Christian College. Soft copies of several issues and OCR support were
provided by the Roja Muthiah Research Library (RMRL), Chennai. Original material
was preserved and supplied by Perry Alagappan. These contributions are
acknowledged in full in the resource paper.
