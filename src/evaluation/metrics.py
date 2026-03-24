"""
Metric implementations for PonniRAG evaluation.

Tamil-specific design decisions
--------------------------------
1. Semantic similarity  — uses intfloat/multilingual-e5-large (the same model
   already deployed for indexing/search).  It carries explicit Tamil support
   (`ta` in its language tags) and produces 1024-dim cosine embeddings, making
   it the best single choice for this project: no extra model download, and its
   multilingual STS training means it generalises to paraphrase detection in
   Tamil.

2. BERTScore           — uses intfloat/multilingual-e5-large as the reference
   model (via the `bert_score` library's `model_type` override).  This avoids
   downloading a second heavy model.  BERTScore computes token-level cosine
   similarities between reference and candidate, then reports Precision, Recall,
   and F1.  For Tamil it handles sub-word tokenisation correctly because XLM-R
   (underlying E5-large) was trained on Tamil Wikipedia.

3. BLEU                — computed at the CHARACTER level (unigram + bigram) using
   sacrebleu's `BLEU` scorer with `tokenize="char"`.  This sidesteps the
   absence of reliable whitespace-delimited word boundaries in Tamil (Tamil
   script can agglutinate multiple morphemes without spaces).  Character n-gram
   overlap still captures surface-level lexical similarity.

4. ROUGE-L             — uses rouge-score with character-level tokenisation
   (each Unicode code-point is treated as a token).  Tamil uses the Unicode
   range U+0B80–U+0BFF; splitting on characters preserves these code-points and
   measures longest common subsequence more faithfully than byte-level splits.

5. Composite score     — weighted average: 0.50 × semantic + 0.25 × BERTScore-F1
   + 0.15 × ROUGE-L + 0.10 × BLEU-1.  Semantic similarity dominates because
   Tamil synonyms and inflections are common; exact-match lexical scores alone
   would under-estimate answer quality.

Dependencies (add to requirements.txt as needed):
    sentence-transformers  (already present)
    bert-score
    sacrebleu
    rouge-score
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weights for the composite score
# ---------------------------------------------------------------------------
WEIGHT_SEMANTIC = 0.70
WEIGHT_ROUGE_L = 0.20
WEIGHT_BLEU1 = 0.10

# E5 prefix used when encoding evaluation texts (symmetrical pair comparison,
# so we use the "query:" prefix for both — this matches the STS use-case where
# both sentences are "queries" being compared against each other).
E5_EVAL_PREFIX = "query: "

# The embedding model used by the RAG system — reused here to avoid a second
# large model download.
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


@dataclass
class MetricResult:
    """Stores all metric scores for a single question/answer pair."""

    sample_id: str
    semantic_similarity: float = 0.0
    bleu_1: float = 0.0
    bleu_2: float = 0.0
    rouge_l: float = 0.0
    composite_score: float = 0.0

    def to_dict(self) -> dict:
        """Convert the metric result to a plain dictionary."""
        return {
            "id": self.sample_id,
            "semantic_similarity": round(self.semantic_similarity, 4),
            "bleu_1": round(self.bleu_1, 4),
            "bleu_2": round(self.bleu_2, 4),
            "rouge_l": round(self.rouge_l, 4),
            "composite_score": round(self.composite_score, 4),
        }


class MetricsCalculator:
    """
    Calculates evaluation metrics for Tamil text pairs.

    Lazy-loads all heavy models on first use so that importing this module
    does not trigger model downloads in test environments.

    Args:
        embedding_model_name: HuggingFace model ID for semantic similarity.
            Defaults to intfloat/multilingual-e5-large (same as RAG system).
        device: "cpu" or "cuda".  Defaults to auto-detect.
        use_bertscore: Whether to compute BERTScore.  Set False to skip the
            bert_score library dependency during unit tests.
    """

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        use_bertscore: bool = False,
    ) -> None:
        """Initialise the calculator with model and device settings."""
        self._embedding_model_name = embedding_model_name
        self._use_bertscore = use_bertscore
        self._embed_model = None  # lazy
        self._device = device or self._auto_device()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        sample_id: str,
        reference: str,
        hypothesis: str,
    ) -> MetricResult:
        """
        Compute all metrics for a single (reference, hypothesis) pair.

        Args:
            sample_id: Identifier for the sample (used in the result).
            reference: Human-authored gold answer.
            hypothesis: System-generated (LLM) answer.

        Returns:
            MetricResult populated with all metric values.
        """
        reference = _normalise(reference)
        hypothesis = _normalise(hypothesis)

        if not reference or not hypothesis:
            logger.warning(
                "Sample %s: empty reference or hypothesis after normalisation. "
                "All scores will be 0.",
                sample_id,
            )
            return MetricResult(sample_id=sample_id)

        result = MetricResult(sample_id=sample_id)

        # 1. Semantic similarity
        result.semantic_similarity = self._semantic_similarity(reference, hypothesis)

        # 3. BLEU (character-level)
        result.bleu_1, result.bleu_2 = self._bleu(reference, hypothesis)

        # 4. ROUGE-L (character-level)
        result.rouge_l = self._rouge_l(reference, hypothesis)

        # 5. Composite
        result.composite_score = _composite(result)

        return result

    def compute_batch(
        self,
        ids: List[str],
        references: List[str],
        hypotheses: List[str],
    ) -> List[MetricResult]:
        """Compute metrics for multiple pairs.

        Embeddings are batched for efficiency; BERTScore and
        BLEU/ROUGE are computed per-item.

        Args:
            ids: Sample identifiers.
            references: List of human gold answers.
            hypotheses: List of system answers.

        Returns:
            List of MetricResult, one per input pair.
        """
        assert (
            len(ids) == len(references) == len(hypotheses)
        ), "ids, references, and hypotheses must have the same length."

        refs_norm = [_normalise(r) for r in references]
        hyps_norm = [_normalise(h) for h in hypotheses]

        # Batch-embed for efficiency
        sims = self._semantic_similarity_batch(refs_norm, hyps_norm)

        results: List[MetricResult] = []
        for i, (sid, ref, hyp) in enumerate(zip(ids, refs_norm, hyps_norm)):
            r = MetricResult(sample_id=sid)
            r.semantic_similarity = sims[i]

            if not ref or not hyp:
                results.append(r)
                continue

            r.bleu_1, r.bleu_2 = self._bleu(ref, hyp)
            r.rouge_l = self._rouge_l(ref, hyp)
            r.composite_score = _composite(r)
            results.append(r)

        return results

    # ------------------------------------------------------------------
    # Internal: semantic similarity
    # ------------------------------------------------------------------

    def _get_embed_model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info(
                "Loading embedding model '%s' on %s ...",
                self._embedding_model_name,
                self._device,
            )
            self._embed_model = SentenceTransformer(
                self._embedding_model_name,
                device=self._device,
            )
            logger.info("Embedding model loaded.")
        return self._embed_model

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts with the E5 query prefix, returns (N, D) array."""
        prefixed = [f"{E5_EVAL_PREFIX}{t}" for t in texts]
        model = self._get_embed_model()
        # normalize_embeddings=True ensures cosine similarity == dot product
        embeddings = model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return np.array(embeddings)

    def _semantic_similarity(self, ref: str, hyp: str) -> float:
        """Cosine similarity between a single reference/hypothesis pair."""
        vecs = self._encode([ref, hyp])
        # dot product of two L2-normalised vectors == cosine similarity
        score = float(np.dot(vecs[0], vecs[1]))
        # Clamp to [0, 1] — cosine can technically be negative
        return max(0.0, min(1.0, score))

    def _semantic_similarity_batch(
        self, refs: List[str], hyps: List[str]
    ) -> List[float]:
        """Batch cosine similarity — encodes all texts in one forward pass."""
        # Interleave refs and hyps so the model sees them in a single batch
        combined = refs + hyps
        vecs = self._encode(combined)
        n = len(refs)
        ref_vecs = vecs[:n]
        hyp_vecs = vecs[n:]
        # Element-wise dot product along dim=1
        sims = np.einsum("ij,ij->i", ref_vecs, hyp_vecs)
        return [max(0.0, min(1.0, float(s))) for s in sims]

    # ------------------------------------------------------------------
    # Internal: BLEU (character-level)
    # ------------------------------------------------------------------

    @staticmethod
    def _bleu(ref: str, hyp: str) -> Tuple[float, float]:
        """
        Character-level BLEU-1 and BLEU-2 using sacrebleu.

        Why character-level for Tamil?
        Tamil is an agglutinative language — a single written "word" may
        encode subject, object, tense, and aspect.  Morphological variants
        of the same root look completely different at the word level but share
        most of their characters.  Character n-grams capture this overlap.

        sacrebleu's `BLEU` scorer accepts pre-tokenised inputs; we pre-tokenise
        by splitting into individual Unicode characters (preserving the full
        code-point sequence) and joining with spaces so sacrebleu treats each
        character as a "word".
        """
        try:
            from sacrebleu.metrics import BLEU  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "sacrebleu not installed. Skipping BLEU. "
                "Install with: pip install sacrebleu"
            )
            return 0.0, 0.0

        # Tokenise at character level
        ref_chars = " ".join(list(ref))
        hyp_chars = " ".join(list(hyp))

        # BLEU-1
        bleu1_scorer = BLEU(max_ngram_order=1, smooth_method="add-k", smooth_value=1)
        b1_result = bleu1_scorer.corpus_score([hyp_chars], [[ref_chars]])
        bleu1 = b1_result.score / 100.0  # sacrebleu returns 0–100

        # BLEU-2
        bleu2_scorer = BLEU(max_ngram_order=2, smooth_method="add-k", smooth_value=1)
        b2_result = bleu2_scorer.corpus_score([hyp_chars], [[ref_chars]])
        bleu2 = b2_result.score / 100.0

        return max(0.0, bleu1), max(0.0, bleu2)

    # ------------------------------------------------------------------
    # Internal: ROUGE-L (character-level)
    # ------------------------------------------------------------------

    @staticmethod
    def _rouge_l(ref: str, hyp: str) -> float:
        """
        Character-level ROUGE-L F-measure via rouge-score.

        rouge-score's `rouge_scorer` expects whitespace-tokenised strings.
        We join individual characters with spaces so the scorer treats each
        Tamil Unicode code-point as a separate token.
        """
        try:
            from rouge_score import rouge_scorer  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "rouge_score not installed. Skipping ROUGE-L. "
                "Install with: pip install rouge-score"
            )
            return 0.0

        ref_chars = " ".join(list(ref))
        hyp_chars = " ".join(list(hyp))

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(ref_chars, hyp_chars)
        return float(scores["rougeL"].fmeasure)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_device() -> str:
        """Detect whether CUDA is available and return the device string."""
        try:
            import torch  # noqa: PLC0415

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """
    Normalise Tamil (and mixed) text before metric computation.

    Steps:
    1. Unicode NFC normalisation — ensures consistent code-point sequences
       for composed vs decomposed Tamil characters (e.g., vowel signs).
    2. Collapse multiple whitespace characters to a single space.
    3. Strip leading/trailing whitespace.

    We deliberately do NOT remove punctuation or lowercase because Tamil
    punctuation (।, ?, !) carries sentence-boundary information used by
    character n-gram metrics, and Tamil is case-insensitive by nature.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    # Collapse horizontal whitespace (including NBSP, ZWNJ common in Tamil)
    import re  # noqa: PLC0415

    text = re.sub(r"[ \t\u00a0\u200b\u200c\u200d]+", " ", text)
    return text.strip()


def _composite(r: MetricResult) -> float:
    """
    Compute the weighted composite score from individual metric values.

    Weights (see module docstring for rationale):
        semantic_similarity : 0.70
        rouge_l             : 0.20
        bleu_1              : 0.10
    """
    return (
        WEIGHT_SEMANTIC * r.semantic_similarity
        + WEIGHT_ROUGE_L * r.rouge_l
        + WEIGHT_BLEU1 * r.bleu_1
    )
