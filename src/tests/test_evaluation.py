"""
Tests for the PonniRAG evaluation module.

Coverage targets:
  - evaluation/dataset.py  : EvalSample, load_dataset (JSON + CSV), save_dataset,
                             malformed-input handling
  - evaluation/metrics.py  : MetricsCalculator, MetricResult, _normalise,
                             _composite, graceful degradation when optional
                             deps (bert-score, sacrebleu, rouge-score) are absent
  - evaluation/evaluate.py : run_evaluation, _fill_llm_answers, _build_report,
                             EvaluationReport, --live path with mocked ask_question

Mocking strategy
----------------
- sentence_transformers is already mocked in conftest.py (MockSentenceTransformer).
- bert_score, sacrebleu, rouge_score are mocked per-test using unittest.mock.patch
  so that no real model downloads occur.
- ask_question from hybrid_search is always mocked; the real RAG stack is never
  called.  Tests that call the live path are NOT marked @pytest.mark.integration
  because they mock ask_question internally.
- Heavy MetricsCalculator calls are replaced with patch.object / MagicMock where
  we care about pipeline logic rather than numerical accuracy.

Import note
-----------
conftest.py adds src/ to sys.path, so evaluation.* modules are importable
without further path manipulation here.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure evaluation package is importable (mirrors conftest.py pattern)
# ---------------------------------------------------------------------------
_src_dir = Path(__file__).resolve().parents[1]   # src/
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from evaluation.dataset import EvalSample, load_dataset, save_dataset  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    MetricResult,
    MetricsCalculator,
    _composite,
    _normalise,
)


# ===========================================================================
# Helpers / shared fixtures
# ===========================================================================

MINIMAL_JSON = [
    {
        "id": "q1",
        "question": "பொன்னி இதழ் எந்த ஆண்டில் தொடங்கியது?",
        "human_answer": "பொன்னி இதழ் 1947 ஆம் ஆண்டில் தொடங்கியது.",
        "llm_answer": "1947 ஆம் ஆண்டில் பொன்னி தொடங்கியது.",
    },
    {
        "id": "q2",
        "question": "தமிழ் இலக்கியம் என்றால் என்ன?",
        "human_answer": "தமிழ் இலக்கியம் உலகின் மிகப் பழமையான இலக்கியங்களில் ஒன்றாகும்.",
        "llm_answer": None,
    },
]

ANNOTATED_JSON = [
    {
        "id": "q3",
        "question": "சங்க காலம் என்றால் என்ன?",
        "human_answer": "சங்க காலம் தமிழ் இலக்கியத்தின் பொற்காலம்.",
        "llm_answer": "சங்க காலம் தமிழ் இலக்கியத்தின் சிறந்த காலம்.",
        "category": "literary_history",
        "notes": "Basic period question",
    }
]

TAMIL_REF = "தமிழ் மொழி உலகின் மிகப் பழமையான செம்மொழிகளில் ஒன்றாகும்."
TAMIL_HYP = "தமிழ் மொழி மிகவும் பழமையான மொழியாகும்."


def _write_json(data: list, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _make_eval_sample(
    sid: str = "s1",
    question: str = "கேள்வி?",
    human_answer: str = "மனித பதில்.",
    llm_answer: str | None = "LLM பதில்.",
    category: str | None = "test",
    notes: str | None = None,
) -> EvalSample:
    return EvalSample(
        id=sid,
        question=question,
        human_answer=human_answer,
        llm_answer=llm_answer,
        category=category,
        notes=notes,
    )


# ===========================================================================
# Tests for evaluation/dataset.py
# ===========================================================================


class TestEvalSample:
    """Unit tests for the EvalSample dataclass."""

    def test_is_ready_when_both_answers_present(self):
        s = _make_eval_sample(human_answer="ref", llm_answer="hyp")
        assert s.is_ready() is True

    def test_is_ready_when_llm_answer_is_none(self):
        s = _make_eval_sample(llm_answer=None)
        assert s.is_ready() is False

    def test_is_ready_when_llm_answer_is_empty_string(self):
        # llm_answer="" is falsy — is_ready should return False
        s = _make_eval_sample(llm_answer="")
        assert s.is_ready() is False

    def test_is_ready_when_human_answer_empty(self):
        s = _make_eval_sample(human_answer="", llm_answer="some answer")
        assert s.is_ready() is False

    def test_to_dict_keys(self):
        s = _make_eval_sample()
        d = s.to_dict()
        assert set(d.keys()) == {
            "id", "question", "human_answer", "llm_answer", "category", "notes"
        }

    def test_to_dict_values_round_trip(self):
        s = EvalSample(
            id="x1",
            question="கேள்வி",
            human_answer="ref",
            llm_answer="hyp",
            category="cat",
            notes="note",
        )
        d = s.to_dict()
        assert d["id"] == "x1"
        assert d["question"] == "கேள்வி"
        assert d["human_answer"] == "ref"
        assert d["llm_answer"] == "hyp"
        assert d["category"] == "cat"
        assert d["notes"] == "note"

    def test_to_dict_with_none_fields(self):
        s = EvalSample(id="x2", question="q", human_answer="h", llm_answer=None)
        d = s.to_dict()
        assert d["llm_answer"] is None
        assert d["category"] is None
        assert d["notes"] is None


class TestLoadDatasetJSON:
    """Tests for load_dataset with JSON files."""

    def test_load_minimal_json(self, tmp_path):
        p = tmp_path / "data.json"
        _write_json(MINIMAL_JSON, p)
        samples = load_dataset(p)
        assert len(samples) == 2
        assert samples[0].id == "q1"
        assert samples[1].id == "q2"

    def test_load_annotated_json(self, tmp_path):
        p = tmp_path / "data.json"
        _write_json(ANNOTATED_JSON, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].category == "literary_history"
        assert samples[0].notes == "Basic period question"

    def test_llm_answer_none_kept_as_none(self, tmp_path):
        p = tmp_path / "data.json"
        _write_json(MINIMAL_JSON, p)
        samples = load_dataset(p)
        # q2 has llm_answer: null
        assert samples[1].llm_answer is None

    def test_llm_answer_populated(self, tmp_path):
        p = tmp_path / "data.json"
        _write_json(MINIMAL_JSON, p)
        samples = load_dataset(p)
        assert samples[0].llm_answer == "1947 ஆம் ஆண்டில் பொன்னி தொடங்கியது."

    def test_auto_generate_id_when_absent(self, tmp_path):
        data = [
            {"question": "கேள்வி?", "human_answer": "பதில்."},
        ]
        p = tmp_path / "data.json"
        _write_json(data, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].id == "sample_1"

    def test_skip_entry_missing_question(self, tmp_path):
        data = [
            {"id": "a", "question": "", "human_answer": "பதில்."},
            {"id": "b", "question": "கேள்வி?", "human_answer": "பதில்."},
        ]
        p = tmp_path / "data.json"
        _write_json(data, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].id == "b"

    def test_skip_entry_missing_human_answer(self, tmp_path):
        data = [
            {"id": "c", "question": "கேள்வி?", "human_answer": ""},
            {"id": "d", "question": "மற்ற கேள்வி?", "human_answer": "பதில்."},
        ]
        p = tmp_path / "data.json"
        _write_json(data, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].id == "d"

    def test_skip_non_dict_entries(self, tmp_path):
        data = [
            "not_a_dict",
            {"id": "e", "question": "கேள்வி?", "human_answer": "பதில்."},
        ]
        p = tmp_path / "data.json"
        _write_json(data, p)
        samples = load_dataset(p)
        assert len(samples) == 1

    def test_empty_json_array(self, tmp_path):
        p = tmp_path / "data.json"
        _write_json([], p)
        samples = load_dataset(p)
        assert samples == []

    def test_raises_value_error_for_non_array_json(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        with pytest.raises(ValueError, match="top-level array"):
            load_dataset(p)

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.json")

    def test_raises_value_error_for_unsupported_extension(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported dataset format"):
            load_dataset(p)

    def test_llm_answer_whitespace_only_becomes_none(self, tmp_path):
        data = [
            {
                "id": "ws",
                "question": "கேள்வி?",
                "human_answer": "பதில்.",
                "llm_answer": "   ",
            }
        ]
        p = tmp_path / "data.json"
        _write_json(data, p)
        samples = load_dataset(p)
        assert samples[0].llm_answer is None


class TestLoadDatasetCSV:
    """Tests for load_dataset with CSV files."""

    def _write_csv(self, rows: list[dict], path: Path) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_load_minimal_csv(self, tmp_path):
        rows = [
            {
                "id": "c1",
                "question": "கேள்வி?",
                "human_answer": "பதில்.",
                "llm_answer": "LLM பதில்.",
            }
        ]
        p = tmp_path / "data.csv"
        self._write_csv(rows, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].id == "c1"
        assert samples[0].llm_answer == "LLM பதில்."

    def test_load_csv_without_llm_answer_column(self, tmp_path):
        rows = [
            {"id": "c2", "question": "கேள்வி?", "human_answer": "பதில்."}
        ]
        p = tmp_path / "data.csv"
        self._write_csv(rows, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].llm_answer is None

    def test_csv_skips_rows_with_empty_question(self, tmp_path):
        rows = [
            {"id": "r1", "question": "", "human_answer": "பதில்."},
            {"id": "r2", "question": "கேள்வி?", "human_answer": "பதில்."},
        ]
        p = tmp_path / "data.csv"
        self._write_csv(rows, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].id == "r2"

    def test_csv_skips_rows_with_empty_human_answer(self, tmp_path):
        rows = [
            {"id": "r3", "question": "கேள்வி?", "human_answer": ""},
            {"id": "r4", "question": "மற்ற கேள்வி?", "human_answer": "பதில்."},
        ]
        p = tmp_path / "data.csv"
        self._write_csv(rows, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].id == "r4"

    def test_csv_raises_on_missing_required_columns(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("id,llm_answer\nq1,ans\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            load_dataset(p)

    def test_csv_auto_generates_id(self, tmp_path):
        rows = [{"question": "கேள்வி?", "human_answer": "பதில்."}]
        p = tmp_path / "data.csv"
        self._write_csv(rows, p)
        samples = load_dataset(p)
        assert samples[0].id == "sample_1"

    def test_csv_with_category_column(self, tmp_path):
        rows = [
            {
                "id": "cat1",
                "question": "கேள்வி?",
                "human_answer": "பதில்.",
                "category": "history",
            }
        ]
        p = tmp_path / "data.csv"
        self._write_csv(rows, p)
        samples = load_dataset(p)
        assert samples[0].category == "history"


class TestSaveDataset:
    """Tests for save_dataset."""

    def test_save_and_reload(self, tmp_path):
        samples = [
            _make_eval_sample("s1", "கேள்வி 1?", "ref1", "hyp1"),
            _make_eval_sample("s2", "கேள்வி 2?", "ref2", None),
        ]
        out = tmp_path / "out.json"
        save_dataset(samples, out)
        assert out.exists()
        reloaded = load_dataset(out)
        assert len(reloaded) == 2
        assert reloaded[0].id == "s1"
        assert reloaded[1].llm_answer is None

    def test_save_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "out.json"
        save_dataset([_make_eval_sample()], out)
        assert out.exists()

    def test_save_preserves_tamil_unicode(self, tmp_path):
        s = _make_eval_sample(human_answer="தமிழ் மொழி")
        out = tmp_path / "out.json"
        save_dataset([s], out)
        text = out.read_text(encoding="utf-8")
        assert "தமிழ்" in text

    def test_save_empty_list(self, tmp_path):
        out = tmp_path / "empty.json"
        save_dataset([], out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == []


# ===========================================================================
# Tests for evaluation/metrics.py
# ===========================================================================


class TestNormalise:
    """Tests for the _normalise module-level helper."""

    def test_nfc_normalisation(self):
        # Tamil vowel sign composed vs decomposed — both should normalise to NFC
        composed = "\u0BA4\u0BAE\u0BBF\u0BB4\u0BCD"        # தமிழ் in NFC
        decomposed = "\u0BA4\u0BAE\u0BBF\u0BB4\u0BCD"      # same chars
        assert _normalise(composed) == _normalise(decomposed)

    def test_collapses_multiple_spaces(self):
        assert _normalise("hello   world") == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        assert _normalise("  hello  ") == "hello"

    def test_collapses_non_breaking_space(self):
        # \u00a0 is a non-breaking space common in Tamil text
        result = _normalise("hello\u00a0world")
        assert result == "hello world"

    def test_collapses_zero_width_non_joiner(self):
        # \u200c (ZWNJ) is common in Tamil digital text.
        # The _normalise regex replaces it with a single space (it is in the
        # [ \t\u00a0\u200b\u200c\u200d]+ character class), then strips
        # leading/trailing but not interior whitespace.
        result = _normalise("hello\u200cworld")
        assert result == "hello world"

    def test_empty_string_returns_empty(self):
        assert _normalise("") == ""

    def test_plain_ascii(self):
        assert _normalise("hello world") == "hello world"

    def test_tamil_text_unchanged_otherwise(self):
        text = "தமிழ் மொழி பழமையானது."
        result = _normalise(text)
        assert "தமிழ்" in result
        assert result == result.strip()


class TestComposite:
    """Tests for the _composite helper function."""

    def test_all_zeros_gives_zero(self):
        r = MetricResult(sample_id="x")
        assert _composite(r) == 0.0

    def test_all_ones_gives_one(self):
        r = MetricResult(
            sample_id="x",
            semantic_similarity=1.0,
            bertscore_f1=1.0,
            rouge_l=1.0,
            bleu_1=1.0,
        )
        assert abs(_composite(r) - 1.0) < 1e-9

    def test_weights_applied_correctly(self):
        # Only semantic set: composite should be 0.50 * 0.8 = 0.40
        r = MetricResult(sample_id="x", semantic_similarity=0.8)
        assert abs(_composite(r) - 0.40) < 1e-9

    def test_bertscore_weight(self):
        r = MetricResult(sample_id="x", bertscore_f1=1.0)
        assert abs(_composite(r) - 0.25) < 1e-9

    def test_rouge_l_weight(self):
        r = MetricResult(sample_id="x", rouge_l=1.0)
        assert abs(_composite(r) - 0.15) < 1e-9

    def test_bleu1_weight(self):
        r = MetricResult(sample_id="x", bleu_1=1.0)
        assert abs(_composite(r) - 0.10) < 1e-9


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_to_dict_keys(self):
        r = MetricResult(sample_id="q1")
        d = r.to_dict()
        assert set(d.keys()) == {
            "id",
            "semantic_similarity",
            "bertscore_f1",
            "bleu_1",
            "bleu_2",
            "rouge_l",
            "composite_score",
        }

    def test_to_dict_rounding(self):
        r = MetricResult(sample_id="q1", semantic_similarity=0.123456789)
        d = r.to_dict()
        assert d["semantic_similarity"] == 0.1235  # rounded to 4 places

    def test_default_values_are_zero(self):
        r = MetricResult(sample_id="q1")
        assert r.semantic_similarity == 0.0
        assert r.bertscore_f1 == 0.0
        assert r.bleu_1 == 0.0
        assert r.rouge_l == 0.0
        assert r.composite_score == 0.0


class TestMetricsCalculatorInit:
    """Tests for MetricsCalculator construction (no heavy model loads)."""

    def test_default_model_name(self):
        calc = MetricsCalculator(use_bertscore=False)
        assert calc._embedding_model_name == "intfloat/multilingual-e5-large"

    def test_custom_model_name(self):
        calc = MetricsCalculator(
            embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            use_bertscore=False,
        )
        assert "paraphrase" in calc._embedding_model_name

    def test_use_bertscore_flag_stored(self):
        calc = MetricsCalculator(use_bertscore=False)
        assert calc._use_bertscore is False

    def test_embed_model_is_none_before_first_use(self):
        calc = MetricsCalculator(use_bertscore=False)
        assert calc._embed_model is None

    def test_device_defaults_to_cpu_when_torch_absent(self):
        # Simulate torch not installed
        with patch.dict("sys.modules", {"torch": None}):
            calc = MetricsCalculator(use_bertscore=False)
        assert calc._device in ("cpu", "cuda")  # auto-detect; CI typically = cpu

    def test_explicit_device_respected(self):
        calc = MetricsCalculator(device="cpu", use_bertscore=False)
        assert calc._device == "cpu"


class TestMetricsCalculatorCompute:
    """
    Tests for MetricsCalculator.compute() with mocked model internals.

    conftest.py already patches sentence_transformers.SentenceTransformer.
    We additionally mock the internal _encode and _bertscore methods so
    that compute() exercises its logic paths without real model calls.
    """

    def _make_calculator(self, use_bertscore: bool = False) -> MetricsCalculator:
        calc = MetricsCalculator(device="cpu", use_bertscore=use_bertscore)
        # Inject a mock embed model that returns normalised unit vectors
        mock_model = MagicMock()
        # Return two L2-normalised 4-dim vectors; cosine similarity = dot product
        mock_model.encode.return_value = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.8, 0.6, 0.0, 0.0]]
        )
        calc._embed_model = mock_model
        return calc

    def test_compute_returns_metric_result(self):
        calc = self._make_calculator()
        with patch.object(
            calc, "_bleu", return_value=(0.5, 0.4)
        ), patch.object(calc, "_rouge_l", return_value=0.6):
            result = calc.compute("q1", TAMIL_REF, TAMIL_HYP)
        assert isinstance(result, MetricResult)
        assert result.sample_id == "q1"

    def test_compute_semantic_similarity_in_range(self):
        calc = self._make_calculator()
        with patch.object(
            calc, "_bleu", return_value=(0.5, 0.4)
        ), patch.object(calc, "_rouge_l", return_value=0.6):
            result = calc.compute("q1", TAMIL_REF, TAMIL_HYP)
        assert 0.0 <= result.semantic_similarity <= 1.0

    def test_compute_empty_reference_returns_zero_result(self):
        calc = self._make_calculator()
        result = calc.compute("q_empty", "", TAMIL_HYP)
        assert result.semantic_similarity == 0.0
        assert result.composite_score == 0.0

    def test_compute_empty_hypothesis_returns_zero_result(self):
        calc = self._make_calculator()
        result = calc.compute("q_empty", TAMIL_REF, "")
        assert result.semantic_similarity == 0.0

    def test_compute_with_bertscore_enabled(self):
        calc = self._make_calculator(use_bertscore=True)
        with patch.object(
            calc, "_bertscore", return_value=(0.7, 0.8, 0.75)
        ), patch.object(
            calc, "_bleu", return_value=(0.5, 0.4)
        ), patch.object(
            calc, "_rouge_l", return_value=0.6
        ):
            result = calc.compute("q1", TAMIL_REF, TAMIL_HYP)
        assert result.bertscore_f1 == 0.75
        assert result.bertscore_precision == 0.7
        assert result.bertscore_recall == 0.8

    def test_compute_bertscore_disabled_leaves_zeros(self):
        calc = self._make_calculator(use_bertscore=False)
        with patch.object(
            calc, "_bleu", return_value=(0.5, 0.4)
        ), patch.object(calc, "_rouge_l", return_value=0.6):
            result = calc.compute("q1", TAMIL_REF, TAMIL_HYP)
        assert result.bertscore_f1 == 0.0
        assert result.bertscore_precision == 0.0
        assert result.bertscore_recall == 0.0

    def test_compute_composite_matches_formula(self):
        calc = self._make_calculator(use_bertscore=False)
        with patch.object(
            calc, "_semantic_similarity", return_value=0.8
        ), patch.object(
            calc, "_bleu", return_value=(0.5, 0.3)
        ), patch.object(
            calc, "_rouge_l", return_value=0.6
        ):
            result = calc.compute("q1", TAMIL_REF, TAMIL_HYP)
        expected = 0.50 * 0.8 + 0.25 * 0.0 + 0.15 * 0.6 + 0.10 * 0.5
        assert abs(result.composite_score - expected) < 1e-9


class TestMetricsCalculatorComputeBatch:
    """Tests for MetricsCalculator.compute_batch()."""

    def _make_calculator(self) -> MetricsCalculator:
        calc = MetricsCalculator(device="cpu", use_bertscore=False)
        # encode returns interleaved ref+hyp vectors; batch of 4 texts -> 4 vecs
        mock_model = MagicMock()

        def mock_encode(prefixed, **kwargs):
            n = len(prefixed)
            # Return random normalised vectors
            vecs = np.random.rand(n, 4)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms

        mock_model.encode.side_effect = mock_encode
        calc._embed_model = mock_model
        return calc

    def test_batch_returns_correct_length(self):
        calc = self._make_calculator()
        ids = ["q1", "q2"]
        refs = [TAMIL_REF, "ref2"]
        hyps = [TAMIL_HYP, "hyp2"]
        with patch.object(
            calc, "_bleu", return_value=(0.5, 0.4)
        ), patch.object(calc, "_rouge_l", return_value=0.6):
            results = calc.compute_batch(ids, refs, hyps)
        assert len(results) == 2

    def test_batch_sample_ids_preserved(self):
        calc = self._make_calculator()
        ids = ["alpha", "beta"]
        refs = ["ref_a", "ref_b"]
        hyps = ["hyp_a", "hyp_b"]
        with patch.object(
            calc, "_bleu", return_value=(0.4, 0.3)
        ), patch.object(calc, "_rouge_l", return_value=0.5):
            results = calc.compute_batch(ids, refs, hyps)
        assert results[0].sample_id == "alpha"
        assert results[1].sample_id == "beta"

    def test_batch_raises_on_length_mismatch(self):
        calc = self._make_calculator()
        with pytest.raises(AssertionError):
            calc.compute_batch(["q1"], ["ref1", "ref2"], ["hyp1"])

    def test_batch_empty_ref_gives_zero_non_semantic(self):
        calc = self._make_calculator()
        ids = ["q_empty"]
        refs = [""]
        hyps = ["some answer"]
        # compute_batch normalises and skips BLEU/ROUGE when ref is empty
        results = calc.compute_batch(ids, refs, hyps)
        assert results[0].bleu_1 == 0.0
        assert results[0].rouge_l == 0.0


class TestBLEUGracefulDegradation:
    """Test that _bleu returns (0, 0) gracefully if sacrebleu is absent."""

    def test_bleu_returns_zeros_when_sacrebleu_missing(self):
        with patch.dict("sys.modules", {"sacrebleu": None, "sacrebleu.metrics": None}):
            result = MetricsCalculator._bleu("ref text", "hyp text")
        assert result == (0.0, 0.0)


class TestROUGEGracefulDegradation:
    """Test that _rouge_l returns 0.0 gracefully if rouge-score is absent."""

    def test_rouge_returns_zero_when_rouge_score_missing(self):
        with patch.dict("sys.modules", {"rouge_score": None}):
            result = MetricsCalculator._rouge_l("ref text", "hyp text")
        assert result == 0.0


class TestBERTScoreGracefulDegradation:
    """Test that _bertscore returns (0, 0, 0) gracefully if bert_score absent."""

    def test_bertscore_returns_zeros_when_bert_score_missing(self):
        calc = MetricsCalculator(device="cpu", use_bertscore=True)
        with patch.dict("sys.modules", {"bert_score": None}):
            p, r, f1 = calc._bertscore("ref", "hyp")
        assert (p, r, f1) == (0.0, 0.0, 0.0)

    def test_bertscore_returns_zeros_on_exception(self):
        calc = MetricsCalculator(device="cpu", use_bertscore=True)
        mock_bert_score_module = MagicMock()
        mock_bert_score_module.score.side_effect = RuntimeError("CUDA OOM")
        with patch.dict("sys.modules", {"bert_score": mock_bert_score_module}):
            p, r, f1 = calc._bertscore("ref", "hyp")
        assert (p, r, f1) == (0.0, 0.0, 0.0)


class TestBLEUCharacterLevel:
    """Smoke tests for character-level BLEU when sacrebleu is available."""

    def test_bleu_identical_strings(self):
        # Mock sacrebleu to return a perfect score
        mock_scorer = MagicMock()
        mock_result = MagicMock()
        mock_result.score = 100.0
        mock_scorer.corpus_score.return_value = mock_result

        mock_bleu_cls = MagicMock(return_value=mock_scorer)
        mock_sacrebleu = MagicMock()
        mock_sacrebleu.BLEU = mock_bleu_cls

        with patch.dict("sys.modules", {"sacrebleu": MagicMock(), "sacrebleu.metrics": mock_sacrebleu}):
            b1, b2 = MetricsCalculator._bleu("hello", "hello")
        assert b1 == 1.0

    def test_bleu_scores_clamped_to_zero(self):
        mock_scorer = MagicMock()
        mock_result = MagicMock()
        mock_result.score = -5.0  # sacrebleu can return negative with smoothing
        mock_scorer.corpus_score.return_value = mock_result

        mock_bleu_cls = MagicMock(return_value=mock_scorer)
        mock_sacrebleu = MagicMock()
        mock_sacrebleu.BLEU = mock_bleu_cls

        with patch.dict("sys.modules", {"sacrebleu": MagicMock(), "sacrebleu.metrics": mock_sacrebleu}):
            b1, b2 = MetricsCalculator._bleu("ref", "hyp")
        assert b1 >= 0.0
        assert b2 >= 0.0


class TestROUGECharacterLevel:
    """Smoke tests for character-level ROUGE when rouge-score is available."""

    def test_rouge_l_uses_character_tokens(self):
        # Verify that the scorer is called with character-spaced strings
        mock_scorer_instance = MagicMock()
        mock_scores = {"rougeL": MagicMock(fmeasure=0.75)}
        mock_scorer_instance.score.return_value = mock_scores

        mock_rouge_scorer_cls = MagicMock(return_value=mock_scorer_instance)
        mock_rouge_score_module = MagicMock()
        mock_rouge_score_module.rouge_scorer.RougeScorer = mock_rouge_scorer_cls

        with patch.dict("sys.modules", {"rouge_score": mock_rouge_score_module}):
            result = MetricsCalculator._rouge_l("ab", "ab")

        # The scorer must have been called with character-spaced strings
        call_args = mock_scorer_instance.score.call_args
        ref_arg, hyp_arg = call_args[0]
        # "ab" -> "a b"
        assert " " in ref_arg
        assert result == 0.75


# ===========================================================================
# Tests for evaluation/evaluate.py
# ===========================================================================

# We import evaluate functions here (after conftest adds src/ to sys.path)
from evaluation.evaluate import (  # noqa: E402
    EvaluationReport,
    _build_report,
    _fill_llm_answers,
    run_evaluation,
)


class TestEvaluationReport:
    """Tests for the EvaluationReport dataclass."""

    def test_to_dict_structure(self):
        report = EvaluationReport(
            dataset_path="/tmp/data.json",
            total_samples=3,
            evaluated_samples=2,
            skipped_samples=1,
            avg_semantic_similarity=0.8,
            avg_bertscore_f1=0.7,
            avg_bleu_1=0.5,
            avg_bleu_2=0.4,
            avg_rouge_l=0.6,
            avg_composite_score=0.72,
            per_sample=[{"id": "q1"}],
        )
        d = report.to_dict()
        assert d["dataset_path"] == "/tmp/data.json"
        assert d["total_samples"] == 3
        assert d["evaluated_samples"] == 2
        assert d["skipped_samples"] == 1
        assert "averages" in d
        assert d["averages"]["semantic_similarity"] == 0.8
        assert "per_sample" in d

    def test_to_dict_rounds_averages(self):
        report = EvaluationReport(
            dataset_path="x",
            total_samples=1,
            evaluated_samples=1,
            skipped_samples=0,
            avg_composite_score=0.123456789,
        )
        d = report.to_dict()
        assert d["averages"]["composite_score"] == 0.1235


class TestBuildReport:
    """Tests for the _build_report internal function."""

    def _make_result(self, sid: str, sem: float = 0.8, bf1: float = 0.7) -> MetricResult:
        r = MetricResult(
            sample_id=sid,
            semantic_similarity=sem,
            bertscore_f1=bf1,
            bleu_1=0.5,
            bleu_2=0.4,
            rouge_l=0.6,
        )
        r.composite_score = _composite(r)
        return r

    def _make_sample(self, sid: str) -> EvalSample:
        return _make_eval_sample(sid=sid)

    def test_build_report_correct_counts(self):
        samples = [self._make_sample("q1"), self._make_sample("q2")]
        results = [self._make_result("q1"), self._make_result("q2")]
        report = _build_report(
            dataset_path="/tmp/data.json",
            samples=samples,
            metric_results=results,
            skipped=1,
        )
        assert report.total_samples == 3   # 2 evaluated + 1 skipped
        assert report.evaluated_samples == 2
        assert report.skipped_samples == 1

    def test_build_report_macro_averages(self):
        samples = [self._make_sample("q1"), self._make_sample("q2")]
        r1 = self._make_result("q1", sem=0.8)
        r2 = self._make_result("q2", sem=0.6)
        report = _build_report("/d", samples, [r1, r2], skipped=0)
        assert abs(report.avg_semantic_similarity - 0.7) < 1e-9

    def test_build_report_per_sample_contains_question(self):
        s = _make_eval_sample(sid="q1", question="கேள்வி?")
        r = self._make_result("q1")
        report = _build_report("/d", [s], [r], skipped=0)
        assert report.per_sample[0]["question"] == "கேள்வி?"

    def test_build_report_empty_results_returns_zero_report(self):
        report = _build_report("/d", [], [], skipped=2)
        assert report.evaluated_samples == 0
        assert report.avg_semantic_similarity == 0.0


class TestFillLlmAnswers:
    """Tests for _fill_llm_answers()."""

    def test_no_live_with_all_answers_present(self):
        """When live=False and all answers present, no import of hybrid_search."""
        samples = [_make_eval_sample(llm_answer="existing answer")]
        result = _fill_llm_answers(samples, live=False)
        assert result[0].llm_answer == "existing answer"

    def test_live_false_missing_answer_triggers_import(self):
        """When live=False but a sample lacks llm_answer, ask_question is called."""
        sample = _make_eval_sample(llm_answer=None)
        mock_ask = MagicMock(return_value={"answer": "LLM generated answer"})
        with patch("evaluation.evaluate._fill_llm_answers.__module__"):
            pass
        # Patch the hybrid_search import inside _fill_llm_answers
        with patch.dict(
            "sys.modules",
            {"hybrid_search": MagicMock(ask_question=mock_ask)},
        ):
            result = _fill_llm_answers([sample], live=False)
        assert result[0].llm_answer == "LLM generated answer"

    def test_live_true_replaces_existing_answer(self):
        """When live=True, even samples with llm_answer get re-evaluated."""
        sample = _make_eval_sample(llm_answer="old answer")
        mock_ask = MagicMock(return_value={"answer": "new LLM answer"})
        with patch.dict(
            "sys.modules",
            {"hybrid_search": MagicMock(ask_question=mock_ask)},
        ):
            result = _fill_llm_answers([sample], live=True)
        assert result[0].llm_answer == "new LLM answer"

    def test_import_error_leaves_llm_answer_as_none(self):
        """If hybrid_search cannot be imported, llm_answer stays None."""
        sample = _make_eval_sample(llm_answer=None)
        # Remove hybrid_search from sys.modules to simulate ImportError
        original = sys.modules.pop("hybrid_search", None)
        try:
            with patch.dict("sys.modules", {"hybrid_search": None}):
                result = _fill_llm_answers([sample], live=False)
            assert result[0].llm_answer is None
        finally:
            if original is not None:
                sys.modules["hybrid_search"] = original

    def test_ask_question_exception_leaves_llm_answer_as_none(self):
        """If ask_question raises, llm_answer is set to None."""
        sample = _make_eval_sample(llm_answer=None)
        mock_ask = MagicMock(side_effect=RuntimeError("Qdrant timeout"))
        with patch.dict(
            "sys.modules",
            {"hybrid_search": MagicMock(ask_question=mock_ask)},
        ):
            result = _fill_llm_answers([sample], live=False)
        assert result[0].llm_answer is None

    def test_empty_answer_from_ask_question_becomes_none(self):
        """An empty string returned by ask_question becomes None."""
        sample = _make_eval_sample(llm_answer=None)
        mock_ask = MagicMock(return_value={"answer": "   "})
        with patch.dict(
            "sys.modules",
            {"hybrid_search": MagicMock(ask_question=mock_ask)},
        ):
            result = _fill_llm_answers([sample], live=False)
        assert result[0].llm_answer is None


class TestRunEvaluation:
    """
    Integration-style tests for run_evaluation() with fully mocked internals.

    MetricsCalculator.compute_batch is mocked so no model is loaded.
    """

    def _make_ready_json(self, tmp_path: Path, n: int = 2) -> Path:
        """Write a JSON dataset where all samples have llm_answers."""
        data = [
            {
                "id": f"q{i}",
                "question": f"கேள்வி {i}?",
                "human_answer": f"மனித பதில் {i}.",
                "llm_answer": f"LLM பதில் {i}.",
                "category": "test",
            }
            for i in range(1, n + 1)
        ]
        p = tmp_path / "dataset.json"
        _write_json(data, p)
        return p

    def _make_mock_calculator(self, n: int = 2) -> MagicMock:
        calc = MagicMock(spec=MetricsCalculator)
        results = [
            MetricResult(
                sample_id=f"q{i}",
                semantic_similarity=0.8,
                bertscore_f1=0.7,
                bleu_1=0.5,
                bleu_2=0.4,
                rouge_l=0.6,
                composite_score=0.69,
            )
            for i in range(1, n + 1)
        ]
        calc.compute_batch.return_value = results
        return calc

    def test_run_evaluation_returns_report(self, tmp_path):
        p = self._make_ready_json(tmp_path)
        mock_calc = self._make_mock_calculator()
        with patch("evaluation.evaluate.MetricsCalculator", return_value=mock_calc):
            report = run_evaluation(p, use_bertscore=False)
        assert isinstance(report, EvaluationReport)

    def test_run_evaluation_evaluated_count(self, tmp_path):
        p = self._make_ready_json(tmp_path, n=3)
        mock_calc = self._make_mock_calculator(n=3)
        with patch("evaluation.evaluate.MetricsCalculator", return_value=mock_calc):
            report = run_evaluation(p, use_bertscore=False)
        assert report.evaluated_samples == 3
        assert report.skipped_samples == 0

    def test_run_evaluation_skips_missing_llm_answer(self, tmp_path):
        data = [
            {
                "id": "q1",
                "question": "கேள்வி?",
                "human_answer": "ref",
                "llm_answer": "hyp",
            },
            {
                "id": "q2",
                "question": "மற்ற கேள்வி?",
                "human_answer": "ref2",
                "llm_answer": None,  # missing
            },
        ]
        p = tmp_path / "partial.json"
        _write_json(data, p)

        # Mock _fill_llm_answers to leave q2's llm_answer as None
        # (simulates no running RAG stack).  This isolates the skip-counting
        # logic in run_evaluation from the live-query path.
        def _no_op_fill(samples, live):
            return samples  # leave llm_answer=None on q2

        mock_calc = self._make_mock_calculator(n=1)
        with patch(
            "evaluation.evaluate._fill_llm_answers", side_effect=_no_op_fill
        ), patch("evaluation.evaluate.MetricsCalculator", return_value=mock_calc):
            report = run_evaluation(p, live=False, use_bertscore=False)
        assert report.skipped_samples == 1
        assert report.evaluated_samples == 1

    def test_run_evaluation_empty_dataset_returns_zero_report(self, tmp_path):
        p = tmp_path / "empty.json"
        _write_json([], p)
        report = run_evaluation(p)
        assert report.total_samples == 0
        assert report.evaluated_samples == 0

    def test_run_evaluation_saves_answers(self, tmp_path):
        p = self._make_ready_json(tmp_path, n=1)
        save_path = tmp_path / "answers.json"
        mock_calc = self._make_mock_calculator(n=1)
        with patch("evaluation.evaluate.MetricsCalculator", return_value=mock_calc):
            run_evaluation(p, use_bertscore=False, save_answers_path=save_path)
        assert save_path.exists()
        saved = json.loads(save_path.read_text(encoding="utf-8"))
        assert len(saved) == 1

    def test_run_evaluation_saves_report_json(self, tmp_path):
        p = self._make_ready_json(tmp_path, n=1)
        out_path = tmp_path / "report.json"
        mock_calc = self._make_mock_calculator(n=1)
        with patch("evaluation.evaluate.MetricsCalculator", return_value=mock_calc):
            run_evaluation(p, use_bertscore=False, output_path=out_path)
        assert out_path.exists()
        report_data = json.loads(out_path.read_text(encoding="utf-8"))
        assert "averages" in report_data
        assert "per_sample" in report_data

    def test_run_evaluation_live_mode_calls_ask_question(self, tmp_path):
        """--live flag should trigger ask_question for each sample."""
        data = [
            {
                "id": "q1",
                "question": "பொன்னி எப்போது தொடங்கியது?",
                "human_answer": "1947 ஆம் ஆண்டில்.",
                "llm_answer": None,
            }
        ]
        p = tmp_path / "live.json"
        _write_json(data, p)

        mock_ask = MagicMock(return_value={"answer": "1947 ஆம் ஆண்டில் தொடங்கியது."})
        mock_calc = self._make_mock_calculator(n=1)
        with patch.dict(
            "sys.modules",
            {"hybrid_search": MagicMock(ask_question=mock_ask)},
        ), patch("evaluation.evaluate.MetricsCalculator", return_value=mock_calc):
            report = run_evaluation(p, live=True, use_bertscore=False)

        mock_ask.assert_called_once()
        assert report.evaluated_samples == 1

    def test_run_evaluation_composite_score_in_report(self, tmp_path):
        p = self._make_ready_json(tmp_path, n=2)
        mock_calc = self._make_mock_calculator(n=2)
        with patch("evaluation.evaluate.MetricsCalculator", return_value=mock_calc):
            report = run_evaluation(p, use_bertscore=False)
        # Both mock results have composite_score=0.69
        assert abs(report.avg_composite_score - 0.69) < 1e-6

    def test_run_evaluation_per_sample_has_expected_keys(self, tmp_path):
        p = self._make_ready_json(tmp_path, n=1)
        mock_calc = self._make_mock_calculator(n=1)
        with patch("evaluation.evaluate.MetricsCalculator", return_value=mock_calc):
            report = run_evaluation(p, use_bertscore=False)
        entry = report.per_sample[0]
        assert "id" in entry
        assert "semantic_similarity" in entry
        assert "composite_score" in entry
        assert "question" in entry


# ===========================================================================
# Smoke tests — fast end-to-end check without downloading any models
# ===========================================================================


class TestSmokeEndToEnd:
    """
    Lightweight smoke tests that exercise the full call stack.

    These use only tmp_path files and mock every heavyweight call.
    They serve as a regression guard: if the module interfaces change
    (e.g., EvalSample.to_dict() adds/removes a key, run_evaluation
    changes its return type), these tests will catch it immediately.
    """

    def test_full_offline_pipeline(self, tmp_path):
        """
        Write a JSON dataset with pre-filled llm_answers, run evaluation,
        confirm the report has sane structure.
        """
        data = [
            {
                "id": "smoke_q1",
                "question": "தமிழ் மொழி என்று என்ன?",
                "human_answer": "தமிழ் மொழி செம்மொழி.",
                "llm_answer": "தமிழ் மொழி ஒரு செம்மொழி.",
                "category": "language",
            }
        ]
        p = tmp_path / "smoke.json"
        _write_json(data, p)

        # Return a realistic MetricResult without model inference
        fake_result = MetricResult(
            sample_id="smoke_q1",
            semantic_similarity=0.85,
            bertscore_f1=0.80,
            bleu_1=0.60,
            bleu_2=0.50,
            rouge_l=0.70,
            composite_score=0.80,
        )
        mock_calc = MagicMock(spec=MetricsCalculator)
        mock_calc.compute_batch.return_value = [fake_result]

        with patch("evaluation.evaluate.MetricsCalculator", return_value=mock_calc):
            report = run_evaluation(p, live=False, use_bertscore=False)

        assert report.total_samples == 1
        assert report.evaluated_samples == 1
        assert report.skipped_samples == 0
        assert report.avg_semantic_similarity == pytest.approx(0.85)
        assert len(report.per_sample) == 1
        assert report.per_sample[0]["id"] == "smoke_q1"

    def test_dataset_round_trip(self, tmp_path):
        """save_dataset -> load_dataset preserves all fields."""
        original = [
            EvalSample(
                id="rt1",
                question="கேள்வி",
                human_answer="மனித",
                llm_answer="LLM",
                category="hist",
                notes="note",
            )
        ]
        out = tmp_path / "rt.json"
        save_dataset(original, out)
        loaded = load_dataset(out)
        assert loaded[0].id == "rt1"
        assert loaded[0].category == "hist"
        assert loaded[0].llm_answer == "LLM"
        assert loaded[0].notes == "note"
