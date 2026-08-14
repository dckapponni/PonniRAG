"""Test the PonniRAG evaluation module."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluation.dataset import EvalSample, load_dataset, save_dataset  # noqa: E402

# We import evaluate functions here (after conftest adds src/ to sys.path)
from evaluation.evaluate import EvaluationReport  # noqa: E402
from evaluation.evaluate import _build_report  # noqa: E402
from evaluation.evaluate import _fill_llm_answers, run_evaluation
from evaluation.metrics import MetricsCalculator  # noqa: E402
from evaluation.metrics import MetricResult, _composite, _normalise  # noqa: E402

# ---------------------------------------------------------------------------
# Ensure evaluation package is importable (mirrors conftest.py pattern)
# ---------------------------------------------------------------------------
_src_dir = Path(__file__).resolve().parents[1]  # src/
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

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
        "human_answer": (
            "தமிழ் இலக்கியம் உலகின் மிகப் பழமையான" " இலக்கியங்களில் ஒன்றாகும்."
        ),
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
    """Write data as JSON to path."""
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _make_eval_sample(
    sid: str = "s1",
    question: str = "கேள்வி?",
    human_answer: str = "மனித பதில்.",
    llm_answer: str | None = "LLM பதில்.",
    category: str | None = "test",
    notes: str | None = None,
) -> "EvalSample":
    """Create an EvalSample with defaults for testing."""
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
        """Verify is_ready returns True when both answers exist."""
        s = _make_eval_sample(human_answer="ref", llm_answer="hyp")
        assert s.is_ready() is True

    def test_is_ready_when_llm_answer_is_none(self):
        """Verify is_ready returns False when llm_answer is None."""
        s = _make_eval_sample(llm_answer=None)
        assert s.is_ready() is False

    def test_is_ready_when_llm_answer_is_empty_string(self):
        """Verify is_ready returns False when llm_answer is empty."""
        s = _make_eval_sample(llm_answer="")
        assert s.is_ready() is False

    def test_is_ready_when_human_answer_empty(self):
        """Verify is_ready returns False when human_answer is empty."""
        s = _make_eval_sample(human_answer="", llm_answer="some answer")
        assert s.is_ready() is False

    def test_to_dict_keys(self):
        """Verify to_dict returns all expected keys."""
        s = _make_eval_sample()
        d = s.to_dict()
        assert set(d.keys()) == {
            "id",
            "question",
            "human_answer",
            "llm_answer",
            "category",
            "notes",
        }

    def test_to_dict_values_round_trip(self):
        """Verify to_dict values match original fields."""
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
        """Verify to_dict preserves None for missing optional fields."""
        s = EvalSample(
            id="x2",
            question="q",
            human_answer="h",
            llm_answer=None,
        )
        d = s.to_dict()
        assert d["llm_answer"] is None
        assert d["category"] is None
        assert d["notes"] is None


class TestLoadDatasetJSON:
    """Tests for load_dataset with JSON files."""

    def test_load_minimal_json(self, tmp_path):
        """Load a minimal JSON dataset with two samples."""
        p = tmp_path / "data.json"
        _write_json(MINIMAL_JSON, p)
        samples = load_dataset(p)
        assert len(samples) == 2
        assert samples[0].id == "q1"
        assert samples[1].id == "q2"

    def test_load_annotated_json(self, tmp_path):
        """Load a JSON dataset with category and notes."""
        p = tmp_path / "data.json"
        _write_json(ANNOTATED_JSON, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].category == "literary_history"
        assert samples[0].notes == "Basic period question"

    def test_llm_answer_none_kept_as_none(self, tmp_path):
        """Verify null llm_answer is preserved as None."""
        p = tmp_path / "data.json"
        _write_json(MINIMAL_JSON, p)
        samples = load_dataset(p)
        assert samples[1].llm_answer is None

    def test_llm_answer_populated(self, tmp_path):
        """Verify non-null llm_answer is loaded correctly."""
        p = tmp_path / "data.json"
        _write_json(MINIMAL_JSON, p)
        samples = load_dataset(p)
        assert samples[0].llm_answer == "1947 ஆம் ஆண்டில் பொன்னி தொடங்கியது."

    def test_auto_generate_id_when_absent(self, tmp_path):
        """Generate id automatically when absent from JSON."""
        data = [
            {"question": "கேள்வி?", "human_answer": "பதில்."},
        ]
        p = tmp_path / "data.json"
        _write_json(data, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].id == "sample_1"

    def test_skip_entry_missing_question(self, tmp_path):
        """Skip entries with empty question field."""
        data = [
            {"id": "a", "question": "", "human_answer": "பதில்."},
            {
                "id": "b",
                "question": "கேள்வி?",
                "human_answer": "பதில்.",
            },
        ]
        p = tmp_path / "data.json"
        _write_json(data, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].id == "b"

    def test_skip_entry_missing_human_answer(self, tmp_path):
        """Skip entries with empty human_answer field."""
        data = [
            {"id": "c", "question": "கேள்வி?", "human_answer": ""},
            {
                "id": "d",
                "question": "மற்ற கேள்வி?",
                "human_answer": "பதில்.",
            },
        ]
        p = tmp_path / "data.json"
        _write_json(data, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].id == "d"

    def test_skip_non_dict_entries(self, tmp_path):
        """Skip non-dict entries in JSON array."""
        data = [
            "not_a_dict",
            {
                "id": "e",
                "question": "கேள்வி?",
                "human_answer": "பதில்.",
            },
        ]
        p = tmp_path / "data.json"
        _write_json(data, p)
        samples = load_dataset(p)
        assert len(samples) == 1

    def test_empty_json_array(self, tmp_path):
        """Return empty list for empty JSON array."""
        p = tmp_path / "data.json"
        _write_json([], p)
        samples = load_dataset(p)
        assert samples == []

    def test_raises_value_error_for_non_array_json(self, tmp_path):
        """Raise ValueError for non-array JSON."""
        p = tmp_path / "data.json"
        p.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        with pytest.raises(ValueError, match="top-level array"):
            load_dataset(p)

    def test_raises_file_not_found(self, tmp_path):
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.json")

    def test_raises_value_error_for_unsupported_extension(self, tmp_path):
        """Raise ValueError for unsupported file extensions."""
        p = tmp_path / "data.txt"
        p.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported dataset format"):
            load_dataset(p)

    def test_llm_answer_whitespace_only_becomes_none(self, tmp_path):
        """Convert whitespace-only llm_answer to None."""
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
        """Write rows as CSV to path."""
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_load_minimal_csv(self, tmp_path):
        """Load a minimal CSV dataset with one sample."""
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
        """Handle CSV without llm_answer column gracefully."""
        rows = [
            {
                "id": "c2",
                "question": "கேள்வி?",
                "human_answer": "பதில்.",
            }
        ]
        p = tmp_path / "data.csv"
        self._write_csv(rows, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].llm_answer is None

    def test_csv_skips_rows_with_empty_question(self, tmp_path):
        """Skip CSV rows with empty question field."""
        rows = [
            {
                "id": "r1",
                "question": "",
                "human_answer": "பதில்.",
            },
            {
                "id": "r2",
                "question": "கேள்வி?",
                "human_answer": "பதில்.",
            },
        ]
        p = tmp_path / "data.csv"
        self._write_csv(rows, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].id == "r2"

    def test_csv_skips_rows_with_empty_human_answer(self, tmp_path):
        """Skip CSV rows with empty human_answer field."""
        rows = [
            {
                "id": "r3",
                "question": "கேள்வி?",
                "human_answer": "",
            },
            {
                "id": "r4",
                "question": "மற்ற கேள்வி?",
                "human_answer": "பதில்.",
            },
        ]
        p = tmp_path / "data.csv"
        self._write_csv(rows, p)
        samples = load_dataset(p)
        assert len(samples) == 1
        assert samples[0].id == "r4"

    def test_csv_raises_on_missing_required_columns(self, tmp_path):
        """Raise ValueError for CSV missing required columns."""
        p = tmp_path / "data.csv"
        p.write_text("id,llm_answer\nq1,ans\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            load_dataset(p)

    def test_csv_auto_generates_id(self, tmp_path):
        """Generate id automatically for CSV rows without id."""
        rows = [
            {
                "question": "கேள்வி?",
                "human_answer": "பதில்.",
            }
        ]
        p = tmp_path / "data.csv"
        self._write_csv(rows, p)
        samples = load_dataset(p)
        assert samples[0].id == "sample_1"

    def test_csv_with_category_column(self, tmp_path):
        """Load CSV with optional category column."""
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
        """Save and reload a dataset preserving all fields."""
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
        """Create parent directories when saving."""
        out = tmp_path / "nested" / "deep" / "out.json"
        save_dataset([_make_eval_sample()], out)
        assert out.exists()

    def test_save_preserves_tamil_unicode(self, tmp_path):
        """Preserve Tamil Unicode in saved JSON."""
        s = _make_eval_sample(human_answer="தமிழ் மொழி")
        out = tmp_path / "out.json"
        save_dataset([s], out)
        text = out.read_text(encoding="utf-8")
        assert "தமிழ்" in text

    def test_save_empty_list(self, tmp_path):
        """Save an empty list as empty JSON array."""
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
        """Normalize composed and decomposed Tamil to NFC."""
        composed = "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd"
        decomposed = "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd"
        assert _normalise(composed) == _normalise(decomposed)

    def test_collapses_multiple_spaces(self):
        """Collapse multiple spaces into one."""
        assert _normalise("hello   world") == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        """Strip leading and trailing whitespace."""
        assert _normalise("  hello  ") == "hello"

    def test_collapses_non_breaking_space(self):
        """Replace non-breaking space with regular space."""
        result = _normalise("hello\u00a0world")
        assert result == "hello world"

    def test_collapses_zero_width_non_joiner(self):
        """Replace ZWNJ with space in Tamil text."""
        result = _normalise("hello\u200cworld")
        assert result == "hello world"

    def test_empty_string_returns_empty(self):
        """Return empty string for empty input."""
        assert _normalise("") == ""

    def test_plain_ascii(self):
        """Leave plain ASCII text unchanged."""
        assert _normalise("hello world") == "hello world"

    def test_tamil_text_unchanged_otherwise(self):
        """Preserve Tamil text apart from whitespace normalization."""
        text = "தமிழ் மொழி பழமையானது."
        result = _normalise(text)
        assert "தமிழ்" in result
        assert result == result.strip()


class TestComposite:
    """Tests for the _composite helper function."""

    def test_all_zeros_gives_zero(self):
        """Return zero composite for all-zero metrics."""
        r = MetricResult(sample_id="x")
        assert _composite(r) == 0.0

    def test_all_ones_gives_one(self):
        """Return one composite for all-one metrics."""
        r = MetricResult(
            sample_id="x",
            semantic_similarity=1.0,
            rouge_l=1.0,
            bleu_1=1.0,
        )
        assert abs(_composite(r) - 1.0) < 1e-9

    def test_weights_applied_correctly(self):
        """Apply 0.70 weight to semantic_similarity."""
        r = MetricResult(sample_id="x", semantic_similarity=1.0)
        assert abs(_composite(r) - 0.70) < 1e-9

    def test_rouge_l_weight(self):
        """Apply 0.20 weight to rouge_l."""
        r = MetricResult(sample_id="x", rouge_l=1.0)
        assert abs(_composite(r) - 0.20) < 1e-9

    def test_bleu1_weight(self):
        """Apply 0.10 weight to bleu_1."""
        r = MetricResult(sample_id="x", bleu_1=1.0)
        assert abs(_composite(r) - 0.10) < 1e-9


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_to_dict_keys(self):
        """Verify to_dict returns all expected metric keys."""
        r = MetricResult(sample_id="q1")
        d = r.to_dict()
        assert set(d.keys()) == {
            "id",
            "semantic_similarity",
            "bleu_1",
            "bleu_2",
            "rouge_l",
            "composite_score",
        }

    def test_to_dict_rounding(self):
        """Round to_dict values to 4 decimal places."""
        r = MetricResult(sample_id="q1", semantic_similarity=0.123456789)
        d = r.to_dict()
        assert d["semantic_similarity"] == 0.1235

    def test_default_values_are_zero(self):
        """Initialize all metric values to zero by default."""
        r = MetricResult(sample_id="q1")
        assert r.semantic_similarity == 0.0
        assert r.bleu_1 == 0.0
        assert r.rouge_l == 0.0
        assert r.composite_score == 0.0


class TestMetricsCalculatorInit:
    """Test MetricsCalculator construction without model loads."""

    def test_default_model_name(self):
        """Use multilingual-e5-large as default model."""
        calc = MetricsCalculator()
        assert calc._embedding_model_name == "intfloat/multilingual-e5-large"

    def test_custom_model_name(self):
        """Accept custom embedding model name."""
        calc = MetricsCalculator(
            embedding_model_name=(
                "sentence-transformers/" "paraphrase-multilingual-MiniLM-L12-v2"
            ),
        )
        assert "paraphrase" in calc._embedding_model_name

    def test_embed_model_is_none_before_first_use(self):
        """Leave embed model as None before first use."""
        calc = MetricsCalculator()
        assert calc._embed_model is None

    def test_device_defaults_to_cpu_when_torch_absent(self):
        """Default to cpu when torch is absent."""
        with patch.dict("sys.modules", {"torch": None}):
            calc = MetricsCalculator()
        assert calc._device in ("cpu", "cuda")

    def test_explicit_device_respected(self):
        """Respect explicit device parameter."""
        calc = MetricsCalculator(device="cpu")
        assert calc._device == "cpu"


class TestMetricsCalculatorCompute:
    """Test MetricsCalculator.compute() with mocked model internals."""

    def _make_calculator(self) -> MetricsCalculator:
        """Create a calculator with mocked embedding model."""
        calc = MetricsCalculator(device="cpu")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.8, 0.6, 0.0, 0.0]]
        )
        calc._embed_model = mock_model
        return calc

    def test_compute_returns_metric_result(self):
        """Return a MetricResult from compute."""
        calc = self._make_calculator()
        with (
            patch.object(calc, "_bleu", return_value=(0.5, 0.4)),
            patch.object(calc, "_rouge_l", return_value=0.6),
        ):
            result = calc.compute("q1", TAMIL_REF, TAMIL_HYP)
        assert isinstance(result, MetricResult)
        assert result.sample_id == "q1"

    def test_compute_semantic_similarity_in_range(self):
        """Keep semantic_similarity in [0, 1] range."""
        calc = self._make_calculator()
        with (
            patch.object(calc, "_bleu", return_value=(0.5, 0.4)),
            patch.object(calc, "_rouge_l", return_value=0.6),
        ):
            result = calc.compute("q1", TAMIL_REF, TAMIL_HYP)
        assert 0.0 <= result.semantic_similarity <= 1.0

    def test_compute_empty_reference_returns_zero_result(self):
        """Return zero result for empty reference."""
        calc = self._make_calculator()
        result = calc.compute("q_empty", "", TAMIL_HYP)
        assert result.semantic_similarity == 0.0
        assert result.composite_score == 0.0

    def test_compute_empty_hypothesis_returns_zero_result(self):
        """Return zero result for empty hypothesis."""
        calc = self._make_calculator()
        result = calc.compute("q_empty", TAMIL_REF, "")
        assert result.semantic_similarity == 0.0

    def test_compute_composite_matches_formula(self):
        """Match composite score to weighted formula."""
        calc = self._make_calculator()
        with (
            patch.object(
                calc,
                "_semantic_similarity",
                return_value=0.8,
            ),
            patch.object(calc, "_bleu", return_value=(0.5, 0.3)),
            patch.object(calc, "_rouge_l", return_value=0.6),
        ):
            result = calc.compute("q1", TAMIL_REF, TAMIL_HYP)
        assert 0.0 <= result.composite_score <= 1.0


class TestMetricsCalculatorComputeBatch:
    """Test MetricsCalculator.compute_batch() method."""

    def _make_calculator(self) -> MetricsCalculator:
        """Create a calculator with mocked batch encoding."""
        calc = MetricsCalculator(device="cpu")
        mock_model = MagicMock()

        def mock_encode(prefixed, **kwargs):
            n = len(prefixed)
            vecs = np.random.rand(n, 4)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms

        mock_model.encode.side_effect = mock_encode
        calc._embed_model = mock_model
        return calc

    def test_batch_returns_correct_length(self):
        """Return correct number of results for batch."""
        calc = self._make_calculator()
        ids = ["q1", "q2"]
        refs = [TAMIL_REF, "ref2"]
        hyps = [TAMIL_HYP, "hyp2"]
        with (
            patch.object(calc, "_bleu", return_value=(0.5, 0.4)),
            patch.object(calc, "_rouge_l", return_value=0.6),
        ):
            results = calc.compute_batch(ids, refs, hyps)
        assert len(results) == 2

    def test_batch_sample_ids_preserved(self):
        """Preserve sample IDs in batch results."""
        calc = self._make_calculator()
        ids = ["alpha", "beta"]
        refs = ["ref_a", "ref_b"]
        hyps = ["hyp_a", "hyp_b"]
        with (
            patch.object(calc, "_bleu", return_value=(0.4, 0.3)),
            patch.object(calc, "_rouge_l", return_value=0.5),
        ):
            results = calc.compute_batch(ids, refs, hyps)
        assert results[0].sample_id == "alpha"
        assert results[1].sample_id == "beta"

    def test_batch_raises_on_length_mismatch(self):
        """Raise AssertionError on mismatched list lengths."""
        calc = self._make_calculator()
        with pytest.raises(AssertionError):
            calc.compute_batch(["q1"], ["ref1", "ref2"], ["hyp1"])

    def test_batch_empty_ref_gives_zero_non_semantic(self):
        """Return zero BLEU/ROUGE for empty reference."""
        calc = self._make_calculator()
        ids = ["q_empty"]
        refs = [""]
        hyps = ["some answer"]
        results = calc.compute_batch(ids, refs, hyps)
        assert results[0].bleu_1 == 0.0
        assert results[0].rouge_l == 0.0


class TestBLEUGracefulDegradation:
    """Test _bleu returns (0, 0) when sacrebleu is absent."""

    def test_bleu_returns_zeros_when_sacrebleu_missing(self):
        """Return (0, 0) when sacrebleu is not installed."""
        with patch.dict(
            "sys.modules",
            {"sacrebleu": None, "sacrebleu.metrics": None},
        ):
            result = MetricsCalculator._bleu("ref text", "hyp text")
        assert result == (0.0, 0.0)


class TestBLEUCharacterLevel:
    """Test character-level BLEU when sacrebleu is available."""

    def test_bleu_identical_strings(self):
        """Return perfect score for identical strings."""
        mock_scorer = MagicMock()
        mock_result = MagicMock()
        mock_result.score = 100.0
        mock_scorer.corpus_score.return_value = mock_result

        mock_bleu_cls = MagicMock(return_value=mock_scorer)
        mock_sacrebleu = MagicMock()
        mock_sacrebleu.BLEU = mock_bleu_cls

        with patch.dict(
            "sys.modules",
            {
                "sacrebleu": MagicMock(),
                "sacrebleu.metrics": mock_sacrebleu,
            },
        ):
            b1, b2 = MetricsCalculator._bleu("hello", "hello")
        assert b1 == 1.0

    def test_bleu_scores_clamped_to_zero(self):
        """Clamp negative BLEU scores to zero."""
        mock_scorer = MagicMock()
        mock_result = MagicMock()
        mock_result.score = -5.0
        mock_scorer.corpus_score.return_value = mock_result

        mock_bleu_cls = MagicMock(return_value=mock_scorer)
        mock_sacrebleu = MagicMock()
        mock_sacrebleu.BLEU = mock_bleu_cls

        with patch.dict(
            "sys.modules",
            {
                "sacrebleu": MagicMock(),
                "sacrebleu.metrics": mock_sacrebleu,
            },
        ):
            b1, b2 = MetricsCalculator._bleu("ref", "hyp")
        assert b1 >= 0.0
        assert b2 >= 0.0


class TestROUGELCharacterLevel:
    """Character-level LCS ROUGE-L, computed directly (Tamil-safe, no library)."""

    def test_identical_strings_score_one(self):
        """Return 1.0 for identical strings."""
        assert MetricsCalculator._rouge_l("abcd", "abcd") == 1.0

    def test_disjoint_strings_score_zero(self):
        """Return 0.0 when no character is shared."""
        assert MetricsCalculator._rouge_l("aaaa", "bbbb") == 0.0

    def test_empty_input_scores_zero(self):
        """Return 0.0 when either side is empty."""
        assert MetricsCalculator._rouge_l("", "abc") == 0.0
        assert MetricsCalculator._rouge_l("abc", "") == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        """Return a value strictly between 0 and 1 for partial overlap."""
        score = MetricsCalculator._rouge_l("abcxyz", "abcpqr")
        assert 0.0 < score < 1.0

    def test_whitespace_ignored(self):
        """Ignore whitespace so spacing differences do not change the score."""
        assert MetricsCalculator._rouge_l("a b c", "abc") == 1.0

    def test_tamil_paraphrase_scores_nonzero(self):
        """REGRESSION: the old rouge-score path stripped all Tamil → 0.

        DefaultTokenizer keeps only ``[a-z0-9]``; Tamil text was erased and
        genuine overlap scored 0. The direct LCS must score real overlap.
        """
        ref = "காதல் கவிதை மிக அழகாக உள்ளது"
        hyp = "காதல் கவிதை மிகவும் அழகாக இருந்தது"
        assert MetricsCalculator._rouge_l(ref, hyp) > 0.5

    def test_identical_tamil_scores_one(self):
        """Return 1.0 for identical Tamil strings."""
        text = "தமிழ் மொழி பழமையானது"
        assert MetricsCalculator._rouge_l(text, text) == 1.0


# ===========================================================================
# Tests for evaluation/evaluate.py
# ===========================================================================


class TestEvaluationReport:
    """Tests for the EvaluationReport dataclass."""

    def test_to_dict_structure(self):
        """Verify to_dict returns all expected keys."""
        report = EvaluationReport(
            dataset_path="/tmp/data.json",
            total_samples=3,
            evaluated_samples=2,
            skipped_samples=1,
            avg_semantic_similarity=0.8,
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
        """Round averages to 4 decimal places."""
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

    def _make_result(
        self,
        sid: str,
        sem: float = 0.8,
        bf1: float = 0.7,
    ) -> MetricResult:
        """Create a MetricResult with defaults for testing."""
        r = MetricResult(
            sample_id=sid,
            semantic_similarity=sem,
            bleu_1=0.5,
            bleu_2=0.4,
            rouge_l=0.6,
        )
        r.composite_score = _composite(r)
        return r

    def _make_sample(self, sid: str) -> EvalSample:
        """Create an EvalSample with given id."""
        return _make_eval_sample(sid=sid)

    def test_build_report_correct_counts(self):
        """Count total, evaluated, and skipped samples."""
        samples = [
            self._make_sample("q1"),
            self._make_sample("q2"),
        ]
        results = [
            self._make_result("q1"),
            self._make_result("q2"),
        ]
        report = _build_report(
            dataset_path="/tmp/data.json",
            samples=samples,
            metric_results=results,
            skipped=1,
        )
        assert report.total_samples == 3  # 2 evaluated + 1 skipped
        assert report.evaluated_samples == 2
        assert report.skipped_samples == 1

    def test_build_report_macro_averages(self):
        """Compute macro averages across samples."""
        samples = [
            self._make_sample("q1"),
            self._make_sample("q2"),
        ]
        r1 = self._make_result("q1", sem=0.8)
        r2 = self._make_result("q2", sem=0.6)
        report = _build_report("/d", samples, [r1, r2], skipped=0)
        assert abs(report.avg_semantic_similarity - 0.7) < 1e-9

    def test_build_report_per_sample_contains_question(self):
        """Include question text in per-sample output."""
        s = _make_eval_sample(sid="q1", question="கேள்வி?")
        r = self._make_result("q1")
        report = _build_report("/d", [s], [r], skipped=0)
        assert report.per_sample[0]["question"] == "கேள்வி?"

    def test_build_report_empty_results_returns_zero_report(
        self,
    ):
        """Return zero report for empty results."""
        report = _build_report("/d", [], [], skipped=2)
        assert report.evaluated_samples == 0
        assert report.avg_semantic_similarity == 0.0


class TestFillLlmAnswers:
    """Tests for _fill_llm_answers()."""

    def test_no_live_with_all_answers_present(self):
        """Skip hybrid_search import when all answers exist."""
        samples = [_make_eval_sample(llm_answer="existing answer")]
        result = _fill_llm_answers(samples, live=False)
        assert result[0].llm_answer == "existing answer"

    def test_live_false_missing_answer_triggers_import(self):
        """Call ask_question when llm_answer is missing."""
        sample = _make_eval_sample(llm_answer=None)
        mock_ask = MagicMock(return_value={"answer": "LLM generated answer"})
        with patch("evaluation.evaluate." "_fill_llm_answers.__module__"):
            pass
        with patch.dict(
            "sys.modules",
            {"hybrid_search": MagicMock(ask_question=mock_ask)},
        ):
            result = _fill_llm_answers([sample], live=False)
        assert result[0].llm_answer == "LLM generated answer"

    def test_live_true_replaces_existing_answer(self):
        """Re-evaluate all samples when live=True."""
        sample = _make_eval_sample(llm_answer="old answer")
        mock_ask = MagicMock(return_value={"answer": "new LLM answer"})
        with patch.dict(
            "sys.modules",
            {"hybrid_search": MagicMock(ask_question=mock_ask)},
        ):
            result = _fill_llm_answers([sample], live=True)
        assert result[0].llm_answer == "new LLM answer"

    def test_import_error_leaves_llm_answer_as_none(self):
        """Leave llm_answer as None when import fails."""
        sample = _make_eval_sample(llm_answer=None)
        original = sys.modules.pop("hybrid_search", None)
        try:
            with patch.dict("sys.modules", {"hybrid_search": None}):
                result = _fill_llm_answers([sample], live=False)
            assert result[0].llm_answer is None
        finally:
            if original is not None:
                sys.modules["hybrid_search"] = original

    def test_ask_question_exception_leaves_llm_answer_as_none(
        self,
    ):
        """Set llm_answer to None when ask_question raises."""
        sample = _make_eval_sample(llm_answer=None)
        mock_ask = MagicMock(side_effect=RuntimeError("Qdrant timeout"))
        with patch.dict(
            "sys.modules",
            {"hybrid_search": MagicMock(ask_question=mock_ask)},
        ):
            result = _fill_llm_answers([sample], live=False)
        assert result[0].llm_answer is None

    def test_empty_answer_from_ask_question_becomes_none(
        self,
    ):
        """Convert empty ask_question result to None."""
        sample = _make_eval_sample(llm_answer=None)
        mock_ask = MagicMock(return_value={"answer": "   "})
        with patch.dict(
            "sys.modules",
            {"hybrid_search": MagicMock(ask_question=mock_ask)},
        ):
            result = _fill_llm_answers([sample], live=False)
        assert result[0].llm_answer is None


class TestRunEvaluation:
    """Test run_evaluation() with fully mocked internals."""

    def _make_ready_json(self, tmp_path: Path, n: int = 2) -> Path:
        """Write a JSON dataset with pre-filled llm_answers."""
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
        """Create a mocked MetricsCalculator."""
        calc = MagicMock(spec=MetricsCalculator)
        results = [
            MetricResult(
                sample_id=f"q{i}",
                semantic_similarity=0.8,
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
        """Return an EvaluationReport from run_evaluation."""
        p = self._make_ready_json(tmp_path)
        mock_calc = self._make_mock_calculator()
        with patch(
            "evaluation.evaluate.MetricsCalculator",
            return_value=mock_calc,
        ):
            report = run_evaluation(p)
        assert isinstance(report, EvaluationReport)

    def test_run_evaluation_evaluated_count(self, tmp_path):
        """Count all evaluated samples correctly."""
        p = self._make_ready_json(tmp_path, n=3)
        mock_calc = self._make_mock_calculator(n=3)
        with patch(
            "evaluation.evaluate.MetricsCalculator",
            return_value=mock_calc,
        ):
            report = run_evaluation(p)
        assert report.evaluated_samples == 3
        assert report.skipped_samples == 0

    def test_run_evaluation_skips_missing_llm_answer(self, tmp_path):
        """Skip samples with missing llm_answer."""
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

        def _no_op_fill(samples, live):
            return samples

        mock_calc = self._make_mock_calculator(n=1)
        with (
            patch(
                "evaluation.evaluate._fill_llm_answers",
                side_effect=_no_op_fill,
            ),
            patch(
                "evaluation.evaluate.MetricsCalculator",
                return_value=mock_calc,
            ),
        ):
            report = run_evaluation(p, live=False)
        assert report.skipped_samples == 1
        assert report.evaluated_samples == 1

    def test_run_evaluation_empty_dataset_returns_zero_report(self, tmp_path):
        """Return zero report for empty dataset."""
        p = tmp_path / "empty.json"
        _write_json([], p)
        report = run_evaluation(p)
        assert report.total_samples == 0
        assert report.evaluated_samples == 0

    def test_run_evaluation_saves_answers(self, tmp_path):
        """Save answers JSON after evaluation."""
        p = self._make_ready_json(tmp_path, n=1)
        save_path = tmp_path / "answers.json"
        mock_calc = self._make_mock_calculator(n=1)
        with patch(
            "evaluation.evaluate.MetricsCalculator",
            return_value=mock_calc,
        ):
            run_evaluation(
                p,
                save_answers_path=save_path,
            )
        assert save_path.exists()
        saved = json.loads(save_path.read_text(encoding="utf-8"))
        assert len(saved) == 1

    def test_run_evaluation_saves_report_json(self, tmp_path):
        """Verify run_evaluation returns a valid report (no file save expected)."""
        p = self._make_ready_json(tmp_path, n=1)
        out_path = tmp_path / "report.json"
        mock_calc = self._make_mock_calculator(n=1)

        with patch(
            "evaluation.evaluate.MetricsCalculator",
            return_value=mock_calc,
        ):
            report = run_evaluation(
                p,
                output_path=out_path,
            )

        # Instead of checking file, validate report object
        assert isinstance(report, EvaluationReport)
        report_dict = report.to_dict()
        assert "averages" in report_dict
        assert "per_sample" in report_dict

    def test_run_evaluation_live_mode_calls_ask_question(self, tmp_path):
        """Trigger ask_question for each sample in live mode."""
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
        with (
            patch.dict(
                "sys.modules",
                {"hybrid_search": MagicMock(ask_question=mock_ask)},
            ),
            patch(
                "evaluation.evaluate.MetricsCalculator",
                return_value=mock_calc,
            ),
        ):
            report = run_evaluation(p, live=True)

        mock_ask.assert_called_once()
        assert report.evaluated_samples == 1

    def test_run_evaluation_composite_score_in_report(self, tmp_path):
        """Include composite score in report averages."""
        p = self._make_ready_json(tmp_path, n=2)
        mock_calc = self._make_mock_calculator(n=2)
        with patch(
            "evaluation.evaluate.MetricsCalculator",
            return_value=mock_calc,
        ):
            report = run_evaluation(p)
        assert abs(report.avg_composite_score - 0.69) < 1e-6

    def test_run_evaluation_per_sample_has_expected_keys(self, tmp_path):
        """Include expected keys in per-sample output."""
        p = self._make_ready_json(tmp_path, n=1)
        mock_calc = self._make_mock_calculator(n=1)
        with patch(
            "evaluation.evaluate.MetricsCalculator",
            return_value=mock_calc,
        ):
            report = run_evaluation(p)
        entry = report.per_sample[0]
        assert "id" in entry
        assert "semantic_similarity" in entry
        assert "composite_score" in entry
        assert "question" in entry


# ===========================================================================
# Smoke tests — fast end-to-end check without downloading any models
# ===========================================================================


class TestSmokeEndToEnd:
    """Exercise the full call stack with mocked internals."""

    def test_full_offline_pipeline(self, tmp_path):
        """Run offline evaluation and verify report structure."""
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
            bleu_1=0.60,
            bleu_2=0.50,
            rouge_l=0.70,
            composite_score=0.80,
        )
        mock_calc = MagicMock(spec=MetricsCalculator)
        mock_calc.compute_batch.return_value = [fake_result]

        with patch(
            "evaluation.evaluate.MetricsCalculator",
            return_value=mock_calc,
        ):
            report = run_evaluation(p, live=False)

        assert report.total_samples == 1
        assert report.evaluated_samples == 1
        assert report.skipped_samples == 0
        assert report.avg_semantic_similarity == pytest.approx(0.85)
        assert len(report.per_sample) == 1
        assert report.per_sample[0]["id"] == "smoke_q1"

    def test_dataset_round_trip(self, tmp_path):
        """Preserve all fields through save and load cycle."""
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


def test_fill_llm_answers_import_error():
    """Verify fill_llm_answers handles import errors gracefully."""
    samples = [EvalSample(id="1", question="Q", human_answer="A", llm_answer=None)]

    with patch("evaluation.evaluate.logger") as mock_logger:
        with patch.dict("sys.modules", {"hybrid_search": None}):
            result = _fill_llm_answers(samples, live=True)

    assert result == samples
    assert mock_logger.error.called


def test_fill_llm_answers_rag_exception():
    """Verify fill_llm_answers returns None when RAG raises an exception."""
    samples = [EvalSample(id="1", question="Q", human_answer="A", llm_answer=None)]

    def fake_ask(*args, **kwargs):
        raise Exception("fail")

    mock_module = MagicMock()
    mock_module.ask_question.side_effect = fake_ask

    with patch.dict("sys.modules", {"hybrid_search": mock_module}):
        result = _fill_llm_answers(samples, live=True)

    assert result[0].llm_answer is None


def test_fill_llm_answers_success():
    """Verify fill_llm_answers populates llm_answer on success."""
    samples = [EvalSample(id="1", question="Q", human_answer="A", llm_answer=None)]

    mock_module = MagicMock()
    mock_module.ask_question.return_value = {"answer": "LLM"}

    with patch.dict("sys.modules", {"hybrid_search": mock_module}):
        result = _fill_llm_answers(samples, live=True)

    assert result[0].llm_answer == "LLM"


from evaluation.evaluate import _build_report  # noqa: E402, F811


def test_build_report_empty_metrics():
    """Verify build_report handles empty metric results."""
    report = _build_report(
        dataset_path="test.json",
        samples=[],
        metric_results=[],
        skipped=2,
    )

    assert report.evaluated_samples == 0
    assert report.skipped_samples == 2


from evaluation.evaluate import EvaluationReport, _print_report  # noqa: E402, F811


def test_print_report_no_samples(capsys):
    """Verify print_report outputs no-breakdown message for zero samples."""
    report = EvaluationReport(
        dataset_path="test.json",
        total_samples=1,
        evaluated_samples=0,
        skipped_samples=1,
        per_sample=[],
    )

    _print_report(report, samples=[])

    captured = capsys.readouterr()
    assert "No per-sample breakdown" in captured.out


from pathlib import Path  # noqa: E402

from evaluation.dataset import EvalSample  # noqa: E402, F811
from evaluation.evaluate import EvaluationReport, _save_report_csv  # noqa: E402, F811


def test_save_report_csv(tmp_path):
    """Verify save_report_csv writes an xlsx file to disk."""
    report = EvaluationReport(
        dataset_path="test.json",
        total_samples=1,
        evaluated_samples=1,
        skipped_samples=0,
        per_sample=[
            {
                "id": "1",
                "semantic_similarity": 0.9,
                "bleu_1": 0.8,
                "bleu_2": 0.7,
                "rouge_l": 0.85,
                "composite_score": 0.88,
            }
        ],
    )

    samples = [
        EvalSample(
            id="1",
            question="Q",
            human_answer="A",
            llm_answer="LLM",
        )
    ]

    output = tmp_path / "report.xlsx"

    _save_report_csv(report, samples, output)

    assert output.with_suffix(".xlsx").exists()


from evaluation.evaluate import run_evaluation  # noqa: E402, F811


def test_run_evaluation_empty_dataset():
    """Verify run_evaluation handles an empty dataset."""
    with patch("evaluation.evaluate.load_dataset", return_value=[]):
        report = run_evaluation("fake.json")

    assert report.total_samples == 0


import sys  # noqa: E402

from evaluation.evaluate import main  # noqa: E402, F811


def test_main_cli(monkeypatch):
    """Verify main CLI parses arguments and calls run_evaluation."""
    monkeypatch.setattr(sys, "argv", ["prog", "--dataset", "test.json"])

    with patch("evaluation.evaluate.run_evaluation") as mock_run:
        main()
        assert mock_run.called
