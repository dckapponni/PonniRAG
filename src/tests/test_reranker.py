"""Unit tests for the cross-encoder reranker module."""

import importlib
import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _reset_reranker(monkeypatch):
    """Reload reranker module for each test so env vars and globals reset."""
    monkeypatch.setenv("ENABLE_RERANKER", "1")
    if "reranker" in sys.modules:
        importlib.reload(sys.modules["reranker"])
    yield
    if "reranker" in sys.modules:
        importlib.reload(sys.modules["reranker"])


def _install_fake_model(scores):
    """Install a fake cross-encoder that returns predetermined scores."""
    import reranker

    class FakeModel:
        def predict(self, pairs, **kwargs):
            assert len(pairs) == len(scores)
            return list(scores)

    reranker._model = FakeModel()
    reranker._load_failed = False
    return reranker


def test_empty_sources_returns_empty():
    """Empty input returns empty output with success=True (no fallback needed)."""
    import reranker

    out, ok = reranker.rerank_sources("question", [])
    assert out == []
    assert ok is True


def test_disabled_returns_fallback_signal(monkeypatch):
    """When ENABLE_RERANKER=0, return ok=False so caller falls back."""
    monkeypatch.setenv("ENABLE_RERANKER", "0")
    import reranker

    importlib.reload(reranker)
    out, ok = reranker.rerank_sources("q", [{"heading": "h", "content": "c"}])
    assert out == []
    assert ok is False


def test_reorders_by_score():
    """Sources are sorted descending by cross-encoder score."""
    sources = [
        {"heading": "a", "content": "low relevance"},
        {"heading": "b", "content": "high relevance"},
        {"heading": "c", "content": "medium relevance"},
    ]
    rer = _install_fake_model([0.1, 0.9, 0.5])
    out, ok = rer.rerank_sources("query", sources, top_in=10, top_out=3)
    assert ok is True
    assert [s["heading"] for s in out] == ["b", "c", "a"]
    assert out[0]["rerank_score"] == pytest.approx(0.9)


def test_top_out_truncates():
    """top_out limits the number of returned sources."""
    sources = [{"heading": str(i), "content": "x"} for i in range(5)]
    rer = _install_fake_model([0.1, 0.2, 0.3, 0.4, 0.5])
    out, ok = rer.rerank_sources("q", sources, top_in=5, top_out=2)
    assert ok is True
    assert len(out) == 2
    assert [s["heading"] for s in out] == ["4", "3"]


def test_top_in_caps_candidates():
    """top_in slices the candidate pool before scoring."""
    sources = [{"heading": str(i), "content": "x"} for i in range(10)]
    # Fake will assert len(pairs) == len(scores); supplying only 3 scores
    # forces top_in to slice candidates first.
    rer = _install_fake_model([0.3, 0.2, 0.1])
    out, ok = rer.rerank_sources("q", sources, top_in=3, top_out=3)
    assert ok is True
    # Only first 3 source headings ('0','1','2') were considered
    assert {s["heading"] for s in out} == {"0", "1", "2"}


def test_predict_failure_returns_fallback():
    """Exception in model.predict triggers ok=False fallback signal."""
    import reranker

    class ExplodingModel:
        def predict(self, *_a, **_kw):
            raise RuntimeError("boom")

    reranker._model = ExplodingModel()
    reranker._load_failed = False
    out, ok = reranker.rerank_sources("q", [{"heading": "h", "content": "c"}])
    assert out == []
    assert ok is False


def test_load_failure_short_circuits(monkeypatch):
    """After a prior load failure, calls short-circuit without retrying."""
    import reranker

    reranker._model = None
    reranker._load_failed = True
    out, ok = reranker.rerank_sources("q", [{"heading": "h", "content": "c"}])
    assert out == []
    assert ok is False


def test_doc_text_truncates(monkeypatch):
    """RERANKER_MAX_CHARS caps the per-doc text fed to the cross-encoder."""
    monkeypatch.setenv("RERANKER_MAX_CHARS", "20")
    import reranker

    importlib.reload(reranker)
    src = {"heading": "H", "content": "x" * 500}
    text = reranker._doc_text(src)
    assert len(text) <= 20


@pytest.mark.integration
def test_real_model_load_smoke():
    """Smoke test — loads the real model. Slow; skipped unless integration enabled."""
    if os.environ.get("RUN_RERANKER_SMOKE") != "1":
        pytest.skip("set RUN_RERANKER_SMOKE=1 to enable real model load")

    import reranker

    importlib.reload(reranker)
    model = reranker.get_reranker()
    assert model is not None
    out, ok = reranker.rerank_sources(
        "May Day article",
        [
            {"heading": "May Day", "content": "About workers and labour day"},
            {"heading": "Cooking", "content": "Recipe for dosa"},
        ],
    )
    assert ok is True
    assert out[0]["heading"] == "May Day"
