"""Tests for the ArticleTagger hybrid NLP tagger.

Tests rule-based tagging, TF-IDF fallback, determinism,
cardinality, and Tamil name mapping.
"""

import sys
from pathlib import Path

# Ensure src/db is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "db"))

from article_tagger import TAXONOMY, ArticleTagger  # noqa: E402

# --- Sample articles for testing ---

FICTION_ARTICLE = {
    "title": "சரசா",
    "author_name": "கல்கி",
    "content": "இது ஒரு நீண்ட தொடர்கதை. கதையின் கதாபாத்திரங்கள் பல." * 50,
}

EDITORIAL_ARTICLE = {
    "title": "காலமும் கருத்தும்",
    "author_name": "NA",
    "content": "இந்த வாரம் நாட்டின் நிலைமை பற்றி சிந்திக்க வேண்டும்." * 30,
}

POETRY_ARTICLE = {
    "title": "பாரதிதாசன் பரம்பரை",
    "author_name": "பாரதிதாசன்",
    "content": "கவிதை வரிகள் இங்கே தொடர்கின்றன." * 30,
}

LITERARY_REVIEW_ARTICLE = {
    "title": "வளரும் இலக்கியம்",
    "author_name": "NA",
    "content": "இந்த நூல் விமர்சனம் மிக முக்கியம்." * 30,
}

QA_ARTICLE = {
    "title": "கண் திறக்குமா?",
    "author_name": "NA",
    "content": "கேள்வி பதில் பகுதி." * 30,
}

SATIRE_ARTICLE = {
    "title": "வம்பு மடம்",
    "author_name": "NA",
    "content": "நகைச்சுவை கட்டுரை." * 30,
}

CLASSICAL_ARTICLE = {
    "title": "சிலப்பதிகாரக் காட்சிகள்",
    "author_name": "NA",
    "content": "சிலப்பதிகாரம் பற்றிய விளக்கம்." * 30,
}

FICTION_KEYWORD_ARTICLE = {
    "title": "ஒரு சிறுகதை",
    "author_name": "எழுத்தாளர்",
    "content": "கதையின் ஆரம்பம்." * 30,
}

AMBIGUOUS_ARTICLE = {
    "title": "புதிய பார்வை",
    "author_name": "அறிஞர்",
    "content": (
        "அரசியல் நிலைமை மாறிவிட்டது. "
        "தேர்தல் நெருங்குகிறது. "
        "காங்கிரஸ் கட்சி ஆட்சியில் "
        "திராவிட இயக்கம் முக்கியம்."
    )
    * 30,
}

EMPTY_ARTICLE = {
    "title": "",
    "author_name": "",
    "content": "",
}

ALL_SAMPLE_ARTICLES = [
    FICTION_ARTICLE,
    EDITORIAL_ARTICLE,
    POETRY_ARTICLE,
    LITERARY_REVIEW_ARTICLE,
    QA_ARTICLE,
    SATIRE_ARTICLE,
    CLASSICAL_ARTICLE,
    FICTION_KEYWORD_ARTICLE,
    AMBIGUOUS_ARTICLE,
    EMPTY_ARTICLE,
]


def _make_tagger():
    """Create a tagger trained on sample articles."""
    tagger = ArticleTagger()
    tagger.train_tfidf(ALL_SAMPLE_ARTICLES)
    return tagger


# === Rule-based tests ===


class TestRuleBasedTagging:
    """Test suite for rule-based article tagging."""

    def test_exact_title_fiction(self):
        """Test fiction tag for exact title match."""
        tagger = _make_tagger()
        tags = tagger.tag_article(FICTION_ARTICLE)
        assert "FICTION" in tags

    def test_exact_title_editorial(self):
        """Test editorial tag for exact title match."""
        tagger = _make_tagger()
        tags = tagger.tag_article(EDITORIAL_ARTICLE)
        assert "EDITORIAL" in tags

    def test_exact_title_poetry(self):
        """Test poetry tag for exact title match."""
        tagger = _make_tagger()
        tags = tagger.tag_article(POETRY_ARTICLE)
        assert "POETRY" in tags

    def test_exact_title_literary_review(self):
        """Test literary review tag for exact title match."""
        tagger = _make_tagger()
        tags = tagger.tag_article(LITERARY_REVIEW_ARTICLE)
        assert "LITERARY_REVIEW" in tags

    def test_exact_title_qa(self):
        """Test QA column tag for exact title match."""
        tagger = _make_tagger()
        tags = tagger.tag_article(QA_ARTICLE)
        assert "QA_COLUMN" in tags

    def test_exact_title_satire(self):
        """Test satire/humor tag for exact title match."""
        tagger = _make_tagger()
        tags = tagger.tag_article(SATIRE_ARTICLE)
        assert "SATIRE_HUMOR" in tags

    def test_exact_title_classical(self):
        """Test classical literature tag for exact title match."""
        tagger = _make_tagger()
        tags = tagger.tag_article(CLASSICAL_ARTICLE)
        assert "CLASSICAL_LIT" in tags

    def test_keyword_title_fiction(self):
        """Test fiction tag for keyword title match."""
        tagger = _make_tagger()
        tags = tagger.tag_article(FICTION_KEYWORD_ARTICLE)
        assert "FICTION" in tags

    def test_serial_detection(self):
        """Known serial title with named author should tag as FICTION."""
        tagger = _make_tagger()
        article = {"title": "அவா", "author_name": "கல்கி", "content": "text " * 50}
        tags = tagger.tag_article(article)
        assert "FICTION" in tags


# === TF-IDF fallback tests ===


class TestTfidfFallback:
    """Test suite for TF-IDF fallback tagging."""

    def test_ambiguous_gets_tagged(self):
        """Test ambiguous article gets tagged via TF-IDF."""
        tagger = _make_tagger()
        tags = tagger.tag_article(AMBIGUOUS_ARTICLE)
        assert len(tags) >= 1
        assert (
            "GENERAL" != tags[0] or len(tags) == 1
        )  # either non-GENERAL or just GENERAL

    def test_empty_article_gets_general(self):
        """Test empty article gets GENERAL tag."""
        tagger = _make_tagger()
        tags = tagger.tag_article(EMPTY_ARTICLE)
        assert "GENERAL" in tags

    def test_tfidf_without_training(self):
        """Test untrained tagger returns GENERAL."""
        tagger = ArticleTagger()  # No train_tfidf call
        tags = tagger.tag_article(AMBIGUOUS_ARTICLE)
        assert "GENERAL" in tags


# === Determinism tests ===


class TestDeterminism:
    """Test suite for tagging determinism."""

    def test_same_input_same_output(self):
        """Test same article always gets same tags."""
        tagger = _make_tagger()
        tags1 = tagger.tag_article(FICTION_ARTICLE)
        tags2 = tagger.tag_article(FICTION_ARTICLE)
        assert tags1 == tags2

    def test_deterministic_across_instances(self):
        """Test two instances produce same tags."""
        tagger1 = _make_tagger()
        tagger2 = _make_tagger()
        for article in ALL_SAMPLE_ARTICLES:
            assert tagger1.tag_article(article) == tagger2.tag_article(article)


# === Cardinality tests ===


class TestCardinality:
    """Test suite for tag cardinality constraints."""

    def test_always_1_to_3_tags(self):
        """Test each article gets 1 to 3 tags."""
        tagger = _make_tagger()
        for article in ALL_SAMPLE_ARTICLES:
            tags = tagger.tag_article(article)
            title = article.get("title")
            assert 1 <= len(tags) <= 3, f"Got {len(tags)} tags for {title}: {tags}"

    def test_tags_are_valid_ids(self):
        """Test all tags are valid taxonomy IDs."""
        tagger = _make_tagger()
        for article in ALL_SAMPLE_ARTICLES:
            tags = tagger.tag_article(article)
            for tag in tags:
                assert tag in TAXONOMY, f"Invalid tag ID: {tag}"


# === Tamil name mapping tests ===


class TestTamilNames:
    """Test suite for Tamil name mapping."""

    def test_get_tamil_tags(self):
        """Test Tamil tag names for known IDs."""
        tagger = _make_tagger()
        tamil = tagger.get_tamil_tags(["FICTION", "POETRY"])
        assert tamil == ["புனைவு", "கவிதை"]

    def test_get_tamil_tags_empty(self):
        """Test Tamil tags for empty input."""
        tagger = _make_tagger()
        tamil = tagger.get_tamil_tags([])
        assert tamil == []

    def test_get_tamil_tags_invalid_id(self):
        """Test Tamil tags for invalid IDs."""
        tagger = _make_tagger()
        tamil = tagger.get_tamil_tags(["NONEXISTENT"])
        assert tamil == []

    def test_all_taxonomy_has_tamil(self):
        """Test all taxonomy entries have Tamil and English names."""
        for cat_id, info in TAXONOMY.items():
            assert "tamil" in info, f"Missing tamil for {cat_id}"
            assert "english" in info, f"Missing english for {cat_id}"
            assert len(info["tamil"]) > 0


# === Taxonomy tests ===


class TestTaxonomy:
    """Test suite for taxonomy structure."""

    def test_has_15_categories(self):
        """Test taxonomy has exactly 15 categories."""
        assert len(TAXONOMY) == 15

    def test_general_exists(self):
        """Test GENERAL category exists in taxonomy."""
        assert "GENERAL" in TAXONOMY
