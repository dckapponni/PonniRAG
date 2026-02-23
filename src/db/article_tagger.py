"""
Article Tagger for Ponni Magazine articles.
Hybrid NLP approach: rule-based patterns + TF-IDF fallback.
Assigns 1-3 tags per article from 15 predefined categories.
No LLM calls — fully offline and deterministic.
"""
import re
import logging
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# 15-category taxonomy derived from title frequency analysis of 1,667 articles
TAXONOMY = {
    "FICTION": {"tamil": "புனைவு", "english": "Fiction/Serial"},
    "EDITORIAL": {"tamil": "தலையங்கம்", "english": "Editorial/Opinion"},
    "LITERARY_REVIEW": {"tamil": "இலக்கிய விமர்சனம்", "english": "Literary Review"},
    "POETRY": {"tamil": "கவிதை", "english": "Poetry"},
    "QA_COLUMN": {"tamil": "கேள்வி பதில்", "english": "Q&A Column"},
    "ARTS_CULTURE": {"tamil": "கலை கலாச்சாரம்", "english": "Arts & Culture"},
    "SATIRE_HUMOR": {"tamil": "நகைச்சுவை", "english": "Satire/Humor"},
    "CLASSICAL_LIT": {"tamil": "செந்தமிழ் இலக்கியம்", "english": "Classical Literature"},
    "PUBLIC_FORUM": {"tamil": "பொது மேடை", "english": "Public Forum"},
    "WOMENS_ISSUES": {"tamil": "பெண்கள் நலன்", "english": "Women's Issues"},
    "RELIGIOUS_DEBATE": {"tamil": "மத விவாதம்", "english": "Religious Debate"},
    "CHILDRENS": {"tamil": "சிறுவர் பகுதி", "english": "Children's Section"},
    "POLITICAL": {"tamil": "அரசியல்", "english": "Political Commentary"},
    "BIOGRAPHY": {"tamil": "வரலாறு", "english": "Biography/History"},
    "GENERAL": {"tamil": "பொது", "english": "General/Uncategorized"},
}

# Rule-based patterns per category
# title_exact: exact title match (case-insensitive strip)
# title_contains: substring match in title
# author_heuristics: special author-based rules
RULE_PATTERNS = {
    "FICTION": {
        "title_exact": [
            "சரசா", "அவா", "பொட்டை", "துரோகி", "வேண்டாத ஆசை",
            "மண்ணின் குரல்", "காதல் கடிதம்", "அன்பின் அழுகை",
            "நிழல்", "தீ", "புயல்", "புதிய வழி",
        ],
        "title_contains": [
            "கதை", "நாவல்", "சிறுகதை", "தொடர்கதை", "நீள்கதை",
        ],
    },
    "EDITORIAL": {
        "title_exact": [
            "காலமும் கருத்தும்", "எங்கள் எண்ணம்", "ஆசிரியர் குறிப்பு",
        ],
        "title_contains": [
            "தலையங்கம்", "ஆசிரியர்",
        ],
    },
    "LITERARY_REVIEW": {
        "title_exact": [
            "வளரும் இலக்கியம்",
        ],
        "title_contains": [
            "இலக்கிய விமர்சனம்", "புத்தக விமர்சனம்", "நூல் விமர்சனம்",
            "இலக்கியம்",
        ],
    },
    "POETRY": {
        "title_exact": [
            "பாரதிதாசன் பரம்பரை",
        ],
        "title_contains": [
            "கவிதை", "பாட்டு", "பாடல்", "கவிஞர்",
        ],
    },
    "QA_COLUMN": {
        "title_exact": [
            "கண் திறக்குமா?", "உங்களுக்குத் தெரியுமா?",
        ],
        "title_contains": [
            "கேள்வி பதில்", "வினா விடை",
        ],
    },
    "ARTS_CULTURE": {
        "title_exact": [
            "கலையுலகம்",
        ],
        "title_contains": [
            "கலை", "சினிமா", "நாடகம்", "இசை", "நடனம்",
        ],
    },
    "SATIRE_HUMOR": {
        "title_exact": [
            "வம்பு மடம்", "ஆடும் மாடும்",
        ],
        "title_contains": [
            "நகைச்சுவை", "வம்பு", "சிரிப்பு", "கேலி",
        ],
    },
    "CLASSICAL_LIT": {
        "title_exact": [
            "சிலப்பதிகாரக் காட்சிகள்",
        ],
        "title_contains": [
            "சிலப்பதிகாரம்", "திருக்குறள்", "சங்க இலக்கியம்",
            "தொல்காப்பியம்", "செந்தமிழ்",
        ],
    },
    "PUBLIC_FORUM": {
        "title_exact": [
            "பொது மேடை", "நமக்குள்ளே",
        ],
        "title_contains": [
            "மேடை", "கடிதம்", "வாசகர்",
        ],
    },
    "WOMENS_ISSUES": {
        "title_exact": [
            "பெண்கள் முன்னேற்றம்", "மகளிர் அழகுக் குறிப்புகள்",
        ],
        "title_contains": [
            "பெண்கள்", "மகளிர்", "பெண்",
        ],
    },
    "RELIGIOUS_DEBATE": {
        "title_exact": [
            "மதம் அவசியமா?",
        ],
        "title_contains": [
            "மதம்", "சமயம்", "கடவுள்",
        ],
    },
    "CHILDRENS": {
        "title_exact": [
            "சிறுவர் அரங்கம்",
        ],
        "title_contains": [
            "சிறுவர்", "குழந்தை",
        ],
    },
    "POLITICAL": {
        "title_contains": [
            "அரசியல்", "தேர்தல்", "காங்கிரஸ்", "சுதந்திரம்",
            "ஆட்சி", "திராவிட",
        ],
    },
    "BIOGRAPHY": {
        "title_contains": [
            "வரலாறு", "சரித்திரம்", "வாழ்க்கை வரலாறு",
            "நினைவு", "சுயசரிதை",
        ],
    },
}

# TF-IDF category keywords for fallback classification
_CATEGORY_KEYWORDS = {
    "FICTION": "கதை நாவல் சிறுகதை தொடர்கதை கதாபாத்திரம் காதல் வாழ்க்கை சரசா அவா பொட்டை துரோகி",
    "EDITORIAL": "தலையங்கம் ஆசிரியர் கருத்து எண்ணம் காலம் சமூகம் நாடு முன்னேற்றம்",
    "LITERARY_REVIEW": "இலக்கியம் விமர்சனம் புத்தகம் நூல் படைப்பு எழுத்தாளர் இலக்கிய",
    "POETRY": "கவிதை பாட்டு பாடல் கவிஞர் பாரதிதாசன் வெண்பா செய்யுள்",
    "QA_COLUMN": "கேள்வி பதில் தெரியுமா விடை வினா",
    "ARTS_CULTURE": "கலை சினிமா நாடகம் இசை நடனம் ஓவியம் கலையுலகம்",
    "SATIRE_HUMOR": "நகைச்சுவை வம்பு சிரிப்பு கேலி நையாண்டி",
    "CLASSICAL_LIT": "சிலப்பதிகாரம் திருக்குறள் சங்க தொல்காப்பியம் செந்தமிழ் பழந்தமிழ்",
    "PUBLIC_FORUM": "மேடை கடிதம் வாசகர் பொது கருத்து",
    "WOMENS_ISSUES": "பெண்கள் மகளிர் பெண் அழகு முன்னேற்றம் சமூக நிலை",
    "RELIGIOUS_DEBATE": "மதம் சமயம் கடவுள் ஆன்மீகம் நம்பிக்கை",
    "CHILDRENS": "சிறுவர் குழந்தை பிள்ளை விளையாட்டு",
    "POLITICAL": "அரசியல் தேர்தல் காங்கிரஸ் சுதந்திரம் ஆட்சி திராவிட இயக்கம் கட்சி",
    "BIOGRAPHY": "வரலாறு சரித்திரம் வாழ்க்கை நினைவு சுயசரிதை புகழ்",
    "GENERAL": "பொன்னி இதழ் கட்டுரை பகுதி பொது",
}

# Known serial titles (appear 10+ times with named author) -> FICTION
_SERIAL_TITLES = {
    "சரசா", "அவா", "பொட்டை", "துரோகி", "வேண்டாத ஆசை",
    "மண்ணின் குரல்", "காதல் கடிதம்",
}

TFIDF_THRESHOLD = 0.15
TAMIL_TOKEN_PATTERN = r'[\u0B80-\u0BFF]+|\w+'


class ArticleTagger:
    """Hybrid article tagger: rule-based patterns + TF-IDF fallback."""

    def __init__(self):
        self.taxonomy = TAXONOMY
        self.rules = RULE_PATTERNS
        self.tfidf = None
        self.category_vectors = None
        self.category_ids = list(TAXONOMY.keys())

    def train_tfidf(self, articles: List[Dict]):
        """Train TF-IDF on full corpus. Called once during indexing."""
        if not articles:
            logger.warning("No articles to train TF-IDF on")
            return

        corpus = []
        for article in articles:
            title = article.get("title", "")
            content = article.get("content", "")[:500]
            corpus.append(f"{title} {content}")

        self.tfidf = TfidfVectorizer(
            token_pattern=TAMIL_TOKEN_PATTERN,
            max_features=5000,
            min_df=2,
            max_df=0.95,
        )
        self.tfidf.fit(corpus)

        # Compute category representative vectors from keywords
        category_texts = [_CATEGORY_KEYWORDS[cat_id] for cat_id in self.category_ids]
        self.category_vectors = self.tfidf.transform(category_texts)

        logger.info(f"TF-IDF trained on {len(corpus)} articles, vocab size={len(self.tfidf.vocabulary_)}")

    def tag_article(self, article: Dict) -> List[str]:
        """Assign 1-3 category IDs. Rules first, TF-IDF fallback."""
        tags = self._apply_rules(article)
        if tags:
            return tags[:3]

        tags = self._tfidf_classify(article)
        if tags:
            return tags[:3]

        return ["GENERAL"]

    def get_tamil_tags(self, tag_ids: List[str]) -> List[str]:
        """Map tag IDs to Tamil display names."""
        return [self.taxonomy[tid]["tamil"] for tid in tag_ids if tid in self.taxonomy]

    def _apply_rules(self, article: Dict) -> List[str]:
        """Apply rule-based classification. Returns list of matching category IDs."""
        title = (article.get("title") or "").strip()
        author = (article.get("author_name") or "").strip()
        tags = []

        # Check serial detection first
        if title in _SERIAL_TITLES and author and author.lower() != "na":
            if "FICTION" not in tags:
                tags.append("FICTION")

        for cat_id, patterns in self.rules.items():
            if cat_id in tags:
                continue

            # Exact title match
            exact_titles = patterns.get("title_exact", [])
            if title in exact_titles:
                tags.append(cat_id)
                continue

            # Title substring match
            for keyword in patterns.get("title_contains", []):
                if keyword in title:
                    tags.append(cat_id)
                    break

        return tags

    def _tfidf_classify(self, article: Dict) -> List[str]:
        """TF-IDF based classification for articles not matched by rules."""
        if self.tfidf is None or self.category_vectors is None:
            return ["GENERAL"]

        title = article.get("title", "")
        content = article.get("content", "")[:500]
        text = f"{title} {content}"

        if not text.strip():
            return ["GENERAL"]

        article_vector = self.tfidf.transform([text])
        similarities = cosine_similarity(article_vector, self.category_vectors)[0]

        # Get categories above threshold, sorted by similarity
        scored = [
            (self.category_ids[i], similarities[i])
            for i in range(len(self.category_ids))
            if similarities[i] >= TFIDF_THRESHOLD
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return ["GENERAL"]

        # Return top 1-3 tags (exclude GENERAL from TF-IDF results if others exist)
        tags = [cat_id for cat_id, _ in scored if cat_id != "GENERAL"]
        if not tags:
            return ["GENERAL"]

        return tags[:3]
