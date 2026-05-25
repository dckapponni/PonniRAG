"""CSV query pipeline for author and topic lookups in the Ponni magazine archive.

Handles author name matching, query-type detection, entity extraction,
and result formatting. Uses a two-stage lookup: exact/fuzzy CSV matching
first, multi-stage spelling correction second. Falls back to vector
search when CSV queries fail or are out of scope.

Usage::

    handled, response = handle_author_query(question, csv_path)
    if not handled:
        # fall through to vector search
"""

import logging
import re
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tamil_text import (
    _RE_INITIALS_NAME,
    _RE_TAMIL_WORD,
    FUZZY_THRESHOLD,
    _edit_distance_one,
    _PatternBank,
    _strip_tamil_possessive_suffix_word,
    _strip_tamil_possessive_suffixes,
    fuzzy_match_score,
    normalize_unicode,
)

logger = logging.getLogger(__name__)

_author_system_cache = {}
_author_system_lock = threading.Lock()


# CSV LOADER HELPER


def load_csv(csv_path) -> pd.DataFrame:
    """Public alias for _load_csv_safe (backward compatibility)."""
    return _load_csv_safe(csv_path)


def _load_csv_safe(csv_path) -> pd.DataFrame:
    """Load a CSV file, handling multi-author bracket fields correctly.

    Tries csv_fuzzy_matcher.load_csv first (wraps bracket fields like
    [A, B, C] in quotes before pandas reads them, preventing silent row
    drops). Falls back through four pd.read_csv strategies on failure.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Loaded DataFrame, or an empty DataFrame if all strategies fail.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame()

    # Primary: load_csv handles multi-author bracket fields
    try:
        from csv_fuzzy_matcher import load_csv

        df = load_csv(csv_path)
        df.columns = df.columns.str.strip()
        logger.info(f"CSV loaded via load_csv: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"load_csv failed, trying fallbacks: {str(e)[:60]}")

    # Fallbacks
    for kwargs in [
        {"encoding": "utf-8", "skipinitialspace": True},
        {"encoding": "utf-8", "on_bad_lines": "skip", "skipinitialspace": True},
        {"encoding": "utf-8", "engine": "python", "on_bad_lines": "skip"},
        {"encoding": "utf-8", "engine": "python", "quoting": 3},
    ]:
        try:
            df = pd.read_csv(csv_path, **kwargs)
            df.columns = df.columns.str.strip()
            logger.info(f"CSV loaded via fallback pd.read_csv: {len(df)} rows")
            return df
        except Exception:
            continue

    logger.error(f"All CSV loading strategies failed for {csv_path}")
    return pd.DataFrame()


# AUTHOR FIELD PARSING
def _format_author_display(csv_name: str) -> str:
    """Format a raw CSV author cell as a comma-separated display string.

    Args:
        csv_name: Raw author cell value (may contain brackets or be empty).

    Returns:
        Comma-separated author names, or ``"NA"`` if the cell is empty.
    """
    authors = _parse_csv_authors(csv_name)
    return ", ".join(authors) if authors else "NA"


def _parse_csv_authors(csv_name: str) -> List[str]:
    """Parse a CSV author cell into a list of individual author names.

    Handles bracket-wrapped multi-author fields (e.g. ``[A, B, C]``),
    single names, and empty/sentinel values (NA, nan, None, []).

    Args:
        csv_name: Raw author cell string.

    Returns:
        List of stripped author name strings, or ``[]`` if empty.
    """
    if not csv_name:
        return []

    csv_name = csv_name.strip()

    if csv_name in {"[]", "", "NA", "nan", "None"}:
        return []

    if csv_name.startswith("[") and csv_name.endswith("]"):
        csv_name = csv_name[1:-1].strip()

    if not csv_name:
        return []

    authors = [a.strip() for a in csv_name.split(",") if a.strip()]
    authors = [a for a in authors if a.lower() not in {"na", "nan", "none"}]
    return authors


# AUTHOR NAME NORMALISATION & MATCHING
def normalize_author_name(name: str) -> str:
    """Normalise a Tamil author name for fuzzy comparison.

    Strips common honorific prefixes (டாக்டர், திரு, Dr., etc.),
    removes spaces and dots, and normalises Tamil long-vowel variants
    (ஆ→அ, ஈ→இ, etc.) so minor spelling differences do not prevent matching.

    Args:
        name: Raw author name string.

    Returns:
        Normalised string suitable for comparison (not for display).
    """
    if not name:
        return ""

    name = normalize_unicode(name)

    # Remove prefixes
    prefixes_to_remove = [
        r"மு\.,?\s*",
        r"டாக்டர்\.?\s*",
        r"திரு\.,?\s*",
        r"திருமதி\.?\s*",
        r"Dr\.?\s*",
        r"Mr\.?\s*",
        r"Mrs\.?\s*",
        r"கவிஞர்\.?\s*",
        r"பேராசிரியர்\.?\s*",
        r"அறிஞர்\.?\s*",
        r"புலவர்\.?\s*",
        r"கவியரசு\.?\s*",
        r"பாவேந்தர்\.?\s*",
    ]

    cleaned = name.strip()
    for prefix in prefixes_to_remove:
        cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)

    # Strip spaces and dots for normalized comparison
    cleaned = cleaned.replace(" ", "").replace(".", "")

    # Normalize Tamil vowel length variations
    replacements = {
        "ஆ": "அ",
        "ஈ": "இ",
        "ஊ": "உ",
        "ே": "ெ",
        "ோ": "ொ",
    }

    for k, v in replacements.items():
        cleaned = cleaned.replace(k, v)

    # Handle common name patterns
    cleaned = cleaned.replace("ாமூர்த்தி", "மூர்த்தி")

    return cleaned.strip()


def flexible_author_match(search_name: str, csv_name: str) -> bool:
    """Match a search name against a (possibly multi-author) CSV cell.

    Tries five strategies in order: exact, substring, special-case aliases
    (கலைஞர்/கருணாநிதி etc.), token-level exact, and fuzzy (≥ FUZZY_THRESHOLD)
    / edit-distance-one. Returns True on the first strategy that succeeds.

    Args:
        search_name: Author name from the user query.
        csv_name: Raw author cell value from the CSV.

    Returns:
        True if any matching strategy succeeds, False otherwise.
    """
    search_normalized = normalize_author_name(search_name).lower().strip()
    if not search_normalized:
        return False

    csv_authors = _parse_csv_authors(csv_name)

    special_cases = {
        "கலைஞர்": ["கருணாநிதி", "மு.கருணாநிதி", "மு. கருணாநிதி"],
        "கருணாநிதி": ["கலைஞர்", "மு.கருணாநிதி", "மு. கருணாநிதி"],
        "அண்ணா": ["அண்ணாதுரை", "சி.என்.அண்ணாதுரை"],
        "அண்ணாதுரை": ["அண்ணா", "சி.என்.அண்ணாதுரை"],
        "பாரதிதாசன்": ["பாவேந்தர்", "பாவேந்தர் பாரதிதாசன்"],
        "பாவேந்தர்": ["பாரதிதாசன்", "பாவேந்தர் பாரதிதாசன்"],
        "கண்ணதாசன்": ["கவியரசு", "கவியரசு கண்ணதாசன்"],
        "கவியரசு": ["கண்ணதாசன்", "கவியரசு கண்ணதாசன்"],
    }

    for author in csv_authors:
        csv_normalized = normalize_author_name(author).lower().strip()
        if not csv_normalized:
            continue

        # 1. Exact
        if search_normalized == csv_normalized:
            return True

        # 2. Substring (reject when non-Tamil/Latin chars are directly
        # concatenated, e.g. "கருணாநிதிxyz" should not match "கருணாநிதி")
        if search_normalized in csv_normalized or csv_normalized in search_normalized:
            if len(search_normalized) <= len(csv_normalized):
                shorter, longer = search_normalized, csv_normalized
            else:
                shorter, longer = csv_normalized, search_normalized
            idx = longer.index(shorter)
            extra = longer[:idx] + longer[idx + len(shorter) :]
            if not extra or not re.search(r"[a-zA-Z]", extra):
                return True

        # 3. Special-case aliases
        for key, variations in special_cases.items():
            if key in search_normalized:
                for variant in variations:
                    if (
                        variant.lower().strip()
                        and variant.lower().strip() in csv_normalized
                    ):
                        return True
            if key in csv_normalized:
                for variant in variations:
                    if (
                        variant.lower().strip()
                        and variant.lower().strip() in search_normalized
                    ):
                        return True

        # 4. Token-level match (exact tokens, min length 4)
        # Skip if Tamil+Latin chars are directly concatenated (e.g. "கருணாநிதிxyz")
        # — extracting only Tamil tokens would ignore the gibberish suffix.
        mixed_script = re.search(
            r"[\u0B80-\u0BFF][a-zA-Z]|[a-zA-Z][\u0B80-\u0BFF]", search_normalized
        ) or re.search(
            r"[\u0B80-\u0BFF][a-zA-Z]|[a-zA-Z][\u0B80-\u0BFF]", csv_normalized
        )
        search_tokens = (
            [
                t
                for t in re.findall(r"[\u0B80-\u0BFF]+", search_normalized)
                if len(t) >= 4
            ]
            if not mixed_script
            else []
        )
        csv_tokens = [
            t for t in re.findall(r"[\u0B80-\u0BFF]+", csv_normalized) if len(t) >= 4
        ]
        if search_tokens and csv_tokens:
            if all(any(st == ct for ct in csv_tokens) for st in search_tokens):
                return True

        # 5. Fuzzy >= 95%
        if fuzzy_match_score(search_normalized, csv_normalized) >= FUZZY_THRESHOLD:
            return True

        # 6. Edit distance <= 1
        if _edit_distance_one(search_normalized, csv_normalized):
            return True

    return False


def _find_closest_author(df: pd.DataFrame, search_name: str) -> tuple:
    """Return the closest author name and fuzzy score for a search query.

    Scans every individual parsed author name in the DataFrame and returns
    the best fuzzy match. Used to generate spelling suggestions on miss.

    Args:
        df: DataFrame containing the ``ஆசிரியர்`` column.
        search_name: Author name to match against.

    Returns:
        Tuple of (best_author_name, best_score). Empty string and 0.0 if
        the DataFrame has no author entries.
    """
    best_name, best_score = "", 0.0
    search_norm = normalize_author_name(search_name).lower()

    all_authors = []
    for val in df["ஆசிரியர்"].dropna():
        all_authors.extend(_parse_csv_authors(str(val)))

    for name in set(all_authors):
        score = fuzzy_match_score(search_norm, normalize_author_name(name).lower())
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, best_score


def _find_closest_title(df: pd.DataFrame, topic: str) -> tuple:
    """Return the closest article title and fuzzy score for a topic query.

    Args:
        df: DataFrame containing the ``தலைப்பு`` column.
        topic: Topic string to match against.

    Returns:
        Tuple of (best_title, best_score). Empty string and 0.0 on no match.
    """
    best_title, best_score = "", 0.0
    topic_lower = topic.lower()
    for title in df["தலைப்பு"].dropna().unique():
        score = fuzzy_match_score(topic_lower, str(title).lower())
        if score > best_score:
            best_score = score
            best_title = str(title)
    return best_title, best_score


# MAIN QUERY SYSTEM
class EnhancedAuthorQuerySystem:
    """CSV-backed query system for author and topic lookups.

    Loads the Ponni article summary CSV at construction time and exposes
    methods for listing authors, retrieving articles by author, and finding
    authors by article title. Query routing and entity extraction are also
    handled here.
    """

    def __init__(self, csv_path: str):
        """Initialize with path to the summary CSV file."""
        self.csv_path = Path(csv_path)
        self.df = None
        self._load_csv()

    def _load_csv(self):
        """Load and validate the CSV, populating ``self.df``.

        Uses _load_csv_safe so multi-author bracket rows are not dropped.
        Filters out rows with empty author fields and normalises column names.
        Sets self.df to an empty DataFrame on any failure.
        """
        try:
            if not self.csv_path.exists():
                logger.error(f"CSV not found: {self.csv_path}")
                self.df = pd.DataFrame()
                return

            logger.info(f"Loading CSV: {self.csv_path}")
            self.df = _load_csv_safe(self.csv_path)

            if self.df is None or self.df.empty:
                logger.error("CSV loading failed")
                self.df = pd.DataFrame()
                return

            logger.info(f"Columns: {list(self.df.columns)}")
            logger.info(f"Rows: {len(self.df)}")

            self._fix_column_names()

            if "ஆசிரியர்" in self.df.columns:
                self.df["ஆசிரியர்"] = (
                    self.df["ஆசிரியர்"].fillna("").astype(str).str.strip()
                )
                before = len(self.df)
                self.df = self.df[self.df["ஆசிரியர்"] != ""]
                logger.info(
                    f"Authors: {before} → {len(self.df)} rows after filtering empty"
                )
            else:
                logger.error("Missing 'ஆசிரியர்' column")
                return

            if "தலைப்பு" in self.df.columns:
                self.df["தலைப்பு"] = (
                    self.df["தலைப்பு"].fillna("").astype(str).str.strip()
                )

            if len(self.df) > 0:
                logger.info(
                    f"Sample author: {self.df.iloc[0].get('ஆசிரியர்', 'N/A')[:40]}"
                )

        except Exception as e:
            logger.error(f"CSV error: {e}", exc_info=True)
            self.df = pd.DataFrame()

    def _fix_column_names(self):
        """Rename English column aliases to their Tamil canonical names."""
        mappings = {
            "author": "ஆசிரியர்",
            "Author": "ஆசிரியர்",
            "title": "தலைப்பு",
            "Title": "தலைப்பு",
            "heading": "தலைப்பு",
        }
        for old, new in mappings.items():
            if old in self.df.columns and new not in self.df.columns:
                self.df.rename(columns={old: new}, inplace=True)
                logger.info(f"Renamed '{old}' → '{new}'")

    # Content-seeking words — when present, the query is asking about
    # meaning/theme/summary of a specific work, NOT requesting an article
    # listing.  Checked FIRST so that "கருணாநிதி அவர்கள் எழுதிய வளையல்
    # வாங்கலீயோ கதையின் கருத்து என்ன" goes to vector search even though
    # "அவர்கள் எழுதிய" matches AUTHOR_ACTION.
    _CONTENT_SEEKING = [
        "கருத்து",
        "சுருக்கம்",
        "சுருக்கமாக",
        "கதைச் சுருக்கம்",
        "கதை சுருக்கம்",
        "உள்ளடக்கம்",
        "பொருள்",
        "விளக்கம்",
        "விளக்குக",
        "அர்த்தம்",
        "காட்டு",
        "காட்டுக",
        "படிக்க",
        "முழு கதை",
        "முழு கட்டுரை",
        "முழு கவிதை",
        "theme",
        "summary",
        "meaning",
        "explain",
        "describe",
        "content",
        "full text",
        "show",
    ]

    def detect_query_type(self, question: str) -> str:
        """Classify a question as a CSV query type or 'none'.

        Checks content-seeking intent first (routes to vector search), then
        tests for list-authors, topic-author, and author-topics patterns.

        Args:
            question: Raw user question string.

        Returns:
            One of ``"list_all_authors"``, ``"author_topics"``,
            ``"topic_author"``, or ``"none"``.
        """
        q = question.lower().strip()
        YEAR_PATTERNS = ["எந்த ஆண்டு", "ஆண்டு", "வெளியான ஆண்டு"]
        if any(p in q for p in YEAR_PATTERNS):
            return "none"

        # Content-seeking intent overrides all CSV patterns.
        if any(p in q for p in self._CONTENT_SEEKING):
            logger.info("[QUERY_TYPE] Content-seeking query — bypassing CSV")
            return "none"

        if any(p in q for p in _PatternBank.LIST_ALL_AUTHORS):
            return "list_all_authors"

        # Topic-author patterns (who wrote X?) with word-boundary fix:
        # "யார்" patterns must match as standalone words, not as suffixes
        # of author names (e.g. "பெரியார்" ends with "யார்").
        topic_author_match = False
        for p in _PatternBank.TOPIC_AUTHOR:
            if p in q:
                if "யார்" in p:
                    yaar_pos = q.index(p) + p.index("யார்")
                    if yaar_pos > 0 and q[yaar_pos - 1] != " ":
                        continue  # "யார்" is part of a larger word — skip
                topic_author_match = True
                break
        if topic_author_match:
            return "topic_author"

        has_known_author = any(a in q for a in _PatternBank.KNOWN_AUTHORS)
        has_initials_name = bool(_RE_INITIALS_NAME.search(q))

        if has_known_author or has_initials_name:
            has_author_action = any(p in q for p in _PatternBank.AUTHOR_ACTION)
            if has_author_action:
                return "author_topics"

        topic_content_patterns = [
            "பற்றிய படைப்புகள்",
            "பற்றிய கட்டுரைகள்",
            "பற்றிய கவிதைகள்",
            "பற்றிய கதைகள்",
            "பற்றி படைப்புகள்",
            "பற்றி கட்டுரைகள்",
            "தொடர்பான படைப்புகள்",
            "தொடர்பான கட்டுரைகள்",
            "குறித்த படைப்புகள்",
            "குறித்த கட்டுரைகள்",
            "சார்ந்த படைப்புகள்",
            "சார்ந்த கட்டுரைகள்",
        ]
        if any(p in q for p in topic_content_patterns):
            return "topic_author"
        # 🔥 NEW: handle exam-style queries like "எந்த மலர்", "இதழில் இடம்பெற்றுள்ளது"
        special_patterns = [
            "எந்த மலர்",
            "எந்த இதழ்",
            "இதழில் இடம்பெற்றுள்ளது",
            "எந்த மலர் மற்றும் இதழில்",
        ]

        if any(p in q for p in special_patterns):
            return "topic_author"
        return "none"

    def extract_entity(self, question: str, query_type: str) -> str:
        """Extract the key entity (author name or topic) from a question.

        For ``author_topics``: applies canonical author mappings, strips
        possessive suffixes, and falls back to token extraction.
        For ``topic_author``: strips noise phrases and title suffixes to
        isolate the article title fragment.

        Args:
            question: Raw user question string.
            query_type: One of ``"author_topics"`` or ``"topic_author"``.

        Returns:
            Extracted entity string, or empty string if extraction fails.
        """
        q = question.strip()

        for variation in _PatternBank.PONNI:
            if variation not in ("பொன்னி", "ponni"):
                q = q.replace(variation, "பொன்னி")
        q = re.sub(r"ponni\b", "பொன்னி", q, flags=re.IGNORECASE)

        if query_type == "author_topics":
            q_lower_raw = q.lower()
            for pattern, canonical in _PatternBank.AUTHOR_CANONICAL.items():
                if pattern in q_lower_raw:
                    logger.info(f"[CANONICAL-PRESUFFIX] '{pattern}' → '{canonical}'")
                    return canonical

            q_stripped = _strip_tamil_possessive_suffixes(q)
            q_lower = q_stripped.lower()
            for pattern, canonical in _PatternBank.AUTHOR_CANONICAL.items():
                if pattern in q_lower:
                    logger.info(f"[CANONICAL] '{pattern}' → '{canonical}'")
                    return canonical

            noise = sorted(_PatternBank.AUTHOR_NOISE_PHRASES, key=len, reverse=True)
            q_clean = q_stripped
            for nw in noise:
                q_clean = re.sub(re.escape(nw), " ", q_clean, flags=re.IGNORECASE)
            q_clean = re.sub(r"\s+", " ", q_clean).strip()

            m = _RE_INITIALS_NAME.search(q_clean)
            if m:
                candidate = _strip_tamil_possessive_suffixes(m.group(1).strip()).strip()
                bad = _PatternBank.AUTHOR_NOISE_TOKENS
                if not any(bw in candidate for bw in bad) and len(candidate) > 2:
                    return normalize_author_name(candidate)

            tokens = _RE_TAMIL_WORD.findall(q_clean)
            tokens = [_strip_tamil_possessive_suffix_word(t) for t in tokens]
            tokens = [
                normalize_author_name(t)
                for t in tokens
                if len(t) > 2 and t not in _PatternBank.AUTHOR_NOISE_TOKENS
            ]
            return " ".join(tokens[:3]) if tokens else ""

        elif query_type == "topic_author":
            # 🔥 STEP 1: Extract title using "என்ற"
            match = re.search(r"(.*?)\s*என்ற", q)
            if match:
                title = match.group(1).strip()
                title = re.sub(
                    r"(செய்தி|பாட்டு|பாடல்|கதை|கவிதை|கட்டுரை|சிறுகதை|தொடர்கதை).*",
                    "",
                    title,
                ).strip()
                return title

            # 🔥 STEP 2: Remove noise words
            q = q.replace("செய்திப் பாட்டு", "")
            q = q.replace("பாட்டு", "")

            # 🔥 STEP 3: Clean question words
            noise = sorted(_PatternBank.TOPIC_NOISE_PHRASES, key=len, reverse=True)
            q_title = q
            for np_phrase in noise:
                q_title = re.sub(
                    re.escape(np_phrase), "  ", q_title, flags=re.IGNORECASE
                )

            for sw in _PatternBank.TITLE_SUFFIXES:
                q_title = re.sub(
                    r"\s*" + re.escape(sw) + r"\s*$",
                    "",
                    q_title.strip(),
                    flags=re.IGNORECASE,
                )

            q_title = _strip_tamil_possessive_suffixes(q_title)

            # normalize endings
            for _pat, _repl in [
                (r"யில்(?=\s|$)", ""),
                (r"வில்(?=\s|$)", ""),
            ]:
                q_title = re.sub(_pat, _repl, q_title)

            q_title = q_title.replace('"', "").replace("'", "")
            q_title = re.sub(r"\s+", " ", q_title).strip()

            # extract meaningful words
            words = re.findall(r"[\u0B80-\u0BFF]+", q_title)
            words = [w for w in words if len(w) > 2]

            return " ".join(words[:3]) if words else ""
        return ""

    def list_all_authors(self) -> Dict:
        """Return all unique authors with their article counts.

        Each name in a multi-author row is counted individually.

        Returns:
            Dict with keys: success, type, total_authors, total_articles,
            authors (list of {name, count} dicts sorted by count descending).
        """
        all_authors = []
        for val in self.df["ஆசிரியர்"]:
            all_authors.extend(_parse_csv_authors(val))

        counts = pd.Series(all_authors).value_counts()
        authors = [
            {"name": author, "count": int(count)} for author, count in counts.items()
        ]

        return {
            "type": "list_all_authors",
            "success": True,
            "total_authors": len(authors),
            "total_articles": len(self.df),
            "authors": authors,
        }

    def get_topics_by_author(self, author_name: str) -> Dict:
        """Return all articles written by the given author.

        Uses flexible_author_match for lookup. On miss, returns a fuzzy
        suggestion if a close name (≥ 70 %) exists in the CSV.

        Args:
            author_name: Author name from the user query.

        Returns:
            Dict with keys: success, type, author, matched_author, count,
            articles. On failure, includes a ``suggestion`` key when a
            close match is found.
        """
        if self.df is None or self.df.empty:
            return {
                "type": "author_topics",
                "success": False,
                "message": "CSV தரவு கிடைக்கவில்லை",
                "articles": [],
            }

        matches = self.df[
            self.df["ஆசிரியர்"].apply(
                lambda x: flexible_author_match(author_name, str(x))
            )
        ]

        if matches.empty:
            best_name, best_score = _find_closest_author(self.df, author_name)
            suggestion = ""
            if best_score >= 0.70:
                suggestion = f"நீங்கள் '{best_name}' என்பவரை குறிப்பிட்டீர்களா? (ஒற்றுமை: {best_score:.0%})"
            return {
                "type": "author_topics",
                "success": False,
                "author": author_name,
                "message": f"'{author_name}' கண்டுபிடிக்க முடியவில்லை",
                "suggestion": suggestion,
                "articles": [],
            }

        articles = []
        for _, row in matches.iterrows():
            article = {
                "title": row.get("தலைப்பு", ""),
                "author": _format_author_display(str(row.get("ஆசிரியர்", ""))),
            }
            for col in ["ஆண்டு", "இதழ்", "ச.எ.", "வ.எ."]:
                if col in row and pd.notna(row[col]):
                    try:
                        article[col] = (
                            int(row[col])
                            if col in ["ஆண்டு", "ச.எ.", "வ.எ."]
                            else str(row[col])
                        )
                    except Exception:
                        article[col] = str(row[col])
            articles.append(article)

        return {
            "type": "author_topics",
            "success": True,
            "author": author_name,
            "matched_author": matches.iloc[0]["ஆசிரியர்"],
            "count": len(articles),
            "articles": articles,
        }

    def get_author_by_topic(self, topic: str) -> Dict:
        """Return articles whose titles match the given topic string.

        Applies six progressive match stages in order:
        1. Exact substring match.
        2. All query words present in title.
        3. Majority-word match (≥ 60 % of words, result ≤ 8 rows).
        4. Longest Tamil word match (≥ 6 chars, result ≤ 10 rows).
        5. Single-character deletion variants of the longest word.
        6. Whole-title fuzzy (≥ 0.72) or token-level fuzzy (≥ 0.79).

        Args:
            topic: Topic or title fragment from the user query.

        Returns:
            Dict with keys: success, type, topic, cleaned_topic, count,
            articles. On failure, includes a ``suggestion`` key when a
            close title (≥ 70 %) is found.
        """
        if self.df is None or self.df.empty:
            return {
                "type": "topic_author",
                "success": False,
                "message": "CSV தரவு கிடைக்கவில்லை",
                "articles": [],
            }

        topic_cleaned = topic
        for suffix in _PatternBank.TITLE_SUFFIXES:
            topic_cleaned = re.sub(
                r"\s*" + re.escape(suffix) + r"\s*$",
                "",
                topic_cleaned.strip(),
                flags=re.IGNORECASE,
            )
        topic_cleaned = _strip_tamil_possessive_suffixes(topic_cleaned)
        topic_cleaned = (
            topic_cleaned.replace("'", "").replace('"', "").replace("?", "").strip()
        )

        def _build_articles(df_slice):
            articles = []
            for _, row in df_slice.iterrows():
                article = {
                    "title": row.get("தலைப்பு", ""),
                    "author": _format_author_display(str(row.get("ஆசிரியர்", ""))),
                }
                for col in ["ஆண்டு", "இதழ்", "ச.எ.", "வ.எ."]:
                    if col in row and pd.notna(row[col]):
                        try:
                            article[col] = (
                                int(row[col])
                                if col in ["ஆண்டு", "ச.எ.", "வ.எ."]
                                else str(row[col])
                            )
                        except Exception:
                            article[col] = str(row[col])
                articles.append(article)
            return articles

        # Stage 1: exact substring
        matches = self.df[
            self.df["தலைப்பு"].str.contains(
                topic_cleaned, case=False, na=False, regex=False
            )
        ]

        # Stage 2: all words present
        if matches.empty and len(topic_cleaned.split()) > 1:
            words = topic_cleaned.split()
            matches = self.df[
                self.df["தலைப்பு"].apply(
                    lambda t: pd.notna(t)
                    and all(w.lower() in str(t).lower() for w in words)
                )
            ]

        # Stage 2b: majority-word match
        if matches.empty and len(topic_cleaned.split()) > 2:
            meaningful_words = [w for w in topic_cleaned.split() if len(w) >= 3]
            if len(meaningful_words) >= 2:
                min_matches = max(2, int(len(meaningful_words) * 0.6))
                majority_rows = []
                for _, row in self.df.iterrows():
                    title = str(row.get("தலைப்பு", ""))
                    if not title:
                        continue
                    title_tokens = set(
                        re.findall(r"[\u0B80-\u0BFF]+|[a-zA-Z]+", title.lower())
                    )
                    if (
                        sum(1 for w in meaningful_words if w.lower() in title_tokens)
                        >= min_matches
                    ):
                        majority_rows.append(row)
                if majority_rows and len(majority_rows) <= 8:
                    matches = pd.DataFrame(majority_rows)

        # Stage 3: longest Tamil word (>= 6 chars), result set <= 10
        if matches.empty:
            tamil_words = [
                w for w in topic_cleaned.split() if re.search(r"[\u0B80-\u0BFF]{6,}", w)
            ]
            if tamil_words:
                main_word = max(tamil_words, key=len)
                candidate = self.df[
                    self.df["தலைப்பு"].str.contains(
                        main_word, case=False, na=False, regex=False
                    )
                ]
                if not candidate.empty and len(candidate) <= 10:
                    matches = candidate

        # Stage 3b: 1-char deletion variants of the longest Tamil word.
        #
        # Catches user spelling mistakes where a single character is inserted,
        # swapped, or differs from the CSV title, e.g.:
        #   வாங்கிலீயோ  → வாங்கலீயோ   (extra 'இ' inserted by user)
        #   வருந்தாதீர் → வருத்தாதீர்  (ந்→த் consonant swap)
        #   அறியாதவள்  → அறியாதவன்   (ள்→ன் gender suffix swap)
        #
        # Generates every single-character deletion of the longest Tamil word
        # in the query and does a substring search for each variant.
        # Accepts result only if <= 8 rows (keeps it precise).
        if matches.empty:
            tamil_words_in_query = [
                w for w in re.findall(r"[\u0B80-\u0BFF]+", topic_cleaned) if len(w) >= 5
            ]
            if tamil_words_in_query:
                longest_word = max(tamil_words_in_query, key=len)
                logger.info(
                    f"Stage 3b: generating deletion variants for '{longest_word}'"
                )
                deletion_variants = [
                    longest_word[:i] + longest_word[i + 1 :]
                    for i in range(len(longest_word))
                    if len(longest_word) - 1 >= 4
                ]
                for variant in deletion_variants:
                    candidate = self.df[
                        self.df["தலைப்பு"].str.contains(
                            variant, case=False, na=False, regex=False
                        )
                    ]
                    if not candidate.empty and len(candidate) <= 8:
                        logger.info(
                            f"Stage 3b matched via variant '{variant}' ({len(candidate)} rows)"
                        )
                        matches = candidate
                        break

        # Stage 4: whole-title fuzzy >= 0.72 OR token-level fuzzy >= 0.75.
        #
        # Threshold lowered from 0.80 → 0.72 to catch single-character Tamil
        # spelling differences (vowel markers, consonant clusters).
        #
        # Token-level fallback: splits both query and title into individual Tamil
        # words (>= 4 chars) and requires every query token to fuzzy-match at
        # least one title token at >= 0.75. Handles cases where the misspelled
        # word is only part of a multi-word title — the whole-title score gets
        # diluted by correctly spelled words, but the per-token score stays high.
        if matches.empty:
            fuzzy_rows = []
            search_tokens = [
                t for t in re.findall(r"[\u0B80-\u0BFF]+", topic_cleaned) if len(t) >= 4
            ]
            for _, row in self.df.iterrows():
                title = str(row.get("தலைப்பு", ""))
                if not title:
                    continue

                # Whole-title fuzzy (threshold lowered to 0.72)
                whole_score = fuzzy_match_score(topic_cleaned, title)
                if whole_score >= 0.72:
                    fuzzy_rows.append((row, whole_score))
                    continue

                # Token-level fuzzy: every search token must match
                # some title token at >= 0.75
                if search_tokens:
                    title_tokens = [
                        t for t in re.findall(r"[\u0B80-\u0BFF]+", title) if len(t) >= 4
                    ]
                    if title_tokens:
                        all_matched = all(
                            any(
                                fuzzy_match_score(st, tt) >= 0.79 for tt in title_tokens
                            )
                            for st in search_tokens
                        )
                        if all_matched:
                            best_token_score = min(
                                max(fuzzy_match_score(st, tt) for tt in title_tokens)
                                for st in search_tokens
                            )
                            # Slightly discount token-level vs whole-title matches
                            fuzzy_rows.append((row, best_token_score * 0.95))

            if fuzzy_rows:
                fuzzy_rows.sort(key=lambda x: x[1], reverse=True)
                matches = pd.DataFrame([r for r, _ in fuzzy_rows])

        # Stage 5: all long tokens (>= 5 chars) — AND logic
        if matches.empty:
            tokens = [t for t in topic_cleaned.split() if len(t) >= 5]
            if len(tokens) >= 2:
                candidate = self.df[
                    self.df["தலைப்பு"].apply(
                        lambda t: pd.notna(t)
                        and all(tok.lower() in str(t).lower() for tok in tokens)
                    )
                ]
                if not candidate.empty:
                    matches = candidate
            elif len(tokens) == 1 and len(tokens[0]) >= 7:
                candidate = self.df[
                    self.df["தலைப்பு"].str.contains(
                        tokens[0], case=False, na=False, regex=False
                    )
                ]
                if not candidate.empty and len(candidate) <= 5:
                    matches = candidate

        if matches.empty:
            best_title, best_score = _find_closest_title(self.df, topic_cleaned)
            suggestion = ""
            if best_score >= 0.70:
                suggestion = f"நீங்கள் '{best_title}' என்ற தலைப்பை குறிப்பிட்டீர்களா? (ஒற்றுமை: {best_score:.0%})"
            return {
                "type": "topic_author",
                "success": False,
                "topic": topic,
                "cleaned_topic": topic_cleaned,
                "message": f"'{topic}' தலைப்பு கண்டுபிடிக்க முடியவில்லை. தேடிய சொல்: '{topic_cleaned}'",
                "suggestion": suggestion,
                "articles": [],
            }

        return {
            "type": "topic_author",
            "success": True,
            "topic": topic,
            "cleaned_topic": topic_cleaned,
            "count": len(matches),
            "articles": _build_articles(matches),
        }


# FORMATTERS
def format_author_list(result: Dict) -> str:
    """Format a list_all_authors result as a Tamil display string.

    Args:
        result: Dict returned by list_all_authors.

    Returns:
        Human-readable Tamil string with ranked author list.
    """
    if not result["success"]:
        return f"Error: {result['message']}"
    lines = [
        "பொன்னி இதழில் எழுதிய எழுத்தாளர்கள்",
        f"மொத்த எழுத்தாளர்கள்: {result['total_authors']} | மொத்த கட்டுரைகள்: {result['total_articles']}",
        "",
    ]
    for idx, author in enumerate(
        sorted(result["authors"], key=lambda x: x["count"], reverse=True), 1
    ):
        lines.append(f"{idx}. {author['name']} ({author['count']} கட்டுரைகள்)")
    return "\n".join(lines)


def format_author_topics(result: Dict) -> str:
    """Format a get_topics_by_author result as a Tamil display string.

    Args:
        result: Dict returned by get_topics_by_author.

    Returns:
        Human-readable Tamil string with articles sorted by year and issue.
    """
    if not result["success"]:
        return f"Error: {result['message']}"
    lines = [
        f"எழுத்தாளர்: {result.get('matched_author', result['author'])}",
        f"மொத்த படைப்புகள்: {result['count']}",
        "",
    ]
    for idx, article in enumerate(
        sorted(
            result["articles"],
            key=lambda x: (x.get("ஆண்டு", 9999), str(x.get("இதழ்", ""))),
        ),
        1,
    ):
        parts = [
            f"தலைப்பு: {article.get('title', '-')}",
            f"ஆசிரியர்: {article.get('author', '-')}",
        ]
        if "ஆண்டு" in article:
            parts.append(f"ஆண்டு: {article['ஆண்டு']}")
        if "இதழ்" in article:
            parts.append(f"இதழ்: {article['இதழ்']}")
        lines.append(f"{idx}. {' | '.join(parts)}")
    return "\n".join(lines)


def format_topic_authors(result: Dict) -> str:
    """Format a get_author_by_topic result as a Tamil display string.

    Args:
        result: Dict returned by get_author_by_topic.

    Returns:
        Human-readable Tamil string, including suggestion text on failure.
    """
    if not result["success"]:
        msg = f"Error: {result['message']}"
        if "suggestion" in result:
            msg += f"\n\nசிபாரிசு: {result['suggestion']}"
        return msg
    lines = [f"தலைப்பு: '{result['topic']}' பற்றிய கட்டுரைகள்"]
    if result.get("cleaned_topic") and result["cleaned_topic"] != result["topic"]:
        lines.append(f"(தேடிய சொல்: '{result['cleaned_topic']}')")
    lines.extend([f"கண்டுபிடிக்கப்பட்டவை: {result['count']}", ""])
    for idx, article in enumerate(result["articles"], 1):
        parts = [
            f"தலைப்பு: {article.get('title', '-')}",
            f"ஆசிரியர்: {article.get('author', '-')}",
        ]
        if "ஆண்டு" in article:
            parts.append(f"ஆண்டு: {article['ஆண்டு']}")
        if "இதழ்" in article:
            parts.append(f"இதழ்: {article['இதழ்']}")
        lines.append(f"{idx}. {' | '.join(parts)}")
    return "\n".join(lines)


# ISSUE COUNT & START YEAR
def detect_issue_count_query(question: str) -> bool:
    """Return True if the question asks for a total issue or article count.

    Args:
        question: Raw user question string.
    """
    q = question.lower()
    patterns = [
        "இதழ் எண்ணிக்கை",
        "எத்தனை இதழ்",
        "இதழ்கள் எத்தனை",
        "மொத்த இதழ்",
        "இதழ் count",
        "issue count",
        "how many issues",
        "number of issues",
        "total issues",
        "இதழ் பட்டியல்",
        "இதழ்களின் பட்டியல்",
    ]
    return any(pattern in q for pattern in patterns)


def get_issue_count(csv_path: str) -> Dict:
    """Return per-issue article counts and totals from the CSV.

    Args:
        csv_path: Path to the article summary CSV.

    Returns:
        Dict with keys: success, count, total_articles, issues
        (list of {issue_number, article_count} dicts in sorted order).
    """
    try:
        df = _load_csv_safe(csv_path)
        if df.empty:
            return {
                "success": False,
                "message": "CSV தரவை படிக்க முடியவில்லை",
                "count": 0,
                "issues": [],
            }

        issue_col = next(
            (c for c in ["இதழ்", "issue", "Issue", "doc_issue"] if c in df.columns),
            None,
        )
        if not issue_col:
            return {
                "success": False,
                "message": "இதழ் column கிடைக்கவில்லை",
                "count": 0,
                "issues": [],
            }

        df[issue_col] = df[issue_col].fillna("").astype(str).str.strip()
        df_filtered = df[df[issue_col] != ""]
        unique_issues = df_filtered[issue_col].unique()
        issue_counts = df_filtered[issue_col].value_counts()

        try:
            sorted_issues = sorted(
                unique_issues, key=lambda x: int(x) if x.isdigit() else x
            )
        except Exception:
            sorted_issues = sorted(unique_issues)

        return {
            "success": True,
            "count": len(unique_issues),
            "total_articles": len(df_filtered),
            "issues": [
                {"issue_number": i, "article_count": int(issue_counts[i])}
                for i in sorted_issues
            ],
        }

    except Exception as e:
        logger.error(f"Error getting issue count: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"பிழை: {str(e)}",
            "count": 0,
            "issues": [],
        }


def format_issue_count(result: Dict) -> str:
    """Format a get_issue_count result as a Tamil display string.

    Shows all issues when ≤ 20; otherwise shows first 10 and last 10
    with summary statistics (mean, min, max articles per issue).

    Args:
        result: Dict returned by get_issue_count.

    Returns:
        Human-readable Tamil string.
    """
    if not result["success"]:
        return f"Error: {result['message']}"
    lines = [
        "பொன்னி இதழ்கள் விவரம்",
        f"மொத்த இதழ்கள்: {result['count']}",
        f"மொத்த கட்டுரைகள்: {result['total_articles']}",
        "",
    ]
    issues = result["issues"]
    if len(issues) <= 20:
        for idx, info in enumerate(issues, 1):
            lines.append(
                f"{idx}. இதழ்: {info['issue_number']} | கட்டுரைகள்: {info['article_count']}"
            )
    else:
        lines.append("முதல் 10 இதழ்கள்:")
        for idx, info in enumerate(issues[:10], 1):
            lines.append(
                f"{idx}. இதழ்: {info['issue_number']} | கட்டுரைகள்: {info['article_count']}"
            )
        lines.extend(
            ["", f"... (மேலும் {len(issues) - 20} இதழ்கள்)", "", "கடைசி 10 இதழ்கள்:"]
        )
        for idx, info in enumerate(issues[-10:], len(issues) - 9):
            lines.append(
                f"{idx}. இதழ்: {info['issue_number']} | கட்டுரைகள்: {info['article_count']}"
            )
    lines.append("")
    if result["issues"]:
        counts = [i["article_count"] for i in result["issues"]]
        lines += [
            "புள்ளிவிவரம்:",
            f"  • சராசரி கட்டுரைகள் ஒரு இதழுக்கு: {sum(counts) / len(counts):.1f}",
            f"  • குறைந்தபட்ச கட்டுரைகள்: {min(counts)}",
            f"  • அதிகபட்ச கட்டுரைகள்: {max(counts)}",
        ]
    return "\n".join(lines)


def detect_start_year_query(question: str) -> bool:
    """Return True if the question asks when the magazine started.

    Args:
        question: Raw user question string.
    """
    q = question.lower()
    return any(
        p in q
        for p in [
            "எந்த ஆண்டு தொடங்கியது",
            "முதல் ஆண்டு",
            "தொடங்கிய ஆண்டு",
            "எப்போது தொடங்கியது",
            "start year",
            "when did ponni start",
        ]
    )


def get_start_year(csv_path: str) -> str:
    """Return the earliest publication year found in the CSV.

    Args:
        csv_path: Path to the article summary CSV.

    Returns:
        Tamil sentence stating the start year, or an error message.
    """
    df = _load_csv_safe(csv_path)
    if df.empty or "ஆண்டு" not in df.columns:
        return "CSV தரவில் ஆண்டு தகவல் இல்லை."
    try:
        df["ஆண்டு"] = pd.to_numeric(df["ஆண்டு"], errors="coerce")
        return f"பொன்னி இதழ் {int(df['ஆண்டு'].min())} ஆம் ஆண்டு தொடங்கியது.\n\n"
    except Exception as e:
        logger.error(f"Start year calculation error: {e}")
        return "ஆண்டு தகவலை கணக்கிட முடியவில்லை."


# MAIN ENTRY POINT
def handle_author_query(question: str, csv_path: str) -> Tuple[bool, str]:
    """Route a question to the appropriate CSV handler.

    Tries start-year, issue-count, and author/topic query handlers in
    order. Returns immediately on the first match. Falls back to vector
    search (returns False) when no handler claims the question or when
    a CSV lookup fails.

    Args:
        question: Raw user question string.
        csv_path: Path to the article summary CSV.

    Returns:
        Tuple of (is_handled, response_text). When is_handled is False,
        response_text is an empty string and the caller should proceed
        to vector search.
    """
    if detect_start_year_query(question):
        logger.info("Detected start year query")
        return True, get_start_year(csv_path)
    if detect_issue_count_query(question):
        logger.info("Detected issue count query")
        return True, format_issue_count(get_issue_count(csv_path))

    with _author_system_lock:
        if csv_path not in _author_system_cache:
            _author_system_cache[csv_path] = EnhancedAuthorQuerySystem(csv_path)
        system = _author_system_cache[csv_path]

    if system.df is None or system.df.empty:
        return False, ""

    query_type = system.detect_query_type(question)
    logger.info(f"Query type: {query_type}")

    if query_type == "none":
        if (
            any(word in question for word in ["இதழ்", "மலர்"])
            and "ஆண்டு" not in question
        ):
            query_type = "topic_author"
        else:
            return False, ""
    if query_type == "list_all_authors":
        return True, format_author_list(system.list_all_authors())
    if query_type == "author_topics":
        entity = system.extract_entity(question, "author_topics")
        logger.info(f"Extracted author: '{entity}'")
        if not entity:
            return False, ""  # Fall through to vector search
        result = system.get_topics_by_author(entity)
        if not result.get("success"):
            logger.info("[CSV] Author query failed — falling back to vector search")
            return False, ""
        return True, format_author_topics(result)
    if query_type == "topic_author":
        entity = system.extract_entity(question, "topic_author")
        logger.info(f"Extracted topic: '{entity}'")
        if not entity:
            return False, ""  # Fall through to vector search
        result = system.get_author_by_topic(entity)
        if not result.get("success"):
            logger.info("[CSV] Topic query failed — falling back to vector search")
            return False, ""
        return True, format_topic_authors(result)

    return False, ""


def _csv_source(csv_data: str) -> List[Dict]:
    return [
        {
            "volume": "பொன்னி கட்டுரை தரவுத்தளம்",
            "heading": "Article Database",
            "doc_issue": "",
            "content": csv_data,
            "word_count": len(csv_data.split()),
            "chunks_merged": 0,
            "score": 1.0,
        }
    ]


def _csv_data_suffix(csv_data: str) -> str:
    return f"\n\n---\n\n**தரவுத்தள தகவல்:**\n\n{csv_data}"


def _combine_csv_answer(llm_summary: str, csv_data: str) -> str:
    suffix = _csv_data_suffix(csv_data)
    if llm_summary and llm_summary.strip():
        return llm_summary.strip() + suffix
    return "கட்டுரை தரவுத்தளத்திலிருந்து பெறப்பட்ட தகவல்கள்:" + suffix


# QUERY SPELLING CORRECTION FOR VECTOR SEARCH
# Known question/action words — used both for skipping (exact match)
# and for correcting misspelled question words (fuzzy match).
_KNOWN_QUERY_WORDS = {
    "இதழில்",
    "இதழ்",
    "பொன்னி",
    "பொன்னியில்",
    "என்ன",
    "யாவை",
    "யார்",
    "எனும்",
    "பற்றி",
    "பற்றிய",
    "என்று",
    "உள்ள",
    "உள்ளது",
    "கட்டுரை",
    "கட்டுரைகள்",
    "எழுதிய",
    "எழுதியவர்",
    "ஆசிரியர்",
    "தொகுதி",
    "எப்படி",
    "எங்கே",
    "எப்போது",
    "கூறுக",
    "விளக்குக",
    "விவரி",
    "பட்டியலிடுக",
    "சுருக்கமாக",
    "சுருக்கம்",
    "கதைச்",
    "கதை",
    "கவிதை",
    "கவிதைகள்",
    "படைப்பு",
    "படைப்புகள்",
    "முக்கிய",
    "முக்கியமான",
    "தகவல்",
    "வெளியான",
    "வெளிவந்தது",
    "எழுதியுள்ளார்",
    "எழுதினார்",
    "இயற்றிய",
    "படைத்த",
    "தலைப்பு",
    "ஆசிரியர்கள்",
    "பங்களிப்பு",
    "பங்களித்த",
    "வரலாறு",
    "சிறுகதை",
    "சிறுகதைகள்",
    "நாடகம்",
    "நாடகங்கள்",
    "தொடர்கதை",
    "தொடர்கதைகள்",
}

# Minimum fuzzy score to accept a token-level correction.
# Raised from 0.75 → 0.88 to stop false "corrections" of valid Tamil
# content words (e.g. வடிவத்தை → படித்தவை at 0.75 wrecked recall).
_SPELLING_TOKEN_THRESHOLD = 0.88
# Minimum fuzzy score for whole-title correction.
_SPELLING_TITLE_THRESHOLD = 0.85
# Structural guards layered on top of the fuzzy score — the score alone
# is not a reliable signal between unrelated Tamil agglutinative forms
# that happen to share a suffix.
_SPELLING_MIN_PREFIX_MATCH = 3  # require ≥3 leading chars in common
_SPELLING_MAX_LEN_DIFF = 2  # reject candidates whose length differs by more


def _safe_to_correct(original: str, candidate: str) -> bool:
    """Return True if a spelling correction candidate is structurally plausible.

    Requires the original and candidate to share at least
    _SPELLING_MIN_PREFIX_MATCH leading characters and differ in length
    by no more than _SPELLING_MAX_LEN_DIFF. Guards against high fuzzy
    scores caused by shared Tamil suffixes between unrelated words.

    Args:
        original: Source token from the user query.
        candidate: Proposed corrected form.
    """
    if not original or not candidate:
        return False
    if abs(len(original) - len(candidate)) > _SPELLING_MAX_LEN_DIFF:
        return False
    common = 0
    for a, b in zip(original, candidate):
        if a == b:
            common += 1
        else:
            break
    return common >= _SPELLING_MIN_PREFIX_MATCH


def _get_title_vocab(df: pd.DataFrame) -> list:
    """Return unique Tamil words (≥ 4 chars) extracted from all CSV titles.

    Args:
        df: DataFrame containing the ``தலைப்பு`` column.
    """
    vocab = set()
    for title in df["தலைப்பு"].dropna().unique():
        for word in re.findall(r"[\u0B80-\u0BFF]+", str(title)):
            if len(word) >= 4:
                vocab.add(word)
    return list(vocab)


def _get_author_vocab(df: pd.DataFrame) -> list:
    """Return unique author name strings (≥ 4 chars) from the CSV.

    Args:
        df: DataFrame containing the ``ஆசிரியர்`` column.
    """
    authors = set()
    for val in df["ஆசிரியர்"].dropna():
        for name in _parse_csv_authors(str(val)):
            name = name.strip()
            if len(name) >= 4:
                authors.add(name)
    return list(authors)


def _correct_query_words(question: str) -> str:
    """Correct misspelled question/action words against _KNOWN_QUERY_WORDS.

    Only corrects tokens that are not already an exact match. Uses a
    threshold of 0.88 plus _safe_to_correct to avoid false corrections.

    Args:
        question: Raw user question string.

    Returns:
        Question with misspelled action words replaced.
    """
    query_words = re.findall(r"[\u0B80-\u0BFF]+", question)
    corrected = question
    known_list = list(_KNOWN_QUERY_WORDS)

    for word in query_words:
        if len(word) < 4:
            continue
        # Already a known word — no correction needed
        if word in _KNOWN_QUERY_WORDS:
            continue

        best_match = None
        best_score = 0.0
        for known in known_list:
            if len(known) < 4:
                continue
            score = fuzzy_match_score(word, known)
            if score > best_score:
                best_score = score
                best_match = known

        # Use a higher threshold (0.88) for question words to avoid
        # wrongly "correcting" title/author words into question words.
        # The structural guard further rejects suffix-coincidence matches.
        if best_match and best_score >= 0.88 and _safe_to_correct(word, best_match):
            logger.info(
                f"[SPELLING] query word '{word}' → '{best_match}' "
                f"(score: {best_score:.2f})"
            )
            corrected = corrected.replace(word, best_match, 1)

    return corrected


def correct_query_spelling(
    question: str, csv_path: str, content_vocab: List[str] = None
) -> str:
    """Correct misspelled Tamil words in a query using three vocabulary passes.

    Pass 1 — question/action words (எழூதிய → எழுதிய).
    Pass 2 — title and author words from the CSV.
    Pass 3 — content vocabulary from indexed documents (caller-supplied).

    Words already present as substrings in any reference text are skipped.
    Each correction requires fuzzy score ≥ _SPELLING_TOKEN_THRESHOLD and
    passes _safe_to_correct.

    Args:
        question: Raw user question string.
        csv_path: Path to the article summary CSV (used to load vocabulary).
        content_vocab: Optional list of words from Qdrant indexed headings.

    Returns:
        Corrected question string, or the original if no changes were made.
    """
    # Pass 1: correct question/action words
    corrected = _correct_query_words(question)

    # Pass 2: correct title/author words using CSV vocabulary
    with _author_system_lock:
        if csv_path not in _author_system_cache:
            _author_system_cache[csv_path] = EnhancedAuthorQuerySystem(csv_path)
        system = _author_system_cache[csv_path]

    # Combine all vocabularies for a single pass over query words
    title_vocab = []
    author_vocab = []
    all_titles = []

    if system.df is not None and not system.df.empty:
        df = system.df
        title_vocab = _get_title_vocab(df)
        author_vocab = _get_author_vocab(df)
        all_titles = [str(t) for t in df["தலைப்பு"].dropna().unique()]

    # Merge content vocab (from Qdrant headings) into a combined set
    combined_vocab = set(title_vocab)
    if content_vocab:
        combined_vocab.update(content_vocab)

    # All text to check for existing substring matches
    all_reference_text = list(all_titles)
    if content_vocab:
        all_reference_text.extend(content_vocab)

    # Extract Tamil words from the (already pass-1 corrected) query
    query_words = re.findall(r"[\u0B80-\u0BFF]+", corrected)

    for word in query_words:
        if len(word) < 4:
            continue
        # Strip possessive suffix for matching
        word_stripped = _strip_tamil_possessive_suffix_word(word)
        if word_stripped in _KNOWN_QUERY_WORDS or word in _KNOWN_QUERY_WORDS:
            continue

        # Check if word already exists as substring in titles or content vocab
        word_lower = word.lower()
        if any(word_lower in t.lower() for t in all_reference_text):
            continue

        # Also check stripped form
        stripped_lower = word_stripped.lower()
        if stripped_lower != word_lower and any(
            stripped_lower in t.lower() for t in all_reference_text
        ):
            continue

        # Try to find best match across all vocabularies
        best_match = None
        best_score = 0.0

        for vocab_word in combined_vocab:
            score = fuzzy_match_score(word_stripped, vocab_word)
            if score > best_score:
                best_score = score
                best_match = vocab_word

        # Also check author names
        for author_name in author_vocab:
            for author_word in re.findall(r"[\u0B80-\u0BFF]+", author_name):
                if len(author_word) < 4:
                    continue
                score = fuzzy_match_score(word_stripped, author_word)
                if score > best_score:
                    best_score = score
                    best_match = author_word

        if (
            best_match
            and best_score >= _SPELLING_TOKEN_THRESHOLD
            and _safe_to_correct(word_stripped, best_match)
        ):
            logger.info(
                f"[SPELLING] '{word}' → '{best_match}' " f"(score: {best_score:.2f})"
            )
            corrected = corrected.replace(word, best_match, 1)
        elif best_match and best_score >= _SPELLING_TOKEN_THRESHOLD:
            # Score passed but structural guard rejected — log so we can tune.
            logger.debug(
                f"[SPELLING] rejected '{word}' → '{best_match}' "
                f"(score={best_score:.2f}, len_diff="
                f"{abs(len(word_stripped) - len(best_match))})"
            )

    if corrected != question:
        logger.info(f"[SPELLING] Query corrected: '{question}' → '{corrected}'")

    return corrected
