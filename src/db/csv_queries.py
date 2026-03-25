"""CSV query pipeline for author/topic queries in Tamil document processing.

Handles author name matching, query type detection, entity extraction,
and CSV data formatting for the Ponni magazine archive.
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


# ============================================================================
# CSV LOADER HELPER
# ============================================================================


def load_csv(csv_path) -> pd.DataFrame:
    """Public alias for _load_csv_safe (backward compatibility)."""
    return _load_csv_safe(csv_path)


def _load_csv_safe(csv_path) -> pd.DataFrame:
    """
    Load CSV using load_csv (primary) then pd.read_csv fallbacks.

    load_csv from csv_fuzzy_matcher auto-wraps multi-author bracket fields
    like [பாண்டியன், நா. வேத்தரசன், வணங்காமுடி] in double quotes before
    pandas reads them — preventing silent row drops from on_bad_lines="skip".

    Used by _load_csv(), get_issue_count(), and get_start_year() so that
    ALL CSV reads in this module correctly include multi-author rows.
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


# ============================================================================
# AUTHOR FIELD PARSING
# ============================================================================


def _format_author_display(csv_name: str) -> str:
    """Format a CSV author cell for display: comma-separated, no brackets, 'NA' if empty."""
    authors = _parse_csv_authors(csv_name)
    return ", ".join(authors) if authors else "NA"


def _parse_csv_authors(csv_name: str) -> List[str]:
    """
    Parse a CSV author cell that may contain bracket-wrapped names.

    Handles:
      NA / "" / "[]"                           → []
      "[நக்கீரன்]"                              → ['நக்கீரன்']
      "[பாண்டியன், நா. வேத்தரசன், வணங்காமுடி]" → ['பாண்டியன்', 'நா. வேத்தரசன்', 'வணங்காமுடி']
      "நக்கீரன்"  (no brackets, legacy)          → ['நக்கீரன்']
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


# ============================================================================
# AUTHOR NAME NORMALISATION & MATCHING
# ============================================================================


def normalize_author_name(name: str) -> str:
    """Normalize an author name by stripping honorific prefixes."""
    if not name:
        return ""
    name = normalize_unicode(name)
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
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def flexible_author_match(search_name: str, csv_name: str) -> bool:
    """Flexibly match a search query against a CSV author field.

    Supports multi-author fields like "[A, B, C]".
    Checks each individual parsed author name against the search query.
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

        # 2. Substring (only when lengths are similar — prevents false
        # matches from unrelated suffixes like "கருணாநிதிxyz")
        shorter_len = min(len(search_normalized), len(csv_normalized))
        longer_len = max(len(search_normalized), len(csv_normalized))
        if longer_len > 0 and shorter_len / longer_len >= 0.8:
            if (
                search_normalized in csv_normalized
                or csv_normalized in search_normalized
            ):
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
        search_tokens = [
            t for t in re.findall(r"[\u0B80-\u0BFF]+", search_normalized) if len(t) >= 4
        ]
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
    """Return (best_author_name, best_score) scanning individual parsed names."""
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
    best_title, best_score = "", 0.0
    topic_lower = topic.lower()
    for title in df["தலைப்பு"].dropna().unique():
        score = fuzzy_match_score(topic_lower, str(title).lower())
        if score > best_score:
            best_score = score
            best_title = str(title)
    return best_title, best_score


# ============================================================================
# MAIN QUERY SYSTEM
# ============================================================================


class EnhancedAuthorQuerySystem:
    """Robust author query system with improved CSV parsing and query detection."""

    def __init__(self, csv_path: str):
        """Initialize with path to the summary CSV file."""
        self.csv_path = Path(csv_path)
        self.df = None
        self._load_csv()

    def _load_csv(self):
        """
        Load CSV using _load_csv_safe (load_csv primary, pd.read_csv fallbacks).

        FIXED: previously used pd.read_csv directly with on_bad_lines="skip",
        which silently dropped multi-author rows like
        [பாண்டியன், நா. வேத்தரசன், வணங்காமுடி] because commas inside
        brackets were treated as column separators.
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

    def detect_query_type(self, question: str) -> str:
        """Classify a question into a CSV query type or 'none'."""
        q = question.lower().strip()

        if any(p in q for p in _PatternBank.LIST_ALL_AUTHORS):
            return "list_all_authors"

        # Check known-author + action BEFORE topic_author patterns,
        # because author names can contain substrings that falsely match
        # topic_author patterns (e.g. "பெரியார்" contains "யார்").
        has_known_author = any(a in q for a in _PatternBank.KNOWN_AUTHORS)
        has_initials_name = bool(_RE_INITIALS_NAME.search(q))
        has_author_action = any(p in q for p in _PatternBank.AUTHOR_ACTION)

        if (has_known_author or has_initials_name) and has_author_action:
            return "author_topics"
        if (has_known_author or has_initials_name) and (
            "எழுதிய" in q or "படைப்பு" in q or "இயற்றிய" in q or "படைத்த" in q
        ):
            return "author_topics"

        if any(p in q for p in _PatternBank.TOPIC_AUTHOR):
            return "topic_author"

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

        return "none"

    def extract_entity(self, question: str, query_type: str) -> str:
        """Extract the key entity (author name or topic) from a question."""
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
            for _pat, _repl in [
                (r"த்தில்(?=\s|$)", "ம்"),
                (r"த்தின்(?=\s|$)", "ம்"),
                (r"த்திற்கு(?=\s|$)", "ம்"),
                (r"யில்(?=\s|$)", ""),
                (r"வில்(?=\s|$)", ""),
            ]:
                q_title = re.sub(_pat, _repl, q_title)
            q_title = q_title.replace('"', "").replace("'", "")
            q_title = re.sub(r"  +(யார்|என்ன|யாவை|எவர்)\??$", "", q_title)
            q_title = re.sub(r"\s+", " ", q_title).strip()

            words = re.findall(r"[\u0B80-\u0BFF!?,।]+|[a-zA-Z]+", q_title)
            words = [
                w
                for w in words
                if len(w.rstrip("?!,")) > 1
                and not (
                    w.rstrip("?!,").lower() in _PatternBank.TOPIC_NOISE_TOKENS
                    and not w.endswith("?")
                )
                and w not in {"?", "!", ",", "।"}
            ]
            result = " ".join(words) if words else ""
            if result.strip().lower().rstrip("?!") in {
                "யார்",
                "என்ன",
                "யாவை",
                "எவர்",
                "",
            }:
                return ""
            return result

        return ""

    def list_all_authors(self) -> Dict:
        """List all unique authors with article counts.

        Each individual name in a multi-author row is counted separately.
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
        """Return article titles written by the given author."""
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
        """Return authors who wrote about the given topic."""
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

        # Stage 4: fuzzy >= 0.80
        if matches.empty:
            fuzzy_rows = []
            for _, row in self.df.iterrows():
                title = str(row.get("தலைப்பு", ""))
                if title and fuzzy_match_score(topic_cleaned, title) >= 0.80:
                    fuzzy_rows.append((row, fuzzy_match_score(topic_cleaned, title)))
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


# ============================================================================
# FORMATTERS
# ============================================================================


def format_author_list(result: Dict) -> str:
    """Format a list_all_authors result dict as a Tamil display string."""
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
    """Format a get_topics_by_author result dict as a Tamil display string."""
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
    """Format a get_author_by_topic result dict as a Tamil display string."""
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


# ============================================================================
# ISSUE COUNT & START YEAR
# ============================================================================


def detect_issue_count_query(question: str) -> bool:
    """Return True if the question is asking for an issue/article count."""
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
    """
    Get unique issue count from CSV with article statistics.

    FIXED: uses _load_csv_safe (load_csv primary) so that multi-author rows
    are not dropped — the article count per issue will now be correct.
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
    """Format an issue-count result dict as a Tamil display string."""
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
    """Return True if the question is asking when the magazine started."""
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

    Uses _load_csv_safe so multi-author rows are not dropped.
    Min year result is the same either way, but keeps loading consistent.
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


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def handle_author_query(question: str, csv_path: str) -> Tuple[bool, str]:
    """Route a question to the appropriate CSV query handler.

    Returns (is_handled, response_text). is_handled is False when the
    question is not a CSV query (caller should proceed to vector search).
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
        return False, ""
    if query_type == "list_all_authors":
        return True, format_author_list(system.list_all_authors())
    if query_type == "author_topics":
        entity = system.extract_entity(question, "author_topics")
        logger.info(f"Extracted author: '{entity}'")
        if not entity:
            return True, "Warning: எழுத்தாளர் பெயரை தெளிவாக குறிப்பிடவும்"
        return True, format_author_topics(system.get_topics_by_author(entity))
    if query_type == "topic_author":
        entity = system.extract_entity(question, "topic_author")
        logger.info(f"Extracted topic: '{entity}'")
        if not entity:
            return True, "Warning: தலைப்பை தெளிவாக குறிப்பிடவும்"
        return True, format_topic_authors(system.get_author_by_topic(entity))

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
