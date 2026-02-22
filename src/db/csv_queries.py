"""
CSV query pipeline for author/topic queries in Tamil document processing.
Handles author name matching, query type detection, entity extraction,
and CSV data formatting for the Ponni magazine archive.
"""
import re
import logging
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from tamil_text import (
    FUZZY_THRESHOLD,
    TYPO_DISTANCE,
    fuzzy_match_score,
    _edit_distance_one,
    _strip_tamil_possessive_suffixes,
    _strip_tamil_possessive_suffix_word,
    _PatternBank,
    _RE_INITIALS_NAME,
    _RE_TAMIL_WORD,
)

logger = logging.getLogger(__name__)

_author_system_cache = {}
_author_system_lock = threading.Lock()


def normalize_author_name(name: str) -> str:
    if not name:
        return ""
    prefixes_to_remove = [
        r'மு\.,?\s*', r'டாக்டர்\.?\s*', r'திரு\.,?\s*',
        r'திருமதி\.?\s*', r'Dr\.?\s*', r'Mr\.?\s*', r'Mrs\.?\s*',
        r'கவிஞர்\.?\s*', r'பேராசிரியர்\.?\s*', r'அறிஞர்\.?\s*',
        r'புலவர்\.?\s*', r'கவியரசு\.?\s*', r'பாவேந்தர்\.?\s*',
    ]
    cleaned = name.strip()
    for prefix in prefixes_to_remove:
        cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def flexible_author_match(search_name: str, csv_name: str) -> bool:
    """
    Flexible matching between search query and CSV author name.

    Priority:
    1. Exact match after normalisation
    2. One is a substring of the other
    3. Known alias / special-case mapping
    4. Token-level subset match  (handles split/joined name variants)
    5. Fuzzy >= FUZZY_THRESHOLD (95%)
    6. Edit-distance <= TYPO_DISTANCE
    """
    search_normalized = normalize_author_name(search_name).lower().strip()
    csv_normalized    = normalize_author_name(csv_name).lower().strip()

    if not search_normalized or not csv_normalized:
        return False

    # Dot-normalized versions: collapse "நா. " -> "நா." to handle initials spacing
    def _dot_norm(s: str) -> str:
        return re.sub(r'\s*\.\s*', '.', s)

    search_dot = _dot_norm(search_normalized)
    csv_dot    = _dot_norm(csv_normalized)

    # 1. Exact (with and without dot normalization)
    if search_normalized == csv_normalized:
        return True
    if search_dot == csv_dot:
        logger.info(f"[DOT-EXACT] '{search_normalized}' ~ '{csv_normalized}' ✅")
        return True

    # 2. Substring (with and without dot normalization)
    if search_normalized in csv_normalized:
        return True
    if csv_normalized in search_normalized:
        return True
    if search_dot in csv_dot:
        logger.info(f"[DOT-SUBSTR] '{search_normalized}' ~ '{csv_normalized}' ✅")
        return True
    if csv_dot in search_dot:
        logger.info(f"[DOT-SUBSTR] '{search_normalized}' ~ '{csv_normalized}' ✅")
        return True

    # 3. Special-case aliases
    # CRITICAL BUG FIX: Do NOT use normalize_author_name() on variant strings.
    # Several variants like 'பாவேந்தர்' and 'கவியரசு' are listed as honorific
    # prefixes inside normalize_author_name() and get stripped to '' (empty string).
    # An empty string is always `in` any other string, causing EVERY csv author
    # to match — producing 1271-result responses for பாரதிதாசன் / கண்ணதாசன் queries.
    # Fix: use variant.lower() directly, and guard against empty variant strings.
    special_cases = {
        'கலைஞர்':     ['கருணாநிதி', 'மு.கருணாநிதி', 'மு. கருணாநிதி'],
        'கருணாநிதி':  ['கலைஞர்',    'மு.கருணாநிதி', 'மு. கருணாநிதி'],
        'அண்ணா':      ['அண்ணாதுரை', 'சி.என்.அண்ணாதுரை'],
        'அண்ணாதுரை':  ['அண்ணா',     'சி.என்.அண்ணாதுரை'],
        'பாரதிதாசன்': ['பாவேந்தர்', 'பாவேந்தர் பாரதிதாசன்'],
        'பாவேந்தர்':  ['பாரதிதாசன்', 'பாவேந்தர் பாரதிதாசன்'],
        'கண்ணதாசன்':  ['கவியரசு', 'கவியரசு கண்ணதாசன்'],
        'கவியரசு':    ['கண்ணதாசன்', 'கவியரசு கண்ணதாசன்'],
    }
    for key, variations in special_cases.items():
        if key in search_normalized:
            for variant in variations:
                variant_lower = variant.lower().strip()
                # Guard: skip empty variants — empty string matches everything
                if variant_lower and variant_lower in csv_normalized:
                    return True
        if key in csv_normalized:
            for variant in variations:
                variant_lower = variant.lower().strip()
                if variant_lower and variant_lower in search_normalized:
                    return True

    # 4. Token-level subset match (exact token equality, min 4 chars)
    # FIXED: use exact token equality (not substring containment) and min
    # length 4 to prevent short common Tamil substrings from causing false matches.
    search_tokens = [t for t in re.findall(r'[\u0B80-\u0BFF]+', search_normalized) if len(t) >= 4]
    csv_tokens    = [t for t in re.findall(r'[\u0B80-\u0BFF]+', csv_normalized)    if len(t) >= 4]
    if search_tokens and csv_tokens:
        if all(any(st == ct for ct in csv_tokens) for st in search_tokens):
            logger.info(f"[TOKEN-AUTHOR] '{search_normalized}' ~ '{csv_normalized}' ✅")
            return True

    # 5. Fuzzy >= 95%
    score = fuzzy_match_score(search_normalized, csv_normalized)
    if score >= FUZZY_THRESHOLD:
        logger.info(f"[FUZZY-AUTHOR] '{search_normalized}' ~ '{csv_normalized}' score={score:.2f} ✅")
        return True

    # 6. Edit-distance <= 1
    if _edit_distance_one(search_normalized, csv_normalized):
        logger.info(f"[TYPO-AUTHOR] '{search_normalized}' ≈ '{csv_normalized}' ✅")
        return True

    return False


def _find_closest_author(df: pd.DataFrame, search_name: str) -> tuple:
    """Return (best_author_name, best_score) from the author column."""
    best_name, best_score = "", 0.0
    search_norm = normalize_author_name(search_name).lower()
    for name in df['ஆசிரியர்'].dropna().unique():
        score = fuzzy_match_score(search_norm, normalize_author_name(str(name)).lower())
        if score > best_score:
            best_score = score
            best_name = str(name)
    return best_name, best_score


def _find_closest_title(df: pd.DataFrame, topic: str) -> tuple:
    """Return (best_title, best_score) from the title column."""
    best_title, best_score = "", 0.0
    topic_lower = topic.lower()
    for title in df['தலைப்பு'].dropna().unique():
        score = fuzzy_match_score(topic_lower, str(title).lower())
        if score > best_score:
            best_score = score
            best_title = str(title)
    return best_title, best_score


class EnhancedAuthorQuerySystem:
    """Robust author query system with improved CSV parsing and query detection."""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = None
        self._load_csv()

    def _load_csv(self):
        """Load CSV with multiple fallback strategies for robustness."""
        try:
            if not self.csv_path.exists():
                logger.error(f"CSV not found: {self.csv_path}")
                self.df = pd.DataFrame()
                return

            logger.info(f"Loading CSV: {self.csv_path}")

            strategies = [
                ("Standard", {"encoding": "utf-8", "skipinitialspace": True}),
                ("Skip bad lines", {"encoding": "utf-8", "on_bad_lines": "skip", "skipinitialspace": True}),
                ("Python engine", {"encoding": "utf-8", "engine": "python", "on_bad_lines": "skip"}),
                ("No quoting", {"encoding": "utf-8", "engine": "python", "quoting": 3}),
            ]

            loaded = False
            for strategy_name, kwargs in strategies:
                try:
                    self.df = pd.read_csv(self.csv_path, **kwargs)
                    logger.info(f"Loaded with: {strategy_name}")
                    loaded = True
                    break
                except Exception as e:
                    logger.warning(f"{strategy_name} failed: {str(e)[:60]}")
                    continue

            if not loaded or self.df is None or self.df.empty:
                logger.error("All loading strategies failed")
                self.df = pd.DataFrame()
                return

            self.df.columns = self.df.columns.str.strip()
            logger.info(f"Columns: {list(self.df.columns)}")
            logger.info(f"Rows: {len(self.df)}")

            self._fix_column_names()

            if 'ஆசிரியர்' in self.df.columns:
                self.df['ஆசிரியர்'] = self.df['ஆசிரியர்'].fillna('').astype(str).str.strip()
                before = len(self.df)
                self.df = self.df[self.df['ஆசிரியர்'] != '']
                logger.info(f"Authors: {before} to {len(self.df)} rows")
            else:
                logger.error("Missing 'ஆசிரியர்' column")
                return

            if 'தலைப்பு' in self.df.columns:
                self.df['தலைப்பு'] = self.df['தலைப்பு'].fillna('').astype(str).str.strip()

            if len(self.df) > 0:
                sample = self.df.iloc[0]
                logger.info(f"Sample: {sample.get('ஆசிரியர்', 'N/A')[:30]}")

        except Exception as e:
            logger.error(f"CSV error: {e}", exc_info=True)
            self.df = pd.DataFrame()

    def _fix_column_names(self):
        """Map alternate column names to standard Tamil names."""
        mappings = {
            'author': 'ஆசிரியர்',
            'Author': 'ஆசிரியர்',
            'title': 'தலைப்பு',
            'Title': 'தலைப்பு',
            'heading': 'தலைப்பு',
        }

        for old, new in mappings.items():
            if old in self.df.columns and new not in self.df.columns:
                self.df.rename(columns={old: new}, inplace=True)
                logger.info(f"Renamed '{old}' to '{new}'")

    # -------------------------------------------------------------------------
    def detect_query_type(self, question: str) -> str:
        """
        Comprehensive query-type detection.
        Returns: 'list_all_authors' | 'topic_author' | 'author_topics' | 'none'
        Evaluation order matters — broader patterns checked last.
        """
        q = question.lower().strip()

        # 1. List all authors
        if any(p in q for p in _PatternBank.LIST_ALL_AUTHORS):
            return 'list_all_authors'

        # 2. Topic → Author  (who wrote X?)
        if any(p in q for p in _PatternBank.TOPIC_AUTHOR):
            return 'topic_author'

        # 3. Author → Topics  (what did X write?)
        has_known_author  = any(a in q for a in _PatternBank.KNOWN_AUTHORS)
        has_initials_name = bool(_RE_INITIALS_NAME.search(q))
        has_author_action = any(p in q for p in _PatternBank.AUTHOR_ACTION)

        if (has_known_author or has_initials_name) and has_author_action:
            return 'author_topics'

        # Softer fallback: name present + writing-related keyword
        if (has_known_author or has_initials_name) and (
            'எழுதிய' in q or 'படைப்பு' in q or 'இயற்றிய' in q or 'படைத்த' in q
        ):
            return 'author_topics'

        # 4. Topic → Author fallback: "X பற்றிய படைப்புகள்/கட்டுரைகள்" style
        #    No author name present, but topic + content-type word => topic_author
        topic_content_patterns = [
            'பற்றிய படைப்புகள்', 'பற்றிய கட்டுரைகள்', 'பற்றிய கவிதைகள்',
            'பற்றிய கதைகள்', 'பற்றி படைப்புகள்', 'பற்றி கட்டுரைகள்',
            'தொடர்பான படைப்புகள்', 'தொடர்பான கட்டுரைகள்',
            'குறித்த படைப்புகள்', 'குறித்த கட்டுரைகள்',
            'சார்ந்த படைப்புகள்', 'சார்ந்த கட்டுரைகள்',
        ]
        if any(p in q for p in topic_content_patterns):
            return 'topic_author'

        return 'none'

    # -------------------------------------------------------------------------
    def extract_entity(self, question: str, query_type: str) -> str:
        """
        Robust entity extraction.
        - query_type='author_topics' -> returns the AUTHOR name
        - query_type='topic_author'  -> returns the TITLE / TOPIC string

        KEY FIX: Uses Tamil-safe suffix stripping via _strip_tamil_possessive_suffixes()
        instead of regex with \\b word boundaries, which do NOT work with Tamil Unicode.
        """
        q = question.strip()

        # Normalise Ponni variants
        for variation in _PatternBank.PONNI:
            if variation not in ('பொன்னி', 'ponni'):
                q = q.replace(variation, 'பொன்னி')
        q = re.sub(r'ponni\b', 'பொன்னி', q, flags=re.IGNORECASE)

        # ── AUTHOR TOPICS: extract AUTHOR NAME ───────────────────────────────
        if query_type == 'author_topics':

            # 0. Canonical map lookup BEFORE suffix stripping
            #    (some names like அப்பாத்துரை get wrongly stripped by யை rule)
            q_lower_raw = q.lower()
            for pattern, canonical in _PatternBank.AUTHOR_CANONICAL.items():
                if pattern in q_lower_raw:
                    logger.info(f"[CANONICAL-PRESUFFIX] matched '{pattern}' → '{canonical}'")
                    return canonical

            # 1. Strip Tamil possessive/genitive suffixes
            q_stripped = _strip_tamil_possessive_suffixes(q)

            # 2. Canonical map lookup again (post-strip, for நின்/ரின்/யின் suffixed names)
            q_lower = q_stripped.lower()
            for pattern, canonical in _PatternBank.AUTHOR_CANONICAL.items():
                if pattern in q_lower:
                    logger.info(f"[CANONICAL] matched '{pattern}' → '{canonical}'")
                    return canonical

            # ... rest of extraction unchanged

            # 2. Strip all noise phrases (longest first)
            noise = sorted(_PatternBank.AUTHOR_NOISE_PHRASES, key=len, reverse=True)
            q_clean = q_stripped
            for nw in noise:
                q_clean = re.sub(re.escape(nw), ' ', q_clean, flags=re.IGNORECASE)
            q_clean = re.sub(r'\s+', ' ', q_clean).strip()

            # 3. Try initials-style name:  e.g. "ஆ. வ. இராமநாதன்"
            m = _RE_INITIALS_NAME.search(q_clean)
            if m:
                candidate = m.group(1).strip()
                # Strip possessive suffixes from candidate too (Tamil-safe)
                candidate = _strip_tamil_possessive_suffixes(candidate).strip()
                bad = _PatternBank.AUTHOR_NOISE_TOKENS
                if not any(bw in candidate for bw in bad) and len(candidate) > 2:
                    return normalize_author_name(candidate)

            # 4. Collect remaining Tamil tokens (first 3) as name
            tokens = _RE_TAMIL_WORD.findall(q_clean)
            # Strip possessive suffixes from each token (Tamil-safe)
            tokens = [_strip_tamil_possessive_suffix_word(t) for t in tokens]
            tokens = [
                normalize_author_name(t) for t in tokens
                if len(t) > 2 and t not in _PatternBank.AUTHOR_NOISE_TOKENS
            ]
            if tokens:
                return ' '.join(tokens[:3])

            return ''

        # ── TOPIC AUTHOR: extract TITLE / TOPIC ──────────────────────────────
        elif query_type == 'topic_author':

            # Strip noise phrases (longest first).
            # Use DOUBLE SPACE as a replacement marker so we can later detect
            # whether a trailing யார்?/என்ன? was adjacent to removed noise
            # (= question suffix, strip it) or was part of the actual title
            # (= keep it, e.g. "தமிழர் யார்?").
            noise = sorted(_PatternBank.TOPIC_NOISE_PHRASES, key=len, reverse=True)
            q_title = q
            for np_phrase in noise:
                q_title = re.sub(re.escape(np_phrase), '  ', q_title, flags=re.IGNORECASE)

            # Strip title-type suffix words at END only
            for sw in _PatternBank.TITLE_SUFFIXES:
                q_title = re.sub(
                    r'\s*' + re.escape(sw) + r'\s*$', '', q_title.strip(),
                    flags=re.IGNORECASE
                )

            # Strip possessive suffixes that may be on the last word of a title.
            # FIXED: replaced \b with Tamil-safe (?=\s|$) via helper function.
            # e.g. "திறக்குமாவின்" -> "திறக்குமா"
            q_title = _strip_tamil_possessive_suffixes(q_title)

            # Strip Tamil locative suffixes from each word (e.g. section names):
            # "வளரும் இலக்கியத்தில்" -> "வளரும் இலக்கியம்"
            # "கவிதையில்" -> "கவிதை"
            # Rules: த்தில் -> ம், த்தின் -> ம், யில் -> '', இல் at word end -> ''
            _locative_rules = [
                (r'த்தில்(?=\s|$)', 'ம்'),
                (r'த்தின்(?=\s|$)', 'ம்'),
                (r'த்திற்கு(?=\s|$)', 'ம்'),
                (r'யில்(?=\s|$)', ''),
                (r'வில்(?=\s|$)', ''),
            ]
            for _pat, _repl in _locative_rules:
                q_title = re.sub(_pat, _repl, q_title)

            # Remove quotes but keep ? when part of title
            q_title = q_title.replace('"', '').replace("'", '')
            # Strip trailing "யார்?"/"என்ன?" ONLY when preceded by double-space,
            # meaning noise was removed just before it (question suffix, not title).
            # Preserves "தமிழர் யார்?" but strips "சமூக நீதி  [noise removed]  யார்?"
            q_title = re.sub(r'  +(யார்|என்ன|யாவை|எவர்)\??$', '', q_title)
            q_title = re.sub(r'\s+', ' ', q_title).strip()

            # Collect Tamil + latin words
            words = re.findall(r'[\u0B80-\u0BFF!?,।]+|[a-zA-Z]+', q_title)
            words = [
                w for w in words
                # Strip trailing punctuation before noise-token check,
                # BUT if the word ends with '?' (e.g. 'யார்?' as part of title),
                # keep it — only strip 'யார்' etc when standalone without '?'
                if len(w.rstrip('?!,')) > 1
                and not (w.rstrip('?!,').lower() in _PatternBank.TOPIC_NOISE_TOKENS
                         and not w.endswith('?'))
                and w not in {'?', '!', ',', '।'}
            ]

            result = ' '.join(words) if words else ''
            # Post-guard: if entire extracted topic is just a noise word like 'யார்?'
            # (meaning no real topic was found), return empty so caller can handle gracefully
            if result.strip().lower().rstrip('?!') in {'யார்', 'என்ன', 'யாவை', 'எவர்', ''}:
                return ''
            return result

        return ''

    # -------------------------------------------------------------------------
    def list_all_authors(self) -> Dict:
        """List all unique authors with article counts."""
        if self.df is None or self.df.empty:
            return {
                "type": "list_all_authors",
                "success": False,
                "message": "CSV தரவு கிடைக்கவில்லை",
                "authors": []
            }

        if 'ஆசிரியர்' not in self.df.columns:
            return {
                "type": "list_all_authors",
                "success": False,
                "message": "ஆசிரியர் column இல்லை",
                "authors": []
            }

        counts = self.df['ஆசிரியர்'].value_counts()

        authors = [
            {"name": author, "count": int(count)}
            for author, count in counts.items()
        ]

        return {
            "type": "list_all_authors",
            "success": True,
            "total_authors": len(authors),
            "total_articles": len(self.df),
            "authors": authors
        }

    # -------------------------------------------------------------------------
    def get_topics_by_author(self, author_name: str) -> Dict:
        """
        Get all articles/topics by a specific author.
        Uses flexible_author_match (fuzzy + edit-distance + token-level).
        """
        if self.df is None or self.df.empty:
            return {
                "type": "author_topics",
                "success": False,
                "message": "CSV தரவு கிடைக்கவில்லை",
                "articles": []
            }

        matches = self.df[
            self.df['ஆசிரியர்'].apply(
                lambda x: flexible_author_match(author_name, str(x))
            )
        ]

        if matches.empty:
            best_name, best_score = _find_closest_author(self.df, author_name)
            suggestion = ""
            if best_score >= 0.70:
                suggestion = (
                    f"நீங்கள் '{best_name}' என்பவரை குறிப்பிட்டீர்களா? "
                    f"(ஒற்றுமை: {best_score:.0%})"
                )
            return {
                "type": "author_topics",
                "success": False,
                "author": author_name,
                "message": f"'{author_name}' கண்டுபிடிக்க முடியவில்லை",
                "suggestion": suggestion,
                "articles": []
            }

        articles = []
        for _, row in matches.iterrows():
            article = {
                "title":  row.get('தலைப்பு', ''),
                "author": row.get('ஆசிரியர்', '')
            }
            for col in ['ஆண்டு', 'இதழ்', 'ச.எ.', 'வ.எ.']:
                if col in row and pd.notna(row[col]):
                    try:
                        article[col] = (
                            int(row[col]) if col in ['ஆண்டு', 'ச.எ.', 'வ.எ.']
                            else str(row[col])
                        )
                    except Exception:
                        article[col] = str(row[col])
            articles.append(article)

        return {
            "type": "author_topics",
            "success": True,
            "author": author_name,
            "matched_author": matches.iloc[0]['ஆசிரியர்'],
            "count": len(articles),
            "articles": articles
        }

# -------------------------------------------------------------------------
    def get_author_by_topic(self, topic: str) -> Dict:
        """
        Find authors who wrote about a specific topic.
        Matching stages:
        1.  Exact substring (after suffix cleaning)
        2.  All words present
        2b. Majority-word match (>= 60%, min 2 meaningful words, result set <= 8)
        3.  Longest Tamil word (>= 6 chars, result set <= 10)
        4.  Fuzzy title match >= 0.80  (looser than global FUZZY_THRESHOLD=0.95)
        5.  ALL long tokens (>= 5 chars) must match — AND logic
        """
        if self.df is None or self.df.empty:
            return {
                "type": "topic_author",
                "success": False,
                "message": "CSV தரவு கிடைக்கவில்லை",
                "articles": []
            }

        # Clean common suffixes
        topic_cleaned = topic
        for suffix in _PatternBank.TITLE_SUFFIXES:
            topic_cleaned = re.sub(
                r'\s*' + re.escape(suffix) + r'\s*$', '',
                topic_cleaned.strip(), flags=re.IGNORECASE
            )
        # Strip Tamil possessive suffixes
        topic_cleaned = _strip_tamil_possessive_suffixes(topic_cleaned)
        topic_cleaned = (topic_cleaned
                         .replace("'", "").replace('"', '')
                         .replace('?', '').strip())

        def _build_articles(matches):
            articles = []
            for _, row in matches.iterrows():
                article = {
                    "title":  row.get('தலைப்பு', ''),
                    "author": row.get('ஆசிரியர்', '')
                }
                for col in ['ஆண்டு', 'இதழ்', 'ச.எ.', 'வ.எ.']:
                    if col in row and pd.notna(row[col]):
                        try:
                            article[col] = (
                                int(row[col]) if col in ['ஆண்டு', 'ச.எ.', 'வ.எ.']
                                else str(row[col])
                            )
                        except Exception:
                            article[col] = str(row[col])
                articles.append(article)
            return articles

        # ── Stage 1: exact substring ─────────────────────────────────────────
        matches = self.df[
            self.df['தலைப்பு'].str.contains(
                topic_cleaned, case=False, na=False, regex=False
            )
        ]

        # ── Stage 2: all words present ───────────────────────────────────────
        if matches.empty and len(topic_cleaned.split()) > 1:
            words = topic_cleaned.split()
            matches = self.df[
                self.df['தலைப்பு'].apply(
                    lambda t: pd.notna(t) and all(w.lower() in str(t).lower() for w in words)
                )
            ]

        # ── Stage 2b: majority-word match ────────────────────────────────────
        # Only counts meaningful words (>= 3 chars) so short particles like
        # "இது", "தான்" don't drive false matches.
        # Result-set cap of 8 prevents broad hits from common words.
        if matches.empty and len(topic_cleaned.split()) > 2:
            meaningful_words = [w for w in topic_cleaned.split() if len(w) >= 3]
            if len(meaningful_words) >= 2:
                min_matches = max(2, int(len(meaningful_words) * 0.6))
                majority_rows = []
                for _, row in self.df.iterrows():
                    title = str(row.get('தலைப்பு', ''))
                    if not title:
                        continue
                    # Split title into tokens — prevents "இது" matching "இதுதான்"
                    title_tokens = set(re.findall(r'[\u0B80-\u0BFF]+|[a-zA-Z]+', title.lower()))
                    matched_count = sum(
                        1 for w in meaningful_words
                        if w.lower() in title_tokens  # exact token, not substring
                    )
                    if matched_count >= min_matches:
                        majority_rows.append(row)
                if majority_rows and len(majority_rows) <= 8:
                    matches = pd.DataFrame(majority_rows)
                    logger.info(
                        f"[MAJORITY-MATCH] {len(majority_rows)} rows "
                        f"(min_words={min_matches}, words={meaningful_words})"
                    )

        # ── Stage 3: longest Tamil word (>= 6 chars), result set <= 10 ───────
        # FIXED from min 4 chars: common words like "பிறந்த" (6 chars) were
        # matching hundreds of titles. Now requires truly specific Tamil words
        # AND caps the result set to confirm specificity.
        if matches.empty:
            words = topic_cleaned.split()
            if words:
                # Only consider words made of Tamil Unicode characters, min 6 chars
                tamil_words = [w for w in words if re.search(r'[\u0B80-\u0BFF]{6,}', w)]
                if tamil_words:
                    main_word = max(tamil_words, key=len)
                    candidate = self.df[
                        self.df['தலைப்பு'].str.contains(
                            main_word, case=False, na=False, regex=False
                        )
                    ]
                    if not candidate.empty and len(candidate) <= 10:
                        matches = candidate
                        logger.info(
                            f"[LONGEST-WORD] '{main_word}' → {len(candidate)} results"
                        )

        # ── Stage 4: fuzzy title match >= 0.80 ──────────────────────────────
        # Uses 0.80 (not the global FUZZY_THRESHOLD of 0.95) so that near-miss
        # titles like "காதல் பிறந்த கதை" (89%) match "காதில் பிறந்த கதை",
        # and "இது தான் புராணம்" (≈97%) matches "இது தான் பூராணம்".
        TOPIC_FUZZY_THRESHOLD = 0.80
        if matches.empty:
            fuzzy_rows = []
            for _, row in self.df.iterrows():
                title = str(row.get('தலைப்பு', ''))
                if not title:
                    continue
                score = fuzzy_match_score(topic_cleaned, title)
                if score >= TOPIC_FUZZY_THRESHOLD:
                    logger.info(
                        f"[FUZZY-TOPIC] '{topic_cleaned}' ~ '{title}' score={score:.2f} ✅"
                    )
                    fuzzy_rows.append((row, score))
            if fuzzy_rows:
                # Sort by score descending so best matches appear first
                fuzzy_rows.sort(key=lambda x: x[1], reverse=True)
                matches = pd.DataFrame([r for r, _ in fuzzy_rows])

        # ── Stage 5: ALL long tokens (>= 5 chars) must match — AND logic ─────
        # FIXED from OR logic: previously the first matching token won, letting
        # common words like "பிறந்த" match everything. Now ALL long tokens must
        # be present simultaneously.
        if matches.empty:
            tokens = [t for t in topic_cleaned.split() if len(t) >= 5]
            if len(tokens) >= 2:
                candidate = self.df[
                    self.df['தலைப்பு'].apply(
                        lambda t: pd.notna(t) and all(
                            tok.lower() in str(t).lower() for tok in tokens
                        )
                    )
                ]
                if not candidate.empty:
                    matches = candidate
                    logger.info(f"[MULTI-TOKEN-AND] tokens={tokens}")
            elif len(tokens) == 1 and len(tokens[0]) >= 7:
                # Single very long token — only accept if result set is small (specific)
                candidate = self.df[
                    self.df['தலைப்பு'].str.contains(
                        tokens[0], case=False, na=False, regex=False
                    )
                ]
                if not candidate.empty and len(candidate) <= 5:
                    matches = candidate
                    logger.info(
                        f"[SINGLE-LONG-TOKEN] '{tokens[0]}' → {len(candidate)} results"
                    )

        # ── Nothing found ────────────────────────────────────────────────────
        if matches.empty:
            best_title, best_score = _find_closest_title(self.df, topic_cleaned)
            suggestion = ""
            if best_score >= 0.70:
                suggestion = (
                    f"நீங்கள் '{best_title}' என்ற தலைப்பை குறிப்பிட்டீர்களா? "
                    f"(ஒற்றுமை: {best_score:.0%})"
                )
            return {
                "type": "topic_author",
                "success": False,
                "topic": topic,
                "cleaned_topic": topic_cleaned,
                "message": (
                    f"'{topic}' தலைப்பு கண்டுபிடிக்க முடியவில்லை. "
                    f"தேடிய சொல்: '{topic_cleaned}'"
                ),
                "suggestion": suggestion,
                "articles": [],
            }

        return {
            "type": "topic_author",
            "success": True,
            "topic": topic,
            "cleaned_topic": topic_cleaned,
            "count": len(matches),
            "articles": _build_articles(matches)
        }


def format_author_list(result: Dict) -> str:
    """Format list of all authors in simple numbered list format."""
    if not result['success']:
        return f"Error: {result['message']}"

    lines = [
        "பொன்னி இதழில் எழுதிய எழுத்தாளர்கள்",
        f"மொத்த எழுத்தாளர்கள்: {result['total_authors']} | மொத்த கட்டுரைகள்: {result['total_articles']}",
        ""
    ]

    sorted_authors = sorted(result['authors'], key=lambda x: x['count'], reverse=True)

    for idx, author in enumerate(sorted_authors, 1):
        lines.append(f"{idx}. {author['name']} ({author['count']} கட்டுரைகள்)")

    return '\n'.join(lines)


def format_author_topics(result: Dict) -> str:
    """Format topics by author in simple numbered list format."""
    if not result['success']:
        return f"Error: {result['message']}"

    lines = [
        f"எழுத்தாளர்: {result.get('matched_author', result['author'])}",
        f"மொத்த படைப்புகள்: {result['count']}",
        ""
    ]

    sorted_articles = sorted(
        result['articles'],
        key=lambda x: (x.get('ஆண்டு', 9999), str(x.get('இதழ்', '')))
    )

    for idx, article in enumerate(sorted_articles, 1):
        parts = [f"தலைப்பு: {article.get('title', '-')}"]
        parts.append(f"ஆசிரியர்: {article.get('author', '-')}")

        if 'ஆண்டு' in article:
            parts.append(f"ஆண்டு: {article['ஆண்டு']}")

        if 'இதழ்' in article:
            parts.append(f"இதழ்: {article['இதழ்']}")

        lines.append(f"{idx}. {' | '.join(parts)}")

    return '\n'.join(lines)


def format_topic_authors(result: Dict) -> str:
    """Format topic authors in simple numbered list format."""
    if not result['success']:
        msg = f"Error: {result['message']}"
        if 'suggestion' in result:
            msg += f"\n\nசிபாரிசு: {result['suggestion']}"
        return msg

    lines = [
        f"தலைப்பு: '{result['topic']}' பற்றிய கட்டுரைகள்",
    ]

    if result.get('cleaned_topic') and result['cleaned_topic'] != result['topic']:
        lines.append(f"(தேடிய சொல்: '{result['cleaned_topic']}')")

    lines.append(f"கண்டுபிடிக்கப்பட்டவை: {result['count']}")
    lines.append("")

    for idx, article in enumerate(result['articles'], 1):
        parts = [f"தலைப்பு: {article.get('title', '-')}"]
        parts.append(f"ஆசிரியர்: {article.get('author', '-')}")

        if 'ஆண்டு' in article:
            parts.append(f"ஆண்டு: {article['ஆண்டு']}")

        if 'இதழ்' in article:
            parts.append(f"இதழ்: {article['இதழ்']}")

        lines.append(f"{idx}. {' | '.join(parts)}")

    return '\n'.join(lines)


def detect_issue_count_query(question: str) -> bool:
    """Detect if user is asking about issue count."""
    q = question.lower()

    patterns = [
        'இதழ் எண்ணிக்கை',
        'எத்தனை இதழ்',
        'இதழ்கள் எத்தனை',
        'மொத்த இதழ்',
        'இதழ் count',
        'issue count',
        'how many issues',
        'number of issues',
        'total issues',
        'இதழ் பட்டியல்',
        'இதழ்களின் பட்டியல்'
    ]

    return any(pattern in q for pattern in patterns)


def get_issue_count(csv_path: str) -> Dict:
    """Get unique issue count from CSV with article statistics."""
    try:
        csv_path = Path(csv_path)

        if not csv_path.exists():
            return {
                "success": False,
                "message": "CSV கோப்பு கிடைக்கவில்லை",
                "count": 0,
                "issues": []
            }

        logger.info(f"Loading CSV for issue count: {csv_path}")

        df = None
        strategies = [
            {"encoding": "utf-8", "skipinitialspace": True},
            {"encoding": "utf-8", "on_bad_lines": "skip"},
            {"encoding": "utf-8", "engine": "python", "on_bad_lines": "skip"},
        ]

        for kwargs in strategies:
            try:
                df = pd.read_csv(csv_path, **kwargs)
                break
            except:
                continue

        if df is None or df.empty:
            return {
                "success": False,
                "message": "CSV தரவை படிக்க முடியவில்லை",
                "count": 0,
                "issues": []
            }

        df.columns = df.columns.str.strip()

        issue_col = None
        for col in ['இதழ்', 'issue', 'Issue', 'doc_issue']:
            if col in df.columns:
                issue_col = col
                break

        if not issue_col:
            return {
                "success": False,
                "message": "இதழ் column கிடைக்கவில்லை",
                "count": 0,
                "issues": []
            }

        df[issue_col] = df[issue_col].fillna('').astype(str).str.strip()
        df_filtered = df[df[issue_col] != '']

        unique_issues = df_filtered[issue_col].unique()
        issue_counts = df_filtered[issue_col].value_counts()

        try:
            sorted_issues = sorted(unique_issues, key=lambda x: int(x) if x.isdigit() else x)
        except:
            sorted_issues = sorted(unique_issues)

        issues_list = []
        for issue in sorted_issues:
            article_count = int(issue_counts[issue])
            issues_list.append({
                "issue_number": issue,
                "article_count": article_count
            })

        return {
            "success": True,
            "count": len(unique_issues),
            "total_articles": len(df_filtered),
            "issues": issues_list
        }

    except Exception as e:
        logger.error(f"Error getting issue count: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"பிழை: {str(e)}",
            "count": 0,
            "issues": []
        }


def format_issue_count(result: Dict) -> str:
    """Format issue count result in simple numbered list format."""
    if not result['success']:
        return f"Error: {result['message']}"

    lines = [
        "பொன்னி இதழ்கள் விவரம்",
        f"மொத்த இதழ்கள்: {result['count']}",
        f"மொத்த கட்டுரைகள்: {result['total_articles']}",
        ""
    ]

    issues = result['issues']

    if len(issues) <= 20:
        for idx, issue_info in enumerate(issues, 1):
            lines.append(f"{idx}. இதழ்: {issue_info['issue_number']} | கட்டுரைகள்: {issue_info['article_count']}")
    else:
        lines.append("முதல் 10 இதழ்கள்:")
        for idx, issue_info in enumerate(issues[:10], 1):
            lines.append(f"{idx}. இதழ்: {issue_info['issue_number']} | கட்டுரைகள்: {issue_info['article_count']}")

        lines.append("")
        lines.append(f"... (மேலும் {len(issues) - 20} இதழ்கள்)")
        lines.append("")

        lines.append("கடைசி 10 இதழ்கள்:")
        for idx, issue_info in enumerate(issues[-10:], len(issues) - 9):
            lines.append(f"{idx}. இதழ்: {issue_info['issue_number']} | கட்டுரைகள்: {issue_info['article_count']}")

    lines.append("")

    if result['issues']:
        article_counts = [i['article_count'] for i in result['issues']]
        lines.append("புள்ளிவிவரம்:")
        lines.append(f"  • சராசரி கட்டுரைகள் ஒரு இதழுக்கு: {sum(article_counts) / len(article_counts):.1f}")
        lines.append(f"  • குறைந்தபட்ச கட்டுரைகள்: {min(article_counts)}")
        lines.append(f"  • அதிகபட்ச கட்டுரைகள்: {max(article_counts)}")

    return '\n'.join(lines)


def detect_start_year_query(question: str) -> bool:
    q = question.lower()
    patterns = [
        "எந்த ஆண்டு தொடங்கியது",
        "முதல் ஆண்டு",
        "தொடங்கிய ஆண்டு",
        "எப்போது தொடங்கியது",
        "start year",
        "when did ponni start",
    ]
    return any(p in q for p in patterns)


def get_start_year(csv_path: str) -> str:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return "CSV தரவில் ஆண்டு தகவல் இல்லை."
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
    except Exception:
        return "CSV தரவில் ஆண்டு தகவல் இல்லை."
    if df.empty or 'ஆண்டு' not in df.columns:
        return "CSV தரவில் ஆண்டு தகவல் இல்லை."
    try:
        df['ஆண்டு'] = pd.to_numeric(df['ஆண்டு'], errors='coerce')
        min_year = int(df['ஆண்டு'].min())
        return (
            f"பொன்னி இதழ் {min_year} ஆம் ஆண்டு தொடங்கியது.\n\n"
        )
    except Exception as e:
        logger.error(f"Start year calculation error: {e}")
        return "ஆண்டு தகவலை கணக்கிட முடியவில்லை."


def handle_author_query(question: str, csv_path: str) -> Tuple[bool, str]:
    """
    Returns raw CSV data for author/topic queries.
    The caller passes this data to the LLM for summarization.
    """
    if detect_start_year_query(question):
        logger.info("Detected start year query")
        return True, get_start_year(csv_path)
    if detect_issue_count_query(question):
        logger.info("Detected issue count query")
        result = get_issue_count(csv_path)
        return True, format_issue_count(result)

    with _author_system_lock:
        if csv_path not in _author_system_cache:
            _author_system_cache[csv_path] = EnhancedAuthorQuerySystem(csv_path)
        system = _author_system_cache[csv_path]

    if system.df is None or system.df.empty:
        return False, ""

    query_type = system.detect_query_type(question)
    logger.info(f"Query type: {query_type}")

    if query_type == 'none':
        return False, ""

    if query_type == 'list_all_authors':
        result = system.list_all_authors()
        return True, format_author_list(result)

    elif query_type == 'author_topics':
        entity = system.extract_entity(question, 'author_topics')
        logger.info(f"Extracted author: '{entity}'")

        if not entity:
            return True, "Warning: எழுத்தாளர் பெயரை தெளிவாக குறிப்பிடவும்"

        result = system.get_topics_by_author(entity)
        return True, format_author_topics(result)

    elif query_type == 'topic_author':
        entity = system.extract_entity(question, 'topic_author')
        logger.info(f"Extracted topic: '{entity}'")

        if not entity:
            return True, "Warning: தலைப்பை தெளிவாக குறிப்பிடவும்"

        result = system.get_author_by_topic(entity)
        return True, format_topic_authors(result)

    return False, ""


def _csv_source(csv_data: str) -> List[Dict]:
    """Return a source entry with the raw CSV-retrieved data as evidence."""
    return [{
        "volume": "பொன்னி கட்டுரை தரவுத்தளம்",
        "heading": "Article Database",
        "doc_issue": "",
        "content": csv_data,
        "word_count": len(csv_data.split()),
        "chunks_merged": 0,
        "score": 1.0,
    }]


def _csv_data_suffix(csv_data: str) -> str:
    """Format the raw CSV data as a suffix to append after the LLM summary."""
    return f"\n\n---\n\n**தரவுத்தள தகவல்:**\n\n{csv_data}"


def _combine_csv_answer(llm_summary: str, csv_data: str) -> str:
    """Combine LLM gist with raw CSV data appended below.

    If the LLM summary is empty (failed), prepends a fallback header
    before the raw CSV data so the separator is always visible.
    """
    suffix = _csv_data_suffix(csv_data)
    if llm_summary and llm_summary.strip():
        return llm_summary.strip() + suffix
    # Fallback: no gist available, still show data with separator
    return "கட்டுரை தரவுத்தளத்திலிருந்து பெறப்பட்ட தகவல்கள்:" + suffix
