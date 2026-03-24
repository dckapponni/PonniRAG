"""
Dataset loading and management for PonniRAG evaluation.

Supports two JSON formats:

  Format A — minimal (for pre-collected offline datasets):
    [
      {
        "id": "q1",
        "question": "...",
        "human_answer": "...",
        "llm_answer": "..."          <- optional; filled by live evaluation
      },
      ...
    ]

  Format B — annotated (richer, for curated datasets):
    [
      {
        "id": "q1",
        "question": "...",
        "human_answer": "...",
        "llm_answer": "...",         <- optional
        "category": "history",       <- optional free-text tag
        "notes": "..."               <- optional reviewer note
      },
      ...
    ]

CSV format is also accepted
(columns: id, question, human_answer[, llm_answer, category]).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single question/answer pair used during evaluation."""

    id: str
    question: str
    human_answer: str
    llm_answer: Optional[str] = None  # filled by live evaluation if absent
    category: Optional[str] = None
    notes: Optional[str] = None

    def is_ready(self) -> bool:
        """Return True when both reference and system answers are present."""
        return bool(self.human_answer) and bool(self.llm_answer)

    def to_dict(self) -> dict:
        """Convert the sample to a plain dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "human_answer": self.human_answer,
            "llm_answer": self.llm_answer,
            "category": self.category,
            "notes": self.notes,
        }


def load_dataset(path: str | Path) -> List[EvalSample]:
    """
    Load evaluation samples from a JSON or CSV file.

    Args:
        path: Absolute or relative path to the dataset file.
              Supported extensions: .json, .csv

    Returns:
        List of EvalSample objects.  Samples missing a human_answer are
        skipped with a warning; samples missing an llm_answer are kept
        (they will be populated by the live evaluation pipeline).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is unrecognised or required fields are absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    elif suffix == ".csv":
        return _load_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format '{suffix}'. Use .json or .csv.")


def _load_json(path: Path) -> List[EvalSample]:
    """Parse a JSON array of sample objects into EvalSample instances."""
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError(
            f"JSON dataset must be a top-level array, got {type(raw).__name__}."
        )

    samples: List[EvalSample] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict entry at index %d", idx)
            continue

        # Auto-generate id if absent
        sample_id = str(item.get("id", f"sample_{idx + 1}"))
        question = item.get("question", "").strip()
        human_answer = item.get("human_answer", "").strip()

        if not question:
            logger.warning("Skipping entry %s — missing 'question' field", sample_id)
            continue
        if not human_answer:
            logger.warning(
                "Skipping entry %s — missing 'human_answer' field", sample_id
            )
            continue

        samples.append(
            EvalSample(
                id=sample_id,
                question=question,
                human_answer=human_answer,
                llm_answer=(item.get("llm_answer") or "").strip() or None,
                category=item.get("category"),
                notes=item.get("notes"),
            )
        )

    logger.info("Loaded %d valid samples from %s", len(samples), path)
    return samples


def _load_csv(path: Path) -> List[EvalSample]:
    """Parse a CSV file with header row into EvalSample instances."""
    samples: List[EvalSample] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"question", "human_answer"}
        if reader.fieldnames:
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV is missing required columns: {missing}. "
                    f"Found: {reader.fieldnames}"
                )
        for idx, row in enumerate(reader):
            question = row.get("question", "").strip()
            human_answer = row.get("human_answer", "").strip()
            if not question or not human_answer:
                logger.warning(
                    "Skipping CSV row %d — empty question or human_answer", idx
                )
                continue
            sample_id = row.get("id", f"sample_{idx + 1}").strip()
            llm_raw = (row.get("llm_answer") or "").strip()
            samples.append(
                EvalSample(
                    id=sample_id,
                    question=question,
                    human_answer=human_answer,
                    llm_answer=llm_raw or None,
                    category=row.get("category"),
                    notes=row.get("notes"),
                )
            )

    logger.info("Loaded %d valid samples from CSV %s", len(samples), path)
    return samples


def save_dataset(samples: List[EvalSample], path: str | Path) -> None:
    """Persist a list of EvalSamples back to a JSON file.

    Useful for saving live-evaluated llm_answers alongside results.

    Args:
        samples: List of EvalSample objects to serialise.
        path: Destination file path (must have .json extension).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([s.to_dict() for s in samples], fh, ensure_ascii=False, indent=2)
    logger.info("Saved %d samples to %s", len(samples), path)
