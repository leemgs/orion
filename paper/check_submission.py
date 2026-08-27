#!/usr/bin/env python3
"""Fail when submission sources contain known draft-only claims or markers."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ROOT / "main.tex",
    ROOT / "supplementary.tex",
    *(ROOT / "section").glob("*.tex"),
]

FORBIDDEN = {
    r"REVISION NOTE": "internal revision note",
    r"\bplaceholder\b": "placeholder text",
    r"\bMUST be recomputed\b": "unfinished measurement note",
    r"93\.4\\%": "unsupported classifier result",
    r"theta_B\s*(?:=|\\approx)\s*0\.4": "superseded theta_B value",
    r"\\theta_B\s*(?:=|\\approx)\s*0\.4": "superseded theta_B value",
    r"all five platforms": "unsupported five-platform claim",
    r"across five hardware": "unsupported five-platform claim",
    r"upon acceptance": "deferred data promise",
    r"reasonable request": "data-on-request wording",
    r"phase-like": "unsupported phase-transition framing",
    r"strategy rankings? invert": "unsupported strategy-inversion claim",
}


def abstract_word_count(text: str) -> int:
    """Count rendered abstract words, expanding the result macros it uses."""
    match = re.search(r"\\abstract\{%(.*?)\n\}", text, flags=re.DOTALL)
    if match is None:
        return 0
    abstract = match.group(1)
    expansions = {
        r"\NumAccel{}": "three",
        r"\GpuDevice{}": "Tesla T4",
        r"\AccelCapRatioMin": "1.23",
        r"\AccelCapRatioMax": "2.83",
        r"\times": "times",
    }
    for macro, rendered in expansions.items():
        abstract = abstract.replace(macro, rendered)
    abstract = re.sub(r"\\[A-Za-z]+", " ", abstract)
    abstract = re.sub(r"[$\\{}~%_=]", " ", abstract)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract))


def main() -> int:
    failures: list[str] = []
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for pattern, label in FORBIDDEN.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{path.relative_to(ROOT)}:{line}: {label}")
    main_text = (ROOT / "section" / "006_abstract_nature.tex").read_text(
        encoding="utf-8"
    )
    words = abstract_word_count(main_text)
    if words > 150:
        failures.append(
            "section/006_abstract_nature.tex:1: "
            f"abstract has {words} words (Nature Communications limit: 150)"
        )
    if failures:
        print("Submission-source audit failed:", file=sys.stderr)
        print("\n".join(f"- {item}" for item in failures), file=sys.stderr)
        return 1
    print(f"Submission-source audit passed ({len(SOURCES)} files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
