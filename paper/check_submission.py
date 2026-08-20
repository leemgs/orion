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


def main() -> int:
    failures: list[str] = []
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for pattern, label in FORBIDDEN.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{path.relative_to(ROOT)}:{line}: {label}")
    if failures:
        print("Submission-source audit failed:", file=sys.stderr)
        print("\n".join(f"- {item}" for item in failures), file=sys.stderr)
        return 1
    print(f"Submission-source audit passed ({len(SOURCES)} files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
