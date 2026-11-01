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
    # Reviewers open the linked repository: its front pages must not claim more
    # than the manuscript does.
    ROOT / "README.md",
    ROOT.parent / "README.md",
    ROOT.parent / "code" / "README.md",
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
    r"overlap (?:boundary|crossover).{0,30}resolved": (
        "tautological overlap-boundary validation claim"
    ),
    r"\\rho W\\max\(0,1-R_C\)": "residency-only service-time lower bound",
}


def _rendered_words(tex: str) -> int:
    """Count rendered words: drop citations/labels, expand result macros."""
    expansions = {
        r"\NumAccel{}": "three",
        r"\GpuDevice{}": "Tesla T4",
        r"\AccelCapRatioMin": "1.23",
        r"\AccelCapRatioMax": "2.83",
        r"\times": "times",
    }
    tex = re.sub(r"(?m)%.*$", "", tex)
    tex = re.sub(r"\\(cite|label|ref|eqref|input|url|path|setstretch)\{[^}]*\}", " ", tex)
    for macro, rendered in expansions.items():
        tex = tex.replace(macro, rendered)
    tex = re.sub(r"\\[A-Za-z]+", " ", tex)
    tex = re.sub(r"[$\\{}~%_=]", " ", tex)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", tex))


def abstract_word_count(text: str) -> int:
    """Rendered words of the summary paragraph (citations excluded)."""
    match = re.search(r"\\abstract\{%(.*?)\n\}", text, flags=re.DOTALL)
    return _rendered_words(match.group(1)) if match else 0


def body_word_count() -> int:
    """Main text (no Methods): everything except floats and equations."""
    total = 0
    for name in ("010_introduction", "025_results_ncs", "060_discussion"):
        tex = (ROOT / "section" / f"{name}.tex").read_text(encoding="utf-8")
        tex = re.sub(r"(?s)\\begin\{(table|figure|equation)\}.*?\\end\{\1\}", " ", tex)
        total += _rendered_words(tex)
    return total


# Nature formatting guide (Articles).
TITLE_MAX_CHARS = 75
SUMMARY_MAX_WORDS = 200
BODY_TARGET_WORDS = 2500      # typical for 4 modest display items (6 pages)
BODY_MAX_WORDS = 3000         # hard stop for this 4-display-item manuscript
SUBHEADING_MAX_CHARS = 40
MAIN_REFERENCE_MAX = 50


def main() -> int:
    failures: list[str] = []
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for pattern, label in FORBIDDEN.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{path.relative_to(ROOT.parent)}:{line}: {label}")
    summary = (ROOT / "section" / "006_abstract_nature.tex").read_text(encoding="utf-8")
    words = abstract_word_count(summary)
    if words > SUMMARY_MAX_WORDS:
        failures.append(f"section/006_abstract_nature.tex:1: summary paragraph has "
                        f"{words} words (Nature guideline: {SUMMARY_MAX_WORDS})")
    if "\\cite" not in summary:
        failures.append("section/006_abstract_nature.tex:1: Nature summary "
                        "paragraph must be referenced")

    title_tex = (ROOT / "section" / "001_title.tex").read_text(encoding="utf-8")
    title = re.search(r"\\title\{(.*?)\}\s*$", title_tex, flags=re.M | re.S)
    if title is None or len(title.group(1)) > TITLE_MAX_CHARS:
        n = len(title.group(1)) if title else -1
        failures.append(f"section/001_title.tex:1: title has {n} characters "
                        f"(Nature limit: {TITLE_MAX_CHARS})")

    body = body_word_count()
    if body > BODY_MAX_WORDS:
        failures.append(f"main text has {body} words (Nature: ~{BODY_TARGET_WORDS} "
                        f"for 4 display items; hard stop {BODY_MAX_WORDS})")

    for path in (ROOT / "section").glob("*.tex"):
        if path.name == "900_appendix.tex":  # Supplementary Information
            continue
        for m in re.finditer(r"\\(?:sub)?section\*?\{([^}]*)\}", path.read_text(encoding="utf-8")):
            if len(m.group(1)) > SUBHEADING_MAX_CHARS:
                failures.append(f"{path.relative_to(ROOT)}: subheading "
                                f"'{m.group(1)}' exceeds {SUBHEADING_MAX_CHARS} characters")

    bbl = ROOT / "main.bbl"
    if bbl.exists():
        n_refs = bbl.read_text(encoding="utf-8").count("\\bibitem")
        if n_refs > MAIN_REFERENCE_MAX:
            failures.append(f"main.bbl: {n_refs} references (Nature guideline: "
                            f"<= {MAIN_REFERENCE_MAX} in the main text)")
    print(f"title {len(title.group(1)) if title else '?'} chars | summary {words} words "
          f"| main text {body} words")
    if failures:
        print("Submission-source audit failed:", file=sys.stderr)
        print("\n".join(f"- {item}" for item in failures), file=sys.stderr)
        return 1
    print(f"Submission-source audit passed ({len(SOURCES)} files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
