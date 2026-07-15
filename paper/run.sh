#!/bin/bash
set -e

# Usage:
#   ./run.sh   → build main.pdf (sn-jnl / Springer Nature template)
#
# main.tex is the single entry file, built on the Springer Nature template
# for the Nature Machine Intelligence submission. The NMI reframing toggle
# (\ifNMIframing, default on) lives in main.tex; see README.md §4 / §8.

MAIN="main"

echo "[1/4] pdflatex (first pass) — ${MAIN}..."
pdflatex -interaction=nonstopmode "${MAIN}.tex"

echo "[2/4] bibtex..."
bibtex "${MAIN}"

echo "[3/4] pdflatex (second pass)..."
pdflatex -interaction=nonstopmode "${MAIN}.tex"

echo "[4/4] pdflatex (final pass)..."
pdflatex -interaction=nonstopmode "${MAIN}.tex"

echo "Done: ${MAIN}.pdf"

# Supplementary Information (separate NMI submission file)
SUPP="supplementary"
echo "[SI] building ${SUPP}.pdf..."
pdflatex -interaction=nonstopmode "${SUPP}.tex"
bibtex "${SUPP}" || true
pdflatex -interaction=nonstopmode "${SUPP}.tex"
pdflatex -interaction=nonstopmode "${SUPP}.tex"

echo "Done: ${SUPP}.pdf"
