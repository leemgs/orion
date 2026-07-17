#!/bin/bash
set -e

# Usage:
#   ./run.sh                → build main.pdf + supplementary.pdf (review copy:
#                             reference URLs are printed so reviewers can open
#                             each cited paper directly)
#   ./run.sh --submission   → same build, but the url={...} fields in
#                             reference-data.bib are stripped before bibtex,
#                             so no URLs appear in the reference section.
#                             (\url{} inside note/howpublished — i.e. web
#                             resources whose URL *is* the reference — is kept.)
#
# main.tex is the single entry file, built on the Springer Nature template
# for the Nature Machine Intelligence submission. The NMI reframing toggle
# (\ifNMIframing, default on) lives in main.tex; see README.md §4 / §8.

MAIN="main"
BIB="reference-data.bib"

if [[ "${1:-}" == "--submission" ]]; then
  echo "[submission mode] stripping url={...} fields from ${BIB}..."
  cp "${BIB}" "${BIB}.orig"
  trap 'mv -f "${BIB}.orig" "${BIB}"; echo "[submission mode] restored ${BIB}"' EXIT
  perl -i -ne 'print unless /^\s*url\s*=\s*\{[^}]*\},?\s*$/' "${BIB}"
fi

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
