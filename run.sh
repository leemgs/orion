#!/bin/bash
set -e

MAIN="main"

echo "[1/4] pdflatex (first pass)..."
pdflatex -interaction=nonstopmode "${MAIN}.tex"

echo "[2/4] bibtex..."
bibtex "${MAIN}"

echo "[3/4] pdflatex (second pass)..."
pdflatex -interaction=nonstopmode "${MAIN}.tex"

echo "[4/4] pdflatex (final pass)..."
pdflatex -interaction=nonstopmode "${MAIN}.tex"

echo "Done: ${MAIN}.pdf"
