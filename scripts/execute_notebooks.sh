#!/bin/bash
# Execute notebooks with outputs.
#
# Paper-figure notebooks (paper_figures/*.py) are rendered to BOTH .ipynb
# (with code cells, for code review) and code-free .html (figures and
# captions only, for sharing with co-authors), saved to paper_figures/rendered/.
#
# Exploratory notebooks (exploratory_notebooks/*.py) are rendered to .ipynb
# only, saved to exploratory_notebooks/ipynb/.
#
# Usage:
#   bash scripts/execute_notebooks.sh             # render all
#   bash scripts/execute_notebooks.sh 00          # match by prefix
#   bash scripts/execute_notebooks.sh 04          # exploratory notebook 04

set -e
REPO_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

render_one() {
    local pyfile="$1"
    local outdir="$2"
    local export_html="$3"
    local basename
    basename=$(basename "$pyfile" .py)
    local ipynb="${outdir}/${basename}.ipynb"
    local html="${outdir}/${basename}.html"

    mkdir -p "$outdir"
    echo "Executing $pyfile ..."
    python -m jupytext --to notebook --execute "$pyfile" -o "$ipynb"
    echo "  Done: $ipynb"

    if [ "$export_html" = "yes" ]; then
        echo "  Exporting $html (no code cells) ..."
        python -m jupyter nbconvert --to html --no-input \
            --output "${basename}.html" --output-dir "$outdir" "$ipynb"
    fi
}

# Collect candidate files
paper_files=$(ls "$REPO_ROOT"/paper_figures/*.py 2>/dev/null || true)
explore_files=$(ls "$REPO_ROOT"/exploratory_notebooks/*.py 2>/dev/null || true)

if [ -n "$1" ]; then
    # Filter by prefix match against the basename
    paper_match=$(echo "$paper_files" | grep -E "/0?${1}[^/]*\.py$" || true)
    explore_match=$(echo "$explore_files" | grep -E "/0?${1}[^/]*\.py$" || true)
    paper_files="$paper_match"
    explore_files="$explore_match"
    if [ -z "$paper_files" ] && [ -z "$explore_files" ]; then
        echo "No notebook matching '$1' found."
        exit 1
    fi
fi

for pyfile in $paper_files; do
    render_one "$pyfile" "$REPO_ROOT/paper_figures/rendered" yes
done

for pyfile in $explore_files; do
    render_one "$pyfile" "$REPO_ROOT/exploratory_notebooks/ipynb" no
done

echo ""
echo "To commit the updated outputs:"
echo "  git add paper_figures/rendered/ exploratory_notebooks/ipynb/ && git commit -m 'docs: update notebook outputs'"
