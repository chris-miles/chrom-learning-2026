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

render_dir() {
    # Render every .py in $1 into $2; if $3 == yes, also export code-free
    # HTML alongside the .ipynb.  Filter by prefix match in $4 (optional).
    local srcdir="$1"
    local outdir="$2"
    local export_html="$3"
    local prefix="$4"

    mkdir -p "$outdir"
    cd "$srcdir"

    local pyfile
    for pyfile in *.py; do
        [ -e "$pyfile" ] || continue
        local basename="${pyfile%.py}"
        if [ -n "$prefix" ]; then
            case "$basename" in
                ${prefix}*|0${prefix}*) ;;
                *) continue ;;
            esac
        fi
        local ipynb="${outdir}/${basename}.ipynb"
        local html="${outdir}/${basename}.html"

        echo "Executing $srcdir/$pyfile ..."
        python -m jupytext --to notebook --execute "$pyfile" -o "$ipynb"
        echo "  Done: $ipynb"

        if [ "$export_html" = "yes" ]; then
            echo "  Exporting $html (no code cells) ..."
            python -m jupyter nbconvert --to html --no-input \
                --output "${basename}.html" --output-dir "$outdir" "$ipynb"
        fi
    done

    cd "$REPO_ROOT"
}

prefix_arg="${1:-}"
render_dir "$REPO_ROOT/paper_figures" "$REPO_ROOT/paper_figures/rendered" yes "$prefix_arg"
render_dir "$REPO_ROOT/exploratory_notebooks" "$REPO_ROOT/exploratory_notebooks/ipynb" no "$prefix_arg"

echo ""
echo "To commit the updated outputs:"
echo "  git add paper_figures/rendered/ exploratory_notebooks/ipynb/ && git commit -m 'docs: update notebook outputs'"
