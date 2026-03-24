#!/bin/bash
# Execute all notebooks and save .ipynb with outputs to notebooks/ipynb/.
# Run manually when you want fresh rendered notebooks on GitHub.
# Usage: bash scripts/execute_notebooks.sh [notebook_number]
#   e.g.: bash scripts/execute_notebooks.sh 04

set -e
cd "$(git rev-parse --show-toplevel)"

if [ -n "$1" ]; then
    # Run a single notebook by number
    pattern="notebooks/0${1}*.py notebooks/${1}*.py"
    files=$(ls $pattern 2>/dev/null | head -1)
    if [ -z "$files" ]; then
        echo "No notebook matching '$1' found."
        exit 1
    fi
else
    files=$(ls notebooks/*.py)
fi

mkdir -p notebooks/ipynb

for pyfile in $files; do
    basename=$(basename "$pyfile" .py)
    ipynb="notebooks/ipynb/${basename}.ipynb"
    echo "Executing $pyfile -> $ipynb ..."
    python -m jupytext --to notebook --execute "$pyfile" -o "$ipynb"
    echo "  Done."
done

echo ""
echo "To commit the updated notebooks:"
echo "  git add notebooks/ipynb/*.ipynb && git commit -m 'docs: update notebook outputs'"
