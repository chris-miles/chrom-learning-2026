#!/bin/bash
# Execute all notebooks and save .ipynb with outputs to notebooks/ipynb/.
# Run manually when you want fresh rendered notebooks on GitHub.
# Usage: bash scripts/execute_notebooks.sh [notebook_number]
#   e.g.: bash scripts/execute_notebooks.sh 04

set -e
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/notebooks"

# Ensure chromlearn is importable regardless of kernel cwd
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [ -n "$1" ]; then
    files=$(ls 0${1}*.py ${1}*.py 2>/dev/null | head -1)
    if [ -z "$files" ]; then
        echo "No notebook matching '$1' found."
        exit 1
    fi
else
    files=$(ls *.py)
fi

mkdir -p ipynb

for pyfile in $files; do
    basename=$(basename "$pyfile" .py)
    ipynb="ipynb/${basename}.ipynb"
    echo "Executing $basename ..."
    python -m jupytext --to notebook --execute "$pyfile" -o "$ipynb"
    echo "  Done: $ipynb"
done

echo ""
echo "To commit the updated notebooks:"
echo "  git add notebooks/ipynb/*.ipynb && git commit -m 'docs: update notebook outputs'"
