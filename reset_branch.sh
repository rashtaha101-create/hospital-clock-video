#!/usr/bin/env bash
# Replace the render branch with a single empty commit (drops old frames from the branch, keeps the repo small).
# Plumbing only: the working tree is never touched.
set -euo pipefail
BRANCH="$1"
git config --global user.name "github-actions[bot]"
git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"
TREE=$(git hash-object -t tree /dev/null)
COMMIT=$(git commit-tree "$TREE" -m "reset render branch")
git push -q --force origin "$COMMIT:refs/heads/$BRANCH"
echo "render branch '$BRANCH' reset"
