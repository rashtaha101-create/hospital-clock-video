#!/usr/bin/env bash
# Usage: commit_files.sh <branch> <src_dir> <dest_dir_in_repo> <message>
# Copies src_dir into dest_dir on the render branch and pushes, retrying when parallel jobs collide.
set -euo pipefail
BRANCH="$1"; SRC="$2"; DEST="$3"; MSG="$4"
git config --global user.name "github-actions[bot]"
git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"
git fetch -q origin "$BRANCH"
RW="$(mktemp -d)"
git worktree add -q --detach "$RW" FETCH_HEAD
mkdir -p "$RW/$DEST"
cp -r "$SRC"/. "$RW/$DEST"/
git -C "$RW" add -A -- "$DEST"
if git -C "$RW" diff --cached --quiet; then echo "nothing to commit"; exit 0; fi
git -C "$RW" commit -q -m "$MSG"
for i in $(seq 1 12); do
  if git -C "$RW" push -q origin "HEAD:refs/heads/$BRANCH"; then echo "pushed (attempt $i)"; exit 0; fi
  sleep $((RANDOM % 6 + 1))
  git -C "$RW" fetch -q origin "$BRANCH"
  git -C "$RW" rebase -q FETCH_HEAD
done
echo "push failed after retries" >&2; exit 1
