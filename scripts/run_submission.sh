#!/usr/bin/env bash
set -euo pipefail

# === Input Validation ===
if [[ $# -lt 1 ]]; then
    echo "Error: No username provided."
    echo "Usage: $0 <github_username>"
    exit 1
fi



USERNAME=$1
SUBMISSIONS_DIR="submissions"
RESULTS_DIR="results"
mkdir -p "$SUBMISSIONS_DIR" "$RESULTS_DIR"

# Repo and branch follow a fixed pattern
REPO="git@github.com:${USERNAME}/UTMIST-AI2.git"
BRANCH="main"

echo "Running submission for: $USERNAME"
echo "Repository: $REPO"
echo "Branch: $BRANCH"

# === Clone Submission ===
TARGET_DIR="$SUBMISSIONS_DIR/$USERNAME"
rm -rf "$TARGET_DIR"

git clone --branch "$BRANCH" "$REPO" "$TARGET_DIR"

# === Run Evaluation ===
echo "Evaluating submission for $USERNAME..."
python validate.py
echo "Finished evaluation for $USERNAME."
