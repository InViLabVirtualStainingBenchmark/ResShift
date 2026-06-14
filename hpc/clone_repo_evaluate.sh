#!/bin/bash
# clone_repo_evaluate.sh
# Run once manually on the login node. Do NOT sbatch.
# Usage: bash clone_repo_evaluate.sh

set -euo pipefail

BASE_DIR="$VSC_DATA"
REPO_DIR="$BASE_DIR/evaluate"

echo "Cloning Evaluate into: $REPO_DIR"

if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo already exists — pulling latest changes."
    git -C "$REPO_DIR" pull
else
    cd "$BASE_DIR"
    git clone https://github.com/InViLabVirtualStainingBenchmark/evaluate.git
fi

echo "Done."