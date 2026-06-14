#!/bin/bash
# clone_repo.sh
# Run once manually on the login node. Do NOT sbatch.
# Usage: bash clone_repo.sh

set -euo pipefail

BASE_DIR="$VSC_DATA/projects/resshift"
REPO_DIR="$BASE_DIR/code/ResShift"

echo "Cloning ResShift into: $REPO_DIR"

if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo already exists — pulling latest changes."
    git -C "$REPO_DIR" pull
else
    cd "$BASE_DIR/code"
    git clone https://github.com/InViLabVirtualStainingBenchmark/ResShift.git
fi

echo "Done."
