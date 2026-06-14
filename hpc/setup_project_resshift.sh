#!/bin/bash
# setup_project.sh
# Run once manually on the login node. Do NOT sbatch.
# Usage: bash setup_project.sh

set -euo pipefail

BASE_DIR="$VSC_DATA/projects/resshift"

echo "Creating project structure at: $BASE_DIR"

mkdir -p "$BASE_DIR"/{code,logs,outputs}
mkdir -p "$BASE_DIR/outputs/experiments"

echo "Done. Next: bash clone_repo.sh"
