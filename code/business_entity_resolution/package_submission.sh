#!/usr/bin/env bash
# Build <team_name>_submission.zip in the repo root:
#   output/{matching_results,candidate_pairs}.tsv, code/business_entity_resolution/, Documentation_template.md
# Usage: ./package_submission.sh <team_name>
set -euo pipefail
TEAM=${1:?usage: package_submission.sh <team_name>}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="${TEAM}_submission.zip"
rm -f "$OUT"
zip -r -q "$OUT" output/matching_results.tsv output/candidate_pairs.tsv Documentation_template.md \
    code/business_entity_resolution -x "*/__pycache__/*" "*.pyc"
ls -la "$OUT"
