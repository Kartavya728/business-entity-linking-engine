#!/usr/bin/env bash
# Final zip from a chosen matching file:
#   output/matching_results.tsv   <- the chosen file
#   output/candidate_pairs.tsv    <- stage-1 filtered blocking output (p1 >= 0.001, ~6.4 per S1,
#                                    99.83% train recall) + every predicted match
# Usage: bash scripts/package_final.sh <path/to/matching_results.tsv> <team_name>
source "$(dirname "$0")/common.sh"
SEL=${1:?usage: package_final.sh <matching_results.tsv> <team_name>}
TEAM=${2:?usage: package_final.sh <matching_results.tsv> <team_name>}
SEL="$(cd "$(dirname "$SEL")" && pwd)/$(basename "$SEL")"
cp "$SEL" "$BER_OUT/matching_results.tsv"
ber package_candidates --matching "$BER_OUT/matching_results.tsv" --out "$BER_OUT/candidate_pairs.tsv"
echo "validator: $(validate "$BER_OUT/matching_results.tsv")"
bash "$PKG/package_submission.sh" "$TEAM"
