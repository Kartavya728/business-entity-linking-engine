#!/usr/bin/env bash
# Build submission variants from saved stage-2 probabilities (CPU only, ~2 min each).
# Usage: bash scripts/make_variants.sh <RUN> [REF]
#   RUN : a run tag with work/test/p2_<RUN>.parquet and work/models/decision_<RUN>.json
#   REF : reference run for France (default v6 = best French leaderboard result so far)
# Writes submissions/<RUN>_{all,hybrid,hybrid_em,frblend}/matching_results.tsv and validates them.
source "$(dirname "$0")/common.sh"
RUN=${1:?usage: make_variants.sh <RUN> [REF]}
REF=${2:-v6}
T="$BER_WORK/test"; M="$BER_WORK/models"; S="$BER_ROOT/submissions"
# France reference: a local run (work/test/p2_<REF>.parquet) or the copy shipped in the repo
# (reference/p2_<REF>_france.parquet, keyed by entity ids so it is valid on any machine)
if [ -f "$T/p2_$REF.parquet" ] && [ -f "$M/decision_$REF.json" ]; then
  REF_P2="p2_$REF.parquet"; REF_CFG="decision_$REF.json"
elif [ -f "$PKG/reference/p2_${REF}_france.parquet" ]; then
  REF_P2="$PKG/reference/p2_${REF}_france.parquet"; REF_CFG="$PKG/reference/decision_$REF.json"
else
  REF_P2=""; REF_CFG=""
fi
mk() {  # mk <name> <combine args...>
  local name=$1; shift
  ber combine --out "$S/$name" "$@" > "$LOG/variant_$name.log" 2>&1
  echo "  $name: $(grep -h 'France:' "$LOG/variant_$name.log" | tail -1 | cut -c1-110) | $(validate "$S/$name/matching_results.tsv")"
}
step "variants for $RUN (France reference: $REF)"
mk "${RUN}_all" --default "p2_$RUN.parquet" --default-cfg "decision_$RUN.json" \
   --country France --src "p2_$RUN.parquet" --src-cfg "decision_$RUN.json"
if [ -n "$REF_P2" ]; then
  # US/India from RUN, France from REF: separates the two effects on the leaderboard
  mk "${RUN}_hybrid" --default "p2_$RUN.parquet" --default-cfg "decision_$RUN.json" \
     --country France --src "$REF_P2" --src-cfg "$REF_CFG"
  # + label-shift correction for France (EM prior re-estimation, Saerens et al. 2002)
  mk "${RUN}_hybrid_em" --default "p2_$RUN.parquet" --default-cfg "decision_$RUN.json" \
     --country France --src "$REF_P2" --src-cfg "$REF_CFG" --em
  # France = average of the REF and RUN French probabilities
  mk "${RUN}_frblend" --default "p2_$RUN.parquet" --default-cfg "decision_$RUN.json" \
     --country France --src "$REF_P2" --src2 "p2_$RUN.parquet" --w2 0.5 --src-cfg "$REF_CFG"
else
  echo "  (no France reference $REF - hybrid variants skipped)"
fi
