#!/usr/bin/env bash
# analyze-vectorization.sh - Analyze vectorization results

set -euo pipefail

INPUT="${1:-vectorization.json}"
VERSION="${2:-}"

echo "=== VECTORIZATION ANALYSIS ==="
echo "Input file: $INPUT"
echo "Version filter: ${VERSION:-none}"
echo ""

# Build jq filter
JQ_FILTER=""

# Add version filter if specified
if [[ -n "$VERSION" ]]; then
    JQ_FILTER="map(select(.file | test(\"$VERSION\"))) |"
fi

# Run analysis
echo "SUMMARY:"
echo "--------"
jq "
  $JQ_FILTER
  [
    {
      total_entries: length,
      vectorized_loops: map(select(.is_vectorized == true)) | length,
      missed_vectorizations: map(select(.is_missed == true)) | length,
      unrolled_loops: map(select(.is_unrolled == true)) | length,
      slp_vectorizations: map(select(.is_slp == true)) | length
    }
  ][0]
" "$INPUT"

echo ""
echo "VECTORIZED LOOPS:"
echo "-----------------"
jq "
  $JQ_FILTER
  map(select(.is_vectorized == true))
  | group_by(.file)
  | map({
      file: .[0].file,
      count: length,
      lines: map(.line) | unique,
      vector_sizes: map(.vector_size) | unique | map(select(. != null))
    })
" "$INPUT"

echo ""
echo "MISSED VECTORIZATIONS (with reasons):"
echo "-------------------------------------"
jq "
  $JQ_FILTER
  map(select(.is_missed == true and .reason != null))
  | group_by(.reason)
  | map({
      reason: .[0].reason,
      count: length,
      files: map(.file) | unique,
      lines: map(\"\(.file):\(.line)\") | unique
    })
" "$INPUT"

echo ""
echo "UNROLLED LOOPS:"
echo "---------------"
jq "
  $JQ_FILTER
  map(select(.is_unrolled == true))
  | map({
      file: .file,
      line: .line,
      message: .msg
    })
" "$INPUT"
