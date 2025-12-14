#!/usr/bin/env bash
# parse-vectorization.sh

set -euo pipefail

OUT="vectorization.json"
SHOW_ALL=false
VERSION=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --out=*) OUT="${1#*=}" ;;
        --all) SHOW_ALL=true ;;
        --version=*) VERSION="${1#*=}" ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Run make with DEBUG=1 and parse vectorization info"
            echo ""
            echo "Options:"
            echo "  --out=FILE     Output file (default: vectorization.json)"
            echo "  --all          Show all optimization messages"
            echo "  --version=STR  Filter by version (e.g., v0, v1)"
            echo "  --help         Show this help"
            exit 0
            ;;
        *) break ;;
    esac
    shift
done

# Build jq filter
JQ_FILTER='
  [inputs as $line |
    # Parse the line
    ($line | capture("^(?<file>[^:]+):(?<line>[0-9]+):(?<col>[0-9]+): (?<kind>[^:]+): (?<msg>.*)")) //
    # If no match, skip
    empty
  ]'

# Add filters
if [[ "$SHOW_ALL" == false ]]; then
    JQ_FILTER+='
    | map(select(
        (.msg | test("loop vectorized|couldn.t vectorize loop|not vectorized:|completely unrolled|SLP|basic block part vectorized|vectorized [0-9]+ loops in function"; "i"))
      ))'
fi

# Add version filter
if [[ -n "$VERSION" ]]; then
    JQ_FILTER+=" | map(select(.file | test(\"$VERSION\")))"
fi

# Add vectorization info
JQ_FILTER+='
  | map(. + {
      is_vectorized: (.kind == "optimized" and (.msg | test("loop vectorized"; "i"))),
      is_missed: (.kind == "missed" and (.msg | test("couldn.t vectorize loop|not vectorized:"; "i"))),
      is_unrolled: (.msg | test("completely unrolled"; "i")),
      is_slp: (.msg | test("SLP|basic block part vectorized"; "i")),
      vector_size: (if .msg | test("using [0-9]+ byte vectors") then (.msg | capture("using (?<size>[0-9]+) byte vectors") | .size | tonumber) else null end),
      reason: (if .kind == "missed" then .msg else null end)
    })'

# Run make and parse
echo "Running make with DEBUG=1..." >&2
make DEBUG=1 "$@" 2>&1 | jq -R -n "$JQ_FILTER" | tee "$OUT"

echo "Results saved to $OUT" >&2
