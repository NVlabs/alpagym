#!/usr/bin/env bash
#
# Build a deck locally to PDF/HTML for quick iteration and manual review.
# Regenerates slides.md from its template, then renders with the SAME
# marpteam/marp-cli Docker image the CI uses, so local output matches CI.
#
#   Usage: tools/build.sh <deck> [formats]
#     deck     folder under decks/ (e.g. "onboarding")
#     formats  comma list of pdf,html   (default: pdf,html)
#
# Output → decks/out/<deck>.<fmt>  (gitignored). Requires local Docker.
set -euo pipefail
cd "$(dirname "$0")/.."          # decks/ root

DECK="${1:?Usage: $0 <deck> [pdf,html]}"
FORMATS="${2:-pdf,html}"
if [ ! -f "$DECK/slides.md" ]; then
    echo "Error: $DECK/slides.md not found. Available decks:" >&2
    for d in */slides.md; do [ -e "$d" ] && echo "  $(dirname "$d")" >&2; done
    exit 1
fi

python3 tools/build_deck.py "$DECK"
mkdir -p out
chmod 777 out                    # marp-cli container runs as uid 1000; let it write here

IMAGE="marpteam/marp-cli:latest"
IFS=',' read -ra FMT_LIST <<< "$FORMATS"
for fmt in "${FMT_LIST[@]}"; do
    fmt="$(echo "$fmt" | tr '[:upper:]' '[:lower:]' | xargs)"
    case "$fmt" in
        pdf)  flag="--pdf" ;;
        html) flag="" ;;
        *) echo "Unknown format: $fmt (use pdf or html)" >&2; exit 1 ;;
    esac
    echo "Exporting $DECK → out/$DECK.$fmt ..."
    docker run --rm --init \
        -v "$PWD":/home/marp/app \
        -e LANG="${LANG:-en_US.UTF-8}" \
        "$IMAGE" \
        "$DECK/slides.md" \
        $flag --html --allow-local-files --theme-set themes/ \
        -o "out/$DECK.$fmt"
    echo "  ✓ out/$DECK.$fmt"
done
echo "Done."
