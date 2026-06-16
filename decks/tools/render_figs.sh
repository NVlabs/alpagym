#!/usr/bin/env bash
# Render a deck's SVG figures to PNG so layout can be visually verified.
# Usage: render_figs.sh <deck> [name]   (name without extension; default: all)
set -euo pipefail
DECK="${1:?usage: $0 <deck> [name]}"
DIR="$(cd "$(dirname "$0")/../$DECK/images" && pwd)"

render() {
  local svg="$DIR/$1"
  local png="${svg%.svg}.png"
  inkscape "$svg" --export-type=png --export-filename="$png" --export-background="#111111" --export-width=1160 2>&1 | grep -viE 'pango|URI|referenced' || true
  echo "rendered $(basename "$png")"
}

cd "$DIR"

if [ "${2:-all}" = "all" ]; then
  for f in *.svg; do [ -e "$f" ] && render "$f"; done
else
  render "$2.svg"
fi
