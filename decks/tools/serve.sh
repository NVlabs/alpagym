#!/usr/bin/env bash
#
# Host the decks locally in a browser via marp-cli's server mode. Renders on the
# fly and serves the figure <iframe>s over HTTP — a file:// open of the built
# HTML can't reach them. Open the printed URL and pick a deck.
#
#   Usage: tools/serve.sh [port]      (default: 8095)
#
# Only the HTTP port is mapped, so this runs fine alongside another marp/slides
# server. Browser auto-reload uses marp's socket port 37717; to get it, free that
# port (stop the other server) and add `-p 37717:37717` below. Otherwise just
# refresh after rebuilding.
#
# Iterating on a deck: edit <deck>/slides.tpl.md, rerun tools/build_deck.py
# <deck>, then refresh the browser. Stop the server with Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")/.."          # decks/ root
PORT="${1:-8095}"

echo "Hosting decks at http://localhost:${PORT}/  (Ctrl-C to stop)"
exec docker run --rm --init \
    -v "$PWD":/home/marp/app \
    -e LANG="${LANG:-en_US.UTF-8}" \
    -p "${PORT}:8080" \
    marpteam/marp-cli:latest \
    --server . --html --theme-set themes/
