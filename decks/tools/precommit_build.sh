#!/usr/bin/env bash
# pre-commit entry: keep every deck's committed slides.md in sync with its
# template. For each deck, assign permalinks to new slides, then regenerate
# slides.md. pre-commit reports the modified files so they get re-staged.
set -euo pipefail
cd "$(dirname "$0")/.."          # decks/ root

for tpl in */slides.tpl.md; do
    [ -e "$tpl" ] || continue
    deck="$(dirname "$tpl")"
    python3 tools/assign_slide_ids.py "$tpl" >/dev/null
    python3 tools/build_deck.py "$deck" >/dev/null
done
