#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idempotently give every slide in a Marp template a stable UUID permalink.

Each slide gets an invisible `<a class="slide-id" id="<uuid4>"></a>` as its first
content line. The id is a permanent, title- and position-independent anchor: the
exported deck's hash router resolves `#<uuid>` to the slide containing the anchor
(it falls back to `getElementById` for any non-numeric hash), so the link keeps
working even after the slide is retitled or reordered.

Slides that already carry a slide-id anchor are left untouched — that's what makes
existing permalinks stable. Run it on the human-edited `slides.tpl.md`, not the
generated `slides.md`; the pre-commit hook does this automatically.

Usage: assign_slide_ids.py <slides.tpl.md> [more.tpl.md ...]
Exit status is 0 whether or not anything changed; it prints which files it touched.
"""

import re
import sys
import uuid
from pathlib import Path

ANCHOR_MARKER = '<a class="slide-id"'


def split_frontmatter(text: str) -> tuple[str, str]:
    """Peel off a leading YAML frontmatter block (`--- ... ---`), if present."""
    m = re.match(r"---\n.*?\n---\n", text, re.DOTALL)
    return (text[: m.end()], text[m.end() :]) if m else ("", text)


def split_slides(body: str) -> list[str]:
    """Split on lines that are exactly `---`, skipping fenced code blocks."""
    chunks: list[str] = []
    cur: list[str] = []
    in_fence = False
    for line in body.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            cur.append(line)
        elif line == "---" and not in_fence:
            chunks.append("\n".join(cur))
            cur = []
        else:
            cur.append(line)
    chunks.append("\n".join(cur))
    return chunks


def ensure_anchor(chunk: str) -> tuple[str, bool]:
    """Prepend a fresh UUID anchor to a slide that lacks one; else leave it."""
    if ANCHOR_MARKER in chunk:
        return chunk, False
    body = chunk.lstrip("\n")
    lead = chunk[: len(chunk) - len(body)]  # preserve original leading blank lines
    anchor = f'<a class="slide-id" id="{uuid.uuid4()}"></a>\n\n'
    return lead + anchor + body, True


def process(path: Path) -> bool:
    """Anchor every anchorless slide in the template; return whether anything changed."""
    fm, body = split_frontmatter(path.read_text())
    out, changed = [], False
    for chunk in split_slides(body):
        new_chunk, did = ensure_anchor(chunk)
        out.append(new_chunk)
        changed = changed or did
    if changed:
        path.write_text(fm + "\n---\n".join(out))
    return changed


def main(argv: list[str]) -> int:
    """Process each template path given on the command line; print which files were touched."""
    if not argv:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2
    for arg in argv:
        path = Path(arg)
        if process(path):
            print(f"assigned slide ids: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
