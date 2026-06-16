#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assemble slides.md from slides.tpl.md by inlining clickable SVG figures.

Each `<!--FIG:name-->` token in the template is replaced with the contents of
`images/name.svg`, wrapped in a centered container and sized to the slide. We
inline the SVG (rather than referencing it as an <img>) so the <a> hyperlinks
inside each figure stay clickable in the exported HTML deck.

Usage: build_deck.py <deck>   (folder name under the decks/ root, e.g. "onboarding")
       Reads <deck>/slides.tpl.md + <deck>/images/*.svg, writes <deck>/slides.md.
"""

import re
import sys
from pathlib import Path

decks_root = Path(__file__).resolve().parent.parent
if len(sys.argv) != 2:
    sys.exit(f"usage: {Path(__file__).name} <deck>  (folder under {decks_root})")
here = decks_root / sys.argv[1]
if not (here / "slides.tpl.md").is_file():
    sys.exit(f"no slides.tpl.md in deck '{sys.argv[1]}' ({here})")
tpl = (here / "slides.tpl.md").read_text()


def inline(match: re.Match) -> str:
    """Default variant: center the figure at standard slide size."""
    name = match.group(1)
    svg = (here / "images" / f"{name}.svg").read_text().strip()
    # Collapse blank lines: a blank line would terminate markdown-it's raw HTML
    # block early and fragment the figure (breaking its sizing/links).
    svg = re.sub(r"\n\s*\n", "\n", svg)
    # Drop the XML declaration if present and make the root <svg> responsive.
    # Explicit pixel width/height (not just CSS max-width) so Marp sizes every
    # figure identically regardless of the SVG's internal <style> block.
    svg = svg.replace(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600"',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" '
        'width="980" height="507" style="max-width:100%"',
    )
    return f'<div style="display:flex;justify-content:center;margin-top:0.4rem">\n{svg}\n</div>'


def inline_full(match: re.Match) -> str:
    """Full-bleed variant: fill the whole slide (use on a padding-0 section)."""
    name = match.group(1)
    svg = (here / "images" / f"{name}.svg").read_text().strip()
    svg = re.sub(r"\n\s*\n", "\n", svg)
    svg = svg.replace(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600"',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" '
        'width="100%" height="100%" preserveAspectRatio="xMidYMid meet" style="display:block"',
    )
    return (
        '<div style="position:absolute;inset:0;display:flex;align-items:center;'
        f'justify-content:center;padding:12px;box-sizing:border-box">\n{svg}\n</div>'
    )


def inline_sm(match: re.Match) -> str:
    """Smaller variant: for dense slides that also carry intro/caption text, so
    the figure fits without overflowing the slide. Same blank-line handling.
    """
    name = match.group(1)
    svg = (here / "images" / f"{name}.svg").read_text().strip()
    svg = re.sub(r"\n\s*\n", "\n", svg)
    svg = svg.replace(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600"',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" '
        'width="860" height="445" style="max-width:100%"',
    )
    return f'<div style="display:flex;justify-content:center;margin-top:0.2rem">\n{svg}\n</div>'


# Clickable permalink. Each slide carries an invisible `<a class="slide-id">`
# (added to the template by scripts/assign_slide_ids.py). This block adds a small
# hotspot over the page-number corner: clicking it puts that slide's stable
# `#<uuid>` link in the address bar *and* the clipboard, so the URL you share
# survives retitling/reordering — unlike the volatile `#<slide-number>`.
#
# CSS is injected into <head> from the script: a <style> written in the markdown
# lands inside the slide's SVG <foreignObject>, where its stylesheet is SVG-scoped
# and never reaches the <body>. Scripts, however, do execute from there.
#
# The hotspot is placed *inside* each <section> (slide space), not fixed to the
# viewport — so it scales and letterboxes with the slide and stays on the number.
# Its offsets (left: 3rem, bottom: 2rem) mirror `section::after` in themes/nvidia.css.
PERMALINK = """

<script>
(function () {
  var CSS = [
    ".slide-permalink{position:absolute;left:2.4rem;bottom:1.6rem;width:3.4rem;",
    "height:1.9rem;z-index:50;cursor:pointer;text-decoration:none}",
    ".slide-permalink::after{content:'\\\\1F517';position:absolute;left:3rem;",
    "bottom:.1rem;font-size:.8rem;opacity:0;transition:opacity .15s}",
    ".slide-permalink:hover::after{opacity:.6}",
    "#slide-permalink-toast{position:fixed;left:50%;top:1.4rem;",
    "transform:translateX(-50%);background:rgba(18,18,18,.94);color:#fff;",
    "border:1px solid #76b900;border-radius:6px;padding:.4rem .85rem;",
    "z-index:10000;font:500 .8rem/1 system-ui,sans-serif;opacity:0;",
    "transition:opacity .2s;pointer-events:none}",
    "#slide-permalink-toast.show{opacity:1}"
  ].join("");
  function permalink(uuid) {
    return location.origin + location.pathname + location.search + "#" + uuid;
  }
  function toast(msg) {
    var t = document.getElementById("slide-permalink-toast");
    if (!t) {
      t = document.createElement("div");
      t.id = "slide-permalink-toast";
      document.body.appendChild(t);
    }
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(t._h);
    t._h = setTimeout(function () { t.classList.remove("show"); }, 1600);
  }
  function copy(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(
        function () { return true; }, function () { return false; });
    }
    try {
      var ta = document.createElement("textarea");
      ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select();
      var ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return Promise.resolve(ok);
    } catch (e) { return Promise.resolve(false); }
  }
  function onClick(e, uuid) {
    e.preventDefault();
    e.stopPropagation();
    var url = permalink(uuid);
    // replaceState (not location.hash=) so bespoke doesn't re-navigate. This
    // alone puts the permalink in the address bar, so feedback shouldn't wait on
    // the clipboard (which can hang without permission).
    history.replaceState(null, "", url);
    toast("Permalink copied");
    copy(url);
  }
  function init() {
    var style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);
    var sections = document.querySelectorAll("section");
    for (var i = 0; i < sections.length; i += 1) {
      var sec = sections[i];
      var anchor = sec.querySelector("a.slide-id");
      if (!anchor) continue;
      var hot = document.createElement("a");
      hot.className = "slide-permalink";
      hot.href = "#" + anchor.id;  // right-click "Copy link" + no-JS fallback
      hot.title = "Copy a permanent link to this slide";
      hot.addEventListener("click", (function (id) {
        return function (e) { onClick(e, id); };
      })(anchor.id));
      sec.appendChild(hot);
    }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
</script>
"""

out = re.sub(r"<!--FIG:([a-z0-9-]+)-->", inline, tpl)
out = re.sub(r"<!--FIGSM:([a-z0-9-]+)-->", inline_sm, out)
out = re.sub(r"<!--FIGFULL:([a-z0-9-]+)-->", inline_full, out)
out += PERMALINK
(here / "slides.md").write_text(out)
print(f"wrote slides.md ({len(out)} bytes)")
