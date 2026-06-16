# Decks

Marp slide decks for the AlpaGym project, with shared build tooling and one
NVIDIA theme. Published to **GitHub Pages** from the `gh-pages` branch (see
[Hosting](#hosting)); every diagram box and file name deep-links into the public
repo at [`github.com/NVlabs/alpagym`](https://github.com/NVlabs/alpagym).

## Layout

```
decks/
  tools/           shared, deck-agnostic infra
    build_deck.py  assemble <deck>/slides.md from <deck>/slides.tpl.md + images/*.svg
    assign_slide_ids.py  give each slide a stable UUID permalink anchor
    build.sh       local render to PDF/HTML via Docker (same image as CI)
    serve.sh       host the decks in a browser (live marp server)
    render_figs.sh render a deck's SVGs to PNG previews (needs inkscape)
  themes/
    nvidia.css     the only theme (deck front-matter uses `theme: nvidia`)
  dev-onboarding/  detailed guided tour of the codebase (hand-authored SVG figures)
    slides.tpl.md  source of truth (edit this, not slides.md)
    slides.md      generated, committed
    images/*.svg   hand-authored, clickable figures (inlined into slides.md)
  user-onboarding/ lightweight quick-start: the layout + the knobs you set (text only)
    slides.tpl.md  source of truth
    slides.md      generated, committed
```

## Build locally

```bash
tools/build.sh user-onboarding       # → out/user-onboarding.{pdf,html}
tools/build.sh dev-onboarding pdf    # PDF only
```

Requires Docker. `slides.md` is regenerated from the template on every build.

## Host locally (browser)

```bash
tools/serve.sh                       # http://localhost:8095/  → pick a deck
tools/serve.sh 9000                  # custom port
```

Stop with Ctrl-C. (Auto-reload needs marp's socket port 37717 free — see
`tools/serve.sh` if another deck server is already running.)

## Edit a deck

Edit `slides.tpl.md`, not the generated `slides.md`. A pre-commit hook assigns
permalinks to new slides and regenerates `slides.md`, so the committed output
never drifts. To regenerate by hand:

```bash
python3 tools/build_deck.py dev-onboarding
```

## Add a deck

1. `mkdir <deck>` and add `<deck>/slides.tpl.md` with Marp front-matter
   (`theme: nvidia`). For figures, add `<deck>/images/*.svg` and reference them with
   `<!--FIG:name-->` tokens (`FIGSM`/`FIGFULL` for smaller / full-bleed variants); a
   text-only deck needs neither.
2. `tools/build.sh <deck>` to preview.

Both the local tools and the GitHub Pages workflow auto-discover it — every
`*/slides.tpl.md` is rendered, no config edit needed.

## Hosting

Published to **GitHub Pages** by [`.github/workflows/pages.yml`](../.github/workflows/pages.yml):
every push to `gh-pages` renders each deck to HTML and deploys the site (landing
page + one page per deck). HTML keeps the in-figure deep-links and per-slide
permalinks live. The published URL is printed in the workflow run summary,
typically:

```
https://nvlabs.github.io/alpagym/
```
