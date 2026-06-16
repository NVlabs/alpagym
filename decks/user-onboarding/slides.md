---
marp: true
theme: nvidia
paginate: true
html: true
---

<a class="slide-id" id="6799ff88-b964-441c-bf98-a7e1bb4a393e"></a>

<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%230a0a0a'/><rect x='4' y='6' width='24' height='16' rx='2' fill='none' stroke='%2376b900' stroke-width='2'/><rect x='8' y='10' width='8' height='2' rx='1' fill='%2376b900'/><rect x='8' y='14' width='12' height='2' rx='1' fill='%2376b900' opacity='.5'/><rect x='8' y='18' width='6' height='2' rx='1' fill='%2376b900' opacity='.3'/><rect x='12' y='22' width='8' height='2' rx='1' fill='%2376b900'/></svg>">

<!-- _class: lead -->

# **AlpaGym**

## Train Alpamayo driving policies in closed-loop simulation

A researcher's quick start — set up, run, tune, and plug in your own policy

---

<a class="slide-id" id="e606169d-b62e-4d3b-8ee8-5fffc729d10a"></a>

## What is AlpaGym?

A **closed-loop RL gym** for Alpamayo driving policies: roll your policy out in **AlpaSim**, score it, and train it with **Cosmos-RL** (GRPO). You bring a policy + checkpoint — the host CLI runs the loop.

<div style="display:flex;justify-content:center;margin-top:0.2rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="860" height="445" style="max-width:100%" font-family="Inter, sans-serif">
  <defs>
    <style>.t{fill:#f0f0f0;font-weight:700}.m{font-family:'JetBrains Mono',monospace;fill:#a3d44a}.d{fill:#a0a0a0}.l{fill:#f0f0f0;font-weight:600}</style>
    <marker id="lg" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#76b900"/></marker>
    <marker id="lb" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#58a6ff"/></marker>
    <marker id="lgr" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#666"/></marker>
  </defs>
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/cli.py" target="_blank">
    <rect x="130" y="26" width="900" height="68" rx="12" fill="#1a1a1a" stroke="#76b900" stroke-width="1.5" stroke-dasharray="6 4"/>
    <text x="580" y="58" text-anchor="middle" class="t" font-size="21">AlpaGym host <tspan class="m" font-size="15">alpagym_host.cli</tspan></text>
    <text x="580" y="82" text-anchor="middle" class="d" font-size="14">one command: resolve config &#183; launch both engines</text>
  </a>
  <path d="M300,94 L300,176" fill="none" stroke="#666" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#lgr)"/>
  <path d="M860,94 L860,176" fill="none" stroke="#666" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#lgr)"/>
  <text x="312" y="138" class="d" font-size="12">launch</text>
  <text x="848" y="138" text-anchor="end" class="d" font-size="12">launch</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/types.py" target="_blank">
    <rect x="120" y="178" width="320" height="152" rx="14" fill="#1a1a1a" stroke="#76b900" stroke-width="2"/>
    <text x="280" y="222" text-anchor="middle" class="t" font-size="23">Policy <tspan fill="#76b900">&#960;&#952;</tspan></text>
    <text x="280" y="252" text-anchor="middle" class="m" font-size="15">Policy.step()</text>
    <text x="280" y="282" text-anchor="middle" class="d" font-size="14">your model + checkpoint —</text>
    <text x="280" y="302" text-anchor="middle" class="d" font-size="14">observations in, trajectory out</text>
  </a>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/driver_server.py" target="_blank">
    <rect x="720" y="178" width="320" height="152" rx="14" fill="#1a1a1a" stroke="#76b900" stroke-width="2"/>
    <text x="880" y="222" text-anchor="middle" class="t" font-size="23">AlpaSim</text>
    <text x="880" y="252" text-anchor="middle" class="m" font-size="15">closed-loop simulator</text>
    <text x="880" y="282" text-anchor="middle" class="d" font-size="14">rolls the ego car through</text>
    <text x="880" y="302" text-anchor="middle" class="d" font-size="14">a recorded driving scene</text>
  </a>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/trainer.py" target="_blank">
    <rect x="420" y="430" width="320" height="142" rx="14" fill="#1a1a1a" stroke="#58a6ff" stroke-width="2"/>
    <text x="580" y="472" text-anchor="middle" class="t" font-size="22">Cosmos-RL</text>
    <text x="580" y="502" text-anchor="middle" class="m" font-size="15">GRPO trainer</text>
    <text x="580" y="532" text-anchor="middle" class="d" font-size="14">scores the rollouts &amp;</text>
    <text x="580" y="552" text-anchor="middle" class="d" font-size="14">updates the policy</text>
  </a>
  <path d="M440,210 C560,176 600,176 720,210" fill="none" stroke="#76b900" stroke-width="2.5" marker-end="url(#lg)"/>
  <text x="580" y="150" text-anchor="middle" class="l" font-size="15">&#9312; acts — planned trajectory each tick</text>
  <text x="580" y="170" text-anchor="middle" fill="#58a6ff" font-size="12">&#8644; over gRPC (egodriver)</text>
  <path d="M832,330 C812,392 770,404 706,430" fill="none" stroke="#76b900" stroke-width="2.5" marker-end="url(#lg)"/>
  <text x="840" y="392" text-anchor="middle" class="l" font-size="14">&#9313; rollouts + reward</text>
  <path d="M420,470 C300,440 296,398 290,332" fill="none" stroke="#58a6ff" stroke-width="2.5" marker-end="url(#lb)"/>
  <text x="318" y="392" text-anchor="middle" fill="#58a6ff" font-weight="600" font-size="14">&#9314; updated weights</text>
  <text x="580" y="300" text-anchor="middle" fill="#666" font-weight="700" font-size="22">the RL loop</text>
</svg>
</div>

<span class="tag tag-green">Tap any box</span> to open the file that implements it.

---

<a class="slide-id" id="210e16c9-3c21-4a4b-8f90-7282d3142677"></a>

## Setup

Three things, once, from the repo root:

<div class="columns">
<div class="card">

### 1 · Workspace

```bash
uv sync --all-packages
```

Local runs also need Docker Compose, Redis, Git LFS, and CUDA/cuDNN/NCCL headers.

</div>
<div class="card">

### 2 · Scenes

Request **NuRec** dataset access (browser), then:

```bash
hf auth login
```

</div>
<div class="card">

### 3 · Weights

Get an Alpamayo **checkpoint** bundle — you'll pass it as `policy.model.path`.

</div>
</div>

> The launcher checks all three up front and **fails fast with a link** if one is missing.

---

<a class="slide-id" id="e1210d46-eea7-4b76-a30e-91a219d19c6c"></a>

## Run it

One local training step with the `alpamayo_r1` policy:

```bash
uv run --no-sync --all-packages python -m alpagym_host.cli \
  experiment=alpamayo_1_5_local_2gpu_smoke \
  policy.model.path=$(pwd)/tmp/checkpoints/<your-ckpt> \
  reward=progress_safety
```

Each run writes a self-contained directory under `./tmp/alpagym-runs/<timestamp>/` (the CLI prints the path). A good smoke run shows `Step: 1/1` in `logs/…/controller.log`.

---

<a class="slide-id" id="b757ad28-dcbf-4324-bda1-10234d3ecdf3"></a>

## Common config changes

Everything is Hydra — override any field as `key=value` on the CLI, or pick a preset by name (the filename in [`conf/`](https://github.com/NVlabs/alpagym/tree/main/packages/host/src/alpagym_host/conf)).

| Override                              | Changes                              | Presets in        |
| ------------------------------------- | ------------------------------------ | ----------------- |
| `deploy=` `local` / `slurm`           | run inline vs. submit a Slurm job    | `conf/deploy/`    |
| `topology=` `local_*` / `slurm_*`     | how GPUs are split across train/sim  | `conf/topology/`  |
| `policy=` (+ `policy.model.kind=`)    | which policy + checkpoint            | `conf/policy/`    |
| `cosmos.rollout.*` / `cosmos.train.*` | rollouts per scene, epochs, LR, …    | — (Hydra override) |

e.g. `cosmos.rollout.n_generation=8 cosmos.train.num_epochs=2`

---

<a class="slide-id" id="8e91b902-c2ad-45f7-aa1f-d79428c32fcc"></a>

<!-- _class: invert -->

# Add your own policy

The part you came for

---

<a class="slide-id" id="40a0da29-ee0e-48df-8004-dfd6a0c941a4"></a>

## 1 · The contract

The simulator calls one small protocol every drive tick — implement [`Policy`](https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/types.py):

```python
class Policy(Protocol):
    def step(self, policy_input: PolicyInput) -> PolicyOutput: ...
    def close(self) -> None: ...
```

For the training side, provide an [`InferenceModel`](https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/inference/types.py) (forward + log-probs for GRPO).

> Copy [`AlpamayoPolicy`](https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/policy.py) as the reference implementation.

---

<a class="slide-id" id="c585c5b9-a247-49e7-8fdd-b81fefc5560f"></a>

## 2 · Wire it in

<div class="columns">
<div>

1. Pick a `kind` string for your policy — e.g. `my_policy` (it's opaque to [`config.py`](https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/config.py), no enum to edit).
2. Ship a package whose `get_bundle()` returns a [`PolicyBundle`](https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/registry.py) and declares an `alpagym.policy_bundles` entry point named `my_policy`. Copy [`alpamayo_r1`](https://github.com/NVlabs/alpagym/blob/main/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/bundle.py).
3. Ship its Hydra configs via the `alpagym.configs` entry point — no public code edits.

</div>
<div>

4. Install the package — `check-policy-bundles` confirms the wiring (no GPU needed).
5. Run it:

```bash
… policy=my_policy \
  policy.model.kind=my_policy \
  policy.model.path=/path/to/ckpt
```

</div>
</div>

<span class="tag tag-green">Why this shape</span> the cosmos trainer stays policy-agnostic — your bundle is imported lazily, only for the kind you run.

---

<a class="slide-id" id="33dd0cfa-5ca1-4b13-bc2f-64c7dce065ce"></a>

<!-- _class: lead -->

## That's the loop

Bring a policy → point at a checkpoint → run → read the run directory.

Go deeper: the **[dev-onboarding deck](dev-onboarding.html)** (full architecture) and the [`README.md`](https://github.com/NVlabs/alpagym/blob/main/README.md). Links point at the public `main` branch.


<script>
(function () {
  var CSS = [
    ".slide-permalink{position:absolute;left:2.4rem;bottom:1.6rem;width:3.4rem;",
    "height:1.9rem;z-index:50;cursor:pointer;text-decoration:none}",
    ".slide-permalink::after{content:'\\1F517';position:absolute;left:3rem;",
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
