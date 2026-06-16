---
marp: true
theme: nvidia
paginate: true
html: true
---

<a class="slide-id" id="0d65c412-6686-4f70-a6d7-a42db356fced"></a>

<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%230a0a0a'/><rect x='4' y='6' width='24' height='16' rx='2' fill='none' stroke='%2376b900' stroke-width='2'/><rect x='8' y='10' width='8' height='2' rx='1' fill='%2376b900'/><rect x='8' y='14' width='12' height='2' rx='1' fill='%2376b900' opacity='.5'/><rect x='8' y='18' width='6' height='2' rx='1' fill='%2376b900' opacity='.3'/><rect x='12' y='22' width='8' height='2' rx='1' fill='%2376b900'/></svg>">

<!-- _class: lead -->

# **AlpaGym**

## The rollout & training harness for Alpamayo policies

A newcomer's guided tour — broad concepts first, then the code

---

<a class="slide-id" id="81d5ee9a-f8e0-433b-9a62-777cb2ecccc0"></a>

<!-- _class: lead -->

## What you'll leave with

A mental model of how a driving **policy**, the **AlpaSim** simulator and the **Cosmos-RL** trainer fit together — and a map of exactly *where in the code* each piece lives.

Every diagram box and file name in this deck is a **clickable link** into the repo at [`github.com/NVlabs/alpagym`](https://github.com/NVlabs/alpagym) · tracks the public `main` branch.

---

<a class="slide-id" id="89d6301d-4b86-46c2-be50-4c7f3cd85b87"></a>

## Agenda

<div class="columns">
<div>

1. [**The Big Picture**](#the-big-picture) — what it is & how it fits
2. [**Run It**](#run-it) — your first rollout in minutes
3. [**The Control Plane**](#the-control-plane) — the `host` package
4. [**The RL Runtime**](#the-rl-runtime) — the Cosmos-RL engine
5. [**The Episode Loop**](#the-episode-loop) — sim ⇄ policy
6. [**Policy & Inference**](#policy-and-inference) — how the car decides

</div>
<div>

7. [**Rewards**](#rewards) — scoring an episode
8. [**Episode Transport**](#episode-transport) — disk → NCCL
9. [**The Plugin Boundary**](#the-plugin-boundary) — bring your own policy
10. [**Engineering Culture**](#engineering-culture) — how we write code
11. [**Where To Look**](#where-to-look) — the reference map

</div>
</div>

---

<a class="slide-id" id="bc60802c-3cf3-417b-b240-6788b80ac851"></a>

<!-- _class: invert -->

# The Big **Picture**

---

<a class="slide-id" id="be5ce82e-f88b-4b30-b50f-d42b29b04d04"></a>

## What is AlpaGym?

> **AlpaGym** is the **rollout and training harness** for Alpamayo autonomous-driving policies. It runs a policy inside a closed-loop simulator, scores the resulting drives, and feeds them to a reinforcement-learning trainer.

<div class="columns">
<div>

### Three moving parts

- **Policy** — the neural net that drives
- **AlpaSim** — the simulated world
- **Cosmos-RL** — the RL trainer

A fourth piece, the **host**, orchestrates them.

</div>
<div>

### Why it exists

- A *lightweight*, researcher-friendly closed-loop RL harness
- Lives at [`github.com/NVlabs/alpagym`][readme]
- Clean interfaces so researchers can swap policies, rewards and models

</div>
</div>

---

<a class="slide-id" id="2e686df7-ca2a-482f-9ab7-38490f191c95"></a>

## The mental model: closed-loop RL for driving

It's the classic **agent ⇄ environment** loop — specialised for autonomous driving. The policy proposes a trajectory; the simulator executes it and returns the next observation; episodes and rewards flow to the trainer.

<div style="display:flex;justify-content:center;margin-top:0.4rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="980" height="507" style="max-width:100%" font-family="Inter, sans-serif">
  <defs>
    <marker id="ah" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto">
      <path d="M0,0 L9,4 L0,8 z" fill="#76b900"/>
    </marker>
    <marker id="ahb" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto">
      <path d="M0,0 L9,4 L0,8 z" fill="#58a6ff"/>
    </marker>
    <marker id="ahg" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto">
      <path d="M0,0 L9,4 L0,8 z" fill="#666"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <!-- Host control plane banner -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/cli.py" target="_blank">
    <rect x="80" y="30" width="1000" height="78" rx="12" fill="#1a1a1a" stroke="#76b900" stroke-width="1.5" stroke-dasharray="6 4"/>
    <text x="580" y="62" text-anchor="middle" fill="#f0f0f0" font-size="22" font-weight="700">AlpaGym host — control plane <tspan fill="#58a6ff" font-size="15" font-weight="500">(Hydra CLI)</tspan></text>
    <text x="580" y="88" text-anchor="middle" fill="#a0a0a0" font-size="15">resolves config · prepares AlpaSim checkout · starts the simulator · launches the Cosmos-RL runtime · wires gRPC endpoints</text>
  </a>
  <!-- launch arrows -->
  <path d="M280,108 L280,168" fill="none" stroke="#666" stroke-width="2" marker-end="url(#ahg)"/>
  <path d="M880,108 L880,168" fill="none" stroke="#666" stroke-width="2" marker-end="url(#ahg)"/>
  <text x="268" y="142" text-anchor="end" fill="#888" font-size="13">launches Cosmos-RL runtime</text>
  <text x="892" y="142" fill="#888" font-size="13">starts AlpaSim (Wizard)</text>
  <!-- Policy (agent) -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/policy.py" target="_blank">
    <rect x="80" y="170" width="300" height="150" rx="14" fill="#1a1a1a" stroke="#76b900" stroke-width="2"/>
    <text x="230" y="212" text-anchor="middle" fill="#f0f0f0" font-size="24" font-weight="700">Policy  <tspan fill="#76b900">&#960;&#952;</tspan></text>
    <text x="230" y="242" text-anchor="middle" fill="#a3d44a" font-size="15" font-family="JetBrains Mono, monospace">AlpamayoPolicy.step()</text>
    <text x="230" y="272" text-anchor="middle" fill="#a0a0a0" font-size="14">the agent — observations in,</text>
    <text x="230" y="292" text-anchor="middle" fill="#a0a0a0" font-size="14">planned trajectory out</text>
  </a>
  <!-- AlpaSim (environment) -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/driver_server.py" target="_blank">
    <rect x="780" y="170" width="300" height="150" rx="14" fill="#1a1a1a" stroke="#76b900" stroke-width="2"/>
    <text x="930" y="212" text-anchor="middle" fill="#f0f0f0" font-size="24" font-weight="700">AlpaSim</text>
    <text x="930" y="242" text-anchor="middle" fill="#a3d44a" font-size="15" font-family="JetBrains Mono, monospace">closed-loop simulator</text>
    <text x="930" y="272" text-anchor="middle" fill="#a0a0a0" font-size="14">the environment — drives the</text>
    <text x="930" y="292" text-anchor="middle" fill="#a0a0a0" font-size="14">ego car, renders the world</text>
  </a>
  <!-- action arrow (top) Policy -> AlpaSim -->
  <path d="M380,205 C540,172 620,172 780,205" fill="none" stroke="#76b900" stroke-width="2.5" marker-end="url(#ah)"/>
  <text x="580" y="132" text-anchor="middle" fill="#f0f0f0" font-size="15" font-weight="600">action: planned trajectory</text>
  <text x="580" y="152" text-anchor="middle" fill="#666" font-size="13" font-family="JetBrains Mono, monospace">DriveResponse</text>
  <!-- observation arrow (bottom) AlpaSim -> Policy -->
  <path d="M780,285 C620,340 540,340 380,285" fill="none" stroke="#58a6ff" stroke-width="2.5" marker-end="url(#ahb)"/>
  <text x="580" y="332" text-anchor="middle" fill="#f0f0f0" font-size="15" font-weight="600">observation: cameras · ego history · route</text>
  <text x="580" y="352" text-anchor="middle" fill="#666" font-size="13" font-family="JetBrains Mono, monospace">DriveRequest</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/proto_conversion.py" target="_blank">
    <text x="580" y="247" text-anchor="middle" fill="#58a6ff" font-size="13">⇄ over gRPC (egodriver)</text>
  </a>
  <!-- Trainer -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/trainer.py" target="_blank">
    <rect x="410" y="438" width="340" height="120" rx="14" fill="#1a1a1a" stroke="#58a6ff" stroke-width="2"/>
    <text x="580" y="478" text-anchor="middle" fill="#f0f0f0" font-size="22" font-weight="700">Cosmos-RL trainer</text>
    <text x="580" y="506" text-anchor="middle" fill="#a3d44a" font-size="15" font-family="JetBrains Mono, monospace">GRPO · step_training()</text>
    <text x="580" y="534" text-anchor="middle" fill="#a0a0a0" font-size="14">learns from rollout episodes &amp; rewards</text>
  </a>
  <!-- episodes+reward arrow AlpaSim -> trainer -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/episode_runner/streaming_worker.py" target="_blank">
    <path d="M930,320 C930,420 820,440 752,470" fill="none" stroke="#76b900" stroke-width="2.5" marker-end="url(#ah)"/>
    <text x="905" y="400" text-anchor="middle" fill="#f0f0f0" font-size="14" font-weight="600">episodes + reward</text>
  </a>
  <!-- weight update (dashed, future) trainer -> policy -->
  <path d="M410,478 C250,460 230,420 230,322" fill="none" stroke="#666" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#ahg)"/>
  <text x="150" y="404" text-anchor="end" fill="#666" font-size="14" font-weight="600">weight update</text>
  <text x="150" y="422" text-anchor="end" fill="#666" font-size="12">(roadmap)</text>
</svg>
</div>

<span class="tag tag-blue">Tap any box</span> Host → [`cli.py`][cli] · Policy → [`policy.py`][policy] · AlpaSim/egodriver → [`driver_server.py`][driver] · Trainer → [`trainer.py`][trainer]

---

<a class="slide-id" id="da515d94-fa93-4d8f-835b-120976c4b39a"></a>

## Key terms (glossary)

<div class="columns">
<div class="card">

### Policy
The driver. [`AlpamayoPolicy`][policy] maps an observation (cameras, ego history, route) to a future **trajectory**.

### AlpaSim
The closed-loop **simulator** (a separate repo). AlpaGym drives it over gRPC via the **egodriver**.

### Cosmos-RL
NVIDIA's RL training framework (GRPO). Supplies the launcher, trainer and rollout abstractions AlpaGym plugs into.

</div>
<div class="card">

### Episode / rollout
One simulated drive of a **scene** → an executed trajectory, metrics and a scalar **reward**.

### Egodriver
The gRPC **server** AlpaGym runs so the simulator can ask *"what should the car do next?"*.

### Wizard
AlpaSim's own launcher. The host starts it to bring the simulator up.

</div>
</div>

---

<a class="slide-id" id="e7500430-c254-41e7-b199-344a04d37c1e"></a>

## Repository layout

Five packages under [`packages/`](https://github.com/NVlabs/alpagym/tree/main/packages). **host** (control plane) and **runtime** (the harness) are the core; **plugins** + **policies** form the extensibility layer — policies plug in via entry points — and **alpasim_configs** ships simulator presets.

<div style="display:flex;justify-content:center;margin-top:0.4rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="980" height="507" style="max-width:100%" font-family="Inter, sans-serif">
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <style>
    .mono { font-family: 'JetBrains Mono', monospace; }
    .lnk { fill: #58a6ff; }
    .desc { fill: #a0a0a0; }
    .grp { fill: #76b900; font-weight: 700; }
  </style>
  <!-- HOST -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/cli.py" target="_blank">
    <rect x="24" y="48" width="300" height="510" rx="12" fill="#161616" stroke="#76b900" stroke-width="1.5"/>
  </a>
  <text x="44" y="86" class="mono grp" font-size="17">packages/host</text>
  <text x="44" y="108" class="desc" font-size="13">control plane · runs on login node</text>
  <line x1="44" y1="122" x2="304" y2="122" stroke="#2a2a2a"/>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/cli.py" target="_blank"><text x="44" y="156" class="mono lnk" font-size="14">cli.py</text></a>
  <text x="170" y="156" class="desc" font-size="12">run / submit entrypoint</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/config.py" target="_blank"><text x="44" y="190" class="mono lnk" font-size="14">config.py</text></a>
  <text x="170" y="190" class="desc" font-size="12">typed Hydra schema</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/run_artifacts.py" target="_blank"><text x="44" y="224" class="mono lnk" font-size="14">run_artifacts.py</text></a>
  <text x="44" y="244" class="desc" font-size="12">run dir · resolved config · cosmos.toml</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/alpasim_wizard.py" target="_blank"><text x="44" y="278" class="mono lnk" font-size="14">alpasim_wizard.py</text></a>
  <text x="44" y="298" class="desc" font-size="12">start / wait / stop the simulator</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/endpoint_registry.py" target="_blank"><text x="44" y="332" class="mono lnk" font-size="14">endpoint_registry.py</text></a>
  <text x="44" y="352" class="desc" font-size="12">file-backed endpoint registry</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/alpasim_dependency.py" target="_blank"><text x="44" y="386" class="mono lnk" font-size="14">alpasim_dependency.py</text></a>
  <text x="44" y="406" class="desc" font-size="12">resolve / cache AlpaSim checkout</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/slurm.py" target="_blank"><text x="44" y="440" class="mono lnk" font-size="14">slurm.py</text></a>
  <text x="170" y="440" class="desc" font-size="12">sbatch · enroot sqsh cache</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/conf/default.yaml" target="_blank"><text x="44" y="474" class="mono lnk" font-size="14">conf/</text></a>
  <text x="170" y="474" class="desc" font-size="12">default · policy/ · reward/</text>
  <text x="44" y="522" class="desc" font-size="12">→ launches the two boxes on the right,</text>
  <text x="44" y="540" class="desc" font-size="12">&#160;&#160;&#160;then hands off to Cosmos-RL.</text>
  <!-- RUNTIME -->
  <a href="https://github.com/NVlabs/alpagym/tree/main/packages/runtime/src/alpagym_runtime" target="_blank">
    <rect x="356" y="48" width="500" height="510" rx="12" fill="#161616" stroke="#76b900" stroke-width="2"/>
  </a>
  <text x="376" y="86" class="mono grp" font-size="17">packages/runtime</text>
  <text x="376" y="108" class="desc" font-size="13">the harness · runs in the GPU container</text>
  <line x1="376" y1="122" x2="836" y2="122" stroke="#2a2a2a"/>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/rollout_backend.py" target="_blank"><text x="376" y="156" class="mono lnk" font-size="15">cosmos/</text></a>
  <text x="376" y="176" class="desc" font-size="12.5">Cosmos-RL glue: rollout backend · trainer · dataset · packer · reward_fn</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/episode_runner/streaming_worker.py" target="_blank"><text x="376" y="212" class="mono lnk" font-size="15">episode_runner/</text></a>
  <text x="376" y="232" class="desc" font-size="12.5">drives one batch of simulator episodes — the rollout loop</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/driver_server.py" target="_blank"><text x="376" y="268" class="mono lnk" font-size="15">alpasim/</text></a>
  <text x="376" y="288" class="desc" font-size="12.5">gRPC egodriver ↔ simulator: driver_server · proto_conversion · tick_buffer</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/inference/inference_engine.py" target="_blank"><text x="376" y="324" class="mono lnk" font-size="15">inference/</text></a>
  <text x="376" y="344" class="desc" font-size="12.5">batched GPU inference dispatcher (futures + one worker thread)</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/policy.py" target="_blank"><text x="376" y="380" class="mono lnk" font-size="15">policies/</text></a>
  <text x="376" y="400" class="desc" font-size="12.5">AlpamayoPolicy + selectors; model code comes from policy packages</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/rewards/compute.py" target="_blank"><text x="376" y="436" class="mono lnk" font-size="15">rewards/</text></a>
  <text x="376" y="456" class="desc" font-size="12.5">reward terms → scalar: compute · distance_to_gt</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/disk.py" target="_blank"><text x="376" y="492" class="mono lnk" font-size="15">transport/</text></a>
  <text x="376" y="512" class="desc" font-size="12.5">episode ⇄ JSON artifact on disk</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/types.py" target="_blank"><text x="376" y="540" class="mono lnk" font-size="14">types.py</text></a>
  <text x="490" y="540" class="desc" font-size="12.5">shared dataclasses: PolicyInput · PolicyOutput · Trajectory · …</text>
  <!-- PLUGINS + POLICIES -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/plugins/src/alpagym_plugins/plugins.py" target="_blank">
    <rect x="888" y="48" width="248" height="250" rx="12" fill="#161616" stroke="#76b900" stroke-width="1.5"/>
  </a>
  <text x="908" y="86" class="mono grp" font-size="15">plugins + policies</text>
  <text x="908" y="108" class="desc" font-size="13">the extensibility layer</text>
  <line x1="908" y1="122" x2="1116" y2="122" stroke="#2a2a2a"/>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/plugins/src/alpagym_plugins/plugins.py" target="_blank"><text x="908" y="156" class="mono lnk" font-size="14">plugins/</text></a>
  <text x="908" y="176" class="desc" font-size="12.5">entry-point registry: discover</text>
  <text x="908" y="194" class="desc" font-size="12.5">installed policy bundles</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/bundle.py" target="_blank"><text x="908" y="228" class="mono lnk" font-size="13.5">policies/alpamayo_r1</text></a>
  <text x="908" y="248" class="desc" font-size="12.5">a model packaged as a plugin;</text>
  <text x="908" y="266" class="desc" font-size="12.5">any policy package plugs in here</text>
  <!-- legend -->
  <rect x="888" y="330" width="248" height="228" rx="12" fill="#0d1117" stroke="#2a2a2a" stroke-width="1"/>
  <text x="908" y="362" class="grp" font-size="14">Reading order</text>
  <text x="908" y="392" class="desc" font-size="12.5">1. host launches everything</text>
  <text x="908" y="416" class="desc" font-size="12.5">2. cosmos/ wires the RL loop</text>
  <text x="908" y="440" class="desc" font-size="12.5">3. episode_runner drives sim</text>
  <text x="908" y="464" class="desc" font-size="12.5">4. alpasim/ talks to simulator</text>
  <text x="908" y="488" class="desc" font-size="12.5">5. policies/ + inference/ act</text>
  <text x="908" y="512" class="desc" font-size="12.5">6. rewards/ score the episode</text>
  <text x="908" y="540" class="desc mono" font-size="11.5" fill="#58a6ff">every name above is a link →</text>
</svg>
</div>

---

<a class="slide-id" id="004c6c25-d432-4652-8bc8-bc997f81fd97"></a>

## The layered architecture

Inside `runtime`, the code is organized as **layers with one job each** — and only `cosmos/` is Cosmos-RL-aware, so everything below it is reusable. *Full write-up: the [runtime README][runtimereadme].*

<div style="display:flex;justify-content:center;margin-top:0.4rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="980" height="507" style="max-width:100%" font-family="Inter, sans-serif">
  <defs>
    <marker id="g" markerWidth="10" markerHeight="10" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#76b900"/></marker>
    <marker id="b" markerWidth="10" markerHeight="10" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#58a6ff"/></marker>
    <marker id="gy" markerWidth="10" markerHeight="10" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#888"/></marker>
  </defs>
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <style>.t{fill:#f0f0f0;font-weight:700}.m{font-family:'JetBrains Mono',monospace;fill:#a3d44a}.d{fill:#a0a0a0}.chip{fill:#1a1a1a;stroke:#2a2a2a}</style>
  <text x="30" y="42" class="d" font-size="15">Zooming out: four orchestration layers wrap the per-tick layers. <tspan fill="#f0f0f0" font-weight="700">Only <tspan class="m" font-size="14">cosmos/</tspan> knows Cosmos-RL.</tspan></text>
  <!-- Cosmos framework -->
  <rect x="40" y="60" width="700" height="42" rx="9" fill="#1a1a1a" stroke="#bc8cff" stroke-width="1.5"/>
  <text x="390" y="87" text-anchor="middle" class="t" font-size="15">Cosmos-RL framework</text>
  <!-- adapter layer -->
  <a href="https://github.com/NVlabs/alpagym/tree/main/packages/runtime/src/alpagym_runtime/cosmos" target="_blank">
    <rect x="40" y="128" width="700" height="96" rx="11" fill="#161616" stroke="#76b900" stroke-width="1.8"/>
  </a>
  <text x="58" y="152" class="t" font-size="15">Cosmos-RL adapter layer <tspan class="m" font-size="12">cosmos/</tspan></text>
  <g font-size="11.5">
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/entrypoint.py" target="_blank"><rect class="chip" x="58" y="166" width="100" height="24" rx="6"/><text x="108" y="182" text-anchor="middle" class="m" font-size="11">entrypoint</text></a>
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/dataset.py" target="_blank"><rect class="chip" x="166" y="166" width="84" height="24" rx="6"/><text x="208" y="182" text-anchor="middle" class="m" font-size="11">dataset</text></a>
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/packer.py" target="_blank"><rect class="chip" x="258" y="166" width="78" height="24" rx="6"/><text x="297" y="182" text-anchor="middle" class="m" font-size="11">packer</text></a>
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/trainer.py" target="_blank"><rect class="chip" x="344" y="166" width="80" height="24" rx="6"/><text x="384" y="182" text-anchor="middle" class="m" font-size="11">trainer</text></a>
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/reward_fn.py" target="_blank"><rect class="chip" x="432" y="166" width="96" height="24" rx="6"/><text x="480" y="182" text-anchor="middle" class="m" font-size="11">reward_fn</text></a>
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/rollout_backend.py" target="_blank"><rect class="chip" x="536" y="166" width="186" height="24" rx="6" stroke="#76b900"/><text x="629" y="182" text-anchor="middle" class="m" font-size="11" fill="#a3d44a">rollout_backend (build site)</text></a>
  </g>
  <text x="58" y="212" class="d" font-size="11.5">six pure adapters + one construction site that builds the engine, policy factory, driver server &amp; worker</text>
  <!-- episode runner -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/episode_runner/streaming_worker.py" target="_blank">
    <rect x="40" y="250" width="700" height="74" rx="11" fill="#161616" stroke="#76b900" stroke-width="1.8"/>
  </a>
  <text x="58" y="276" class="t" font-size="15">Episode runner layer <tspan class="m" font-size="12">episode_runner/</tspan></text>
  <text x="58" y="298" class="d" font-size="12.5">orchestrates rollouts against AlpaSim — runs the drive ticks, then scores &amp; persists</text>
  <text x="58" y="316" class="d" font-size="11.5">streaming <tspan class="m" font-size="11">simulate(1)</tspan> worker — one rollout per call (<tspan class="m" font-size="11">streaming_worker</tspan>)</text>
  <!-- per-tick layers (collapsed) -->
  <rect x="40" y="350" width="700" height="78" rx="11" fill="#0d1117" stroke="#58a6ff" stroke-width="1.5" stroke-dasharray="6 4"/>
  <text x="58" y="376" class="t" font-size="15" fill="#58a6ff">Per-tick rollout layers</text>
  <text x="58" y="398" class="d" font-size="12.5">simulator · policy · inference engine · inference model · released model</text>
  <text x="58" y="418" class="d" font-size="11.5">(the five per-tick layers — run once per drive tick; detailed later)</text>
  <!-- reward + transfer (right column) -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/rewards/compute.py" target="_blank">
    <rect x="772" y="250" width="348" height="74" rx="11" fill="#161616" stroke="#76b900" stroke-width="1.6"/>
    <text x="792" y="276" class="t" font-size="15">Reward layer <tspan class="m" font-size="12">rewards/</tspan></text>
    <text x="792" y="298" class="d" font-size="12.5">compute_reward — pure compute,</text>
    <text x="792" y="316" class="d" font-size="12.5">no I/O, no model deps</text>
  </a>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/base.py" target="_blank">
    <rect x="772" y="350" width="348" height="78" rx="11" fill="#161616" stroke="#76b900" stroke-width="1.6"/>
    <text x="792" y="376" class="t" font-size="15">Transport layer <tspan class="m" font-size="12">transport/</tspan></text>
    <text x="792" y="398" class="d" font-size="12.5">persists EpisodeOutput rollout → trainer</text>
    <text x="792" y="418" class="d" font-size="12.5">disk .pt (default) · NCCL (opt-in)</text>
  </a>
  <!-- arrows -->
  <path d="M390,102 L390,126" stroke="#bc8cff" stroke-width="2" marker-end="url(#b)"/>
  <text x="400" y="119" class="d" font-size="11">rollout_generation · prefetch</text>
  <path d="M390,224 L390,248" stroke="#76b900" stroke-width="2" marker-end="url(#g)"/>
  <text x="400" y="241" class="d" font-size="11">submit_payload</text>
  <path d="M390,324 L390,348" stroke="#76b900" stroke-width="2" marker-end="url(#g)"/>
  <text x="400" y="341" class="d" font-size="11">per drive tick</text>
  <!-- runner -> reward & transfer -->
  <path d="M740,287 L770,287" stroke="#76b900" stroke-width="2" marker-end="url(#g)"/>
  <text x="744" y="278" class="d" font-size="10.5">score</text>
  <path d="M740,300 C758,300 758,386 770,386" fill="none" stroke="#76b900" stroke-width="2" marker-end="url(#g)"/>
  <text x="744" y="360" class="d" font-size="10.5">write</text>
  <!-- transfer read back -> adapter -->
  <path d="M946,348 C946,150 850,150 742,182" fill="none" stroke="#888" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#gy)"/>
  <text x="788" y="132" class="d" font-size="12" fill="#cfcfcf">read back ↩</text>
  <text x="788" y="150" font-size="11" fill="#cfcfcf">→ <tspan class="m" fill="#a3d44a">packer</tspan> / <tspan class="m" fill="#a3d44a">reward_fn</tspan></text>
</svg>
</div>

---

<a class="slide-id" id="80fc7a2e-af70-4e70-83cd-6485e1b16925"></a>

## How a run flows, end to end

<div class="columns">
<div>

### On the host (control plane)
1. Compose config with Hydra → freeze it
2. Resolve / cache the **AlpaSim** checkout
3. Start **AlpaSim's Wizard** launcher (brings the simulator up) & publish its **gRPC endpoint** — the channel AlpaGym opens to drive AlpaSim (the `simulate()` RPC)
4. Launch the **Cosmos-RL** runtime

</div>
<div>

### In the runtime (GPU)
5. Rollout backend pulls scenes, runs **episodes** against AlpaSim
6. Each tick: AlpaSim pushes camera/ego/route observations to AlpaGym's **egodriver** gRPC server → it steps the **policy** → returns the chosen **trajectory** (rig→world); AlpaSim advances the ego car and asks again
7. Score episodes → write artifacts
8. Trainer consumes them (RL)

</div>
</div>

> Local runs do steps 1–8 on one machine; Slurm runs prepare 1–4 on the login node and re-run on the cluster. See [**Run It**](#run-it).

---

<a class="slide-id" id="a265f34d-e0f6-4cdb-865d-ed24636fa9b7"></a>

<!-- _class: invert -->

# Run **It**

---

<a class="slide-id" id="1a9800ce-8e4d-4f81-af3c-aff3f8950955"></a>

## Your first rollout: a local run

Run the whole machine on one box. The simplest entry is an **`experiment`** preset — it bundles a `deploy` target, a `topology` and a policy. The public `alpamayo_1_5` smoke preset brings up AlpaSim via Wizard + Docker Compose and runs the Cosmos-RL runtime inline:

```bash
# from the repo root, after: uv sync --all-packages
uv run --no-sync --all-packages python -m alpagym_host.cli \
  experiment=alpamayo_1_5_local_2gpu_smoke \
  policy.model.path=$(pwd)/tmp/checkpoints/alpamayo-1.5-10B_alpagym_ckpt \
  reward=progress_safety
```

This runs one episode against AlpaSim, computes a [reward](#rewards), takes one **GRPO** step and writes artifacts.

> **Why start here?** It exercises every interface in the [episode loop](#the-episode-loop) and [RL runtime](#the-rl-runtime). `deploy=local` clones the public AlpaSim repo; config defaults live in [`conf/default.yaml`][defaults], and the `alpamayo_1_5` presets ship with the [`alpamayo_r1`](#policy-and-inference) policy package.

---

<a class="slide-id" id="2f11a74a-bf36-454e-8f44-21496e67d891"></a>

## Two ways to run: local vs Slurm

<div class="columns">
<div class="card">

### `deploy=local`
- Runs the host **and** the Cosmos-RL runtime inline on your machine
- AlpaSim comes up via **Docker Compose**
- Great for development & debugging
- Pair with a local topology — `topology=local_colocated_1gpu`

</div>
<div class="card">

### `deploy=slurm`
- Host **prepares** the run on the login node, then submits an `sbatch` job
- Runtime executes inside the pinned **container** on a GPU node
- The cluster node re-loads `resolved_config.yaml` and runs the same lifecycle
- Pair with a Slurm topology — `topology=slurm_full_node_1_3_4`

</div>
</div>

The **`topology`** preset selects the backend (`local_process` vs `slurm`); the **`deploy`** preset fills the site settings (partition, account, image, caches) and, for `deploy=slurm`, switches the command to **submit** — see [`config.py`][config]. Slurm submission & image caching live in [`slurm.py`][slurm].

---

<a class="slide-id" id="6c2cffd0-1316-4ea8-8430-874449159e61"></a>

## Running on the cluster

Override the experiment's `deploy`/`topology` to submit a Slurm job. The container mounts shared storage, so the checkpoint path can be passed directly:

```bash
uv run --no-sync --all-packages python -m alpagym_host.cli \
  experiment=alpamayo_1_5_local_2gpu_smoke \
  deploy=slurm \
  topology=slurm_full_node_1_3_4 \
  policy.model.path=/path/to/alpamayo-1.5-10B_alpagym_ckpt
```

> AlpaGym passes only `policy.model.path`; the [`alpamayo_r1` policy package](#the-plugin-boundary) owns its model code and resolves the trajectory tokenizer through its bundle. Full recipes (local install, HF auth, image cache) are in the [README][readme].

---

<a class="slide-id" id="610bb8a2-3209-4700-8e83-fff0b8d4bfcc"></a>

<!-- _class: invert -->

# The Control **Plane**

<p><code>packages/host</code> — orchestrates everything, runs on the login node</p>

---

<a class="slide-id" id="70d49554-4933-4c7f-8219-f6c93ea8943e"></a>

## The CLI & run lifecycle

One Hydra entrypoint, [`cli.py`][cli]: `deploy=local` executes inline, `deploy=slurm` submits an `sbatch` job. The core lifecycle lives in [`execute_run()`][execute_run] (in `run_lifecycle.py`):

<div class="columns">
<div>

1. Resolve & cache the AlpaSim checkout — [`resolve_alpasim_checkout()`][dep]
2. Start the Wizard subprocess — [`start_wizard()`][wizard]
3. Wait for its gRPC endpoint — [`wait_for_runtime_ready()`][wizard]

</div>
<div>

4. Publish the endpoint to the [topology registry][topo] so the runtime can find it
5. **Launch the [Cosmos-RL runtime](#the-rl-runtime)** — [`_build_cosmos_launcher_command()`](https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/run_lifecycle.py) runs `cosmos_rl.launcher.launch_all` (controller + policy + rollout replicas); on Slurm it's wrapped in `srun`
6. On exit, tear the Wizard down

</div>
</div>

> Everything downstream is driven by one **frozen config** the host writes before launching — never by ad-hoc flags. See next slide.

---

<a class="slide-id" id="e1a89ae3-3856-4751-82bd-acfbba595de0"></a>

## Configuration: one source of truth

The host composes YAML defaults and CLI overrides into a typed [`RunConfig`][config], then writes per-run artifacts via [`write_run_artifacts()`][writeart].

<div style="display:flex;justify-content:center;margin-top:0.4rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="980" height="507" style="max-width:100%" font-family="Inter, sans-serif">
  <defs>
    <marker id="c" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#76b900"/></marker>
  </defs>
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <style>.t{fill:#f0f0f0;font-weight:700}.m{font-family:'JetBrains Mono',monospace;fill:#58a6ff}.d{fill:#a0a0a0}.g{fill:#76b900;font-weight:700}</style>
  <text x="30" y="50" class="d" font-size="15"><tspan class="g">One source of truth</tspan> — Hydra composes YAML defaults + CLI overrides into a typed, frozen config.</text>
  <!-- INPUTS -->
  <text x="30" y="92" class="g" font-size="13">INPUTS</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/conf/default.yaml" target="_blank">
    <rect x="24" y="104" width="288" height="64" rx="10" fill="#1a1a1a" stroke="#2a2a2a"/>
    <text x="40" y="130" class="t" font-size="14">conf/default.yaml</text>
    <text x="40" y="152" class="d" font-size="12">root defaults for every group</text>
  </a>
  <a href="https://github.com/NVlabs/alpagym/tree/main/packages/host/src/alpagym_host/conf/policy" target="_blank">
    <rect x="24" y="180" width="288" height="64" rx="10" fill="#1a1a1a" stroke="#2a2a2a"/>
    <text x="40" y="206" class="t" font-size="14">conf/policy/*.yaml</text>
    <text x="40" y="228" class="d" font-size="12">alpamayo</text>
  </a>
  <a href="https://github.com/NVlabs/alpagym/tree/main/packages/host/src/alpagym_host/conf/reward" target="_blank">
    <rect x="24" y="256" width="288" height="64" rx="10" fill="#1a1a1a" stroke="#2a2a2a"/>
    <text x="40" y="282" class="t" font-size="14">conf/reward/*.yaml</text>
    <text x="40" y="304" class="d" font-size="12">distance_to_gt · metrics · progress_safety</text>
  </a>
  <rect x="24" y="332" width="288" height="64" rx="10" fill="#0d1117" stroke="#f0883e" stroke-width="1.4"/>
  <text x="40" y="358" class="t" font-size="14">CLI overrides</text>
  <text x="40" y="380" class="d m" font-size="11.5" fill="#f0883e">policy=alpamayo  policy.model.path=…</text>
  <!-- compose arrows -->
  <path d="M314,136 C350,136 350,236 384,236" fill="none" stroke="#76b900" stroke-width="2" marker-end="url(#c)"/>
  <path d="M314,212 L384,230" stroke="#76b900" stroke-width="2" marker-end="url(#c)"/>
  <path d="M314,288 C350,288 350,244 384,244" fill="none" stroke="#76b900" stroke-width="2" marker-end="url(#c)"/>
  <path d="M314,364 C350,364 350,256 384,252" fill="none" stroke="#76b900" stroke-width="2" marker-end="url(#c)"/>
  <!-- RunConfig -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/config.py" target="_blank">
    <rect x="388" y="120" width="356" height="300" rx="12" fill="#161616" stroke="#76b900" stroke-width="2"/>
  </a>
  <text x="408" y="152" class="t" font-size="17">RunConfig</text>
  <text x="408" y="172" class="m" font-size="12">config.py · typed dataclasses</text>
  <text x="408" y="190" class="d" font-size="11.5">registered with Hydra ConfigStore</text>
  <g>
    <rect x="408" y="206" width="316" height="30" rx="6" fill="#1a1a1a"/><text x="420" y="226" class="d" font-size="12.5"><tspan class="g">policy</tspan>  — model kind, checkpoint, sampling</text>
    <rect x="408" y="240" width="316" height="30" rx="6" fill="#1a1a1a"/><text x="420" y="260" class="d" font-size="12.5"><tspan class="g">reward</tspan>  — weighted reward terms</text>
    <rect x="408" y="274" width="316" height="30" rx="6" fill="#1a1a1a"/><text x="420" y="294" class="d" font-size="12.5"><tspan class="g">execution</tspan>  — local_process / slurm</text>
    <rect x="408" y="308" width="316" height="30" rx="6" fill="#1a1a1a"/><text x="420" y="328" class="d" font-size="12.5"><tspan class="g">alpasim</tspan>  — Wizard checkout + startup</text>
    <rect x="408" y="342" width="316" height="30" rx="6" fill="#1a1a1a"/><text x="420" y="362" class="d" font-size="12.5"><tspan class="g">cosmos</tspan>  — replicas, parallelism, training</text>
    <rect x="408" y="376" width="316" height="30" rx="6" fill="#1a1a1a"/><text x="420" y="396" class="d" font-size="12.5"><tspan class="g">dataset</tspan>  — scene_ids</text>
  </g>
  <!-- produce arrow -->
  <path d="M744,270 L800,270" stroke="#76b900" stroke-width="2.4" marker-end="url(#c)"/>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/run_artifacts.py" target="_blank">
    <text x="772" y="262" text-anchor="middle" class="d" font-size="10.5">write_run_artifacts()</text>
  </a>
  <!-- OUTPUTS -->
  <text x="816" y="92" class="g" font-size="13">PER-RUN ARTIFACTS</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/run_artifacts.py" target="_blank">
    <rect x="812" y="120" width="324" height="80" rx="10" fill="#1a1a1a" stroke="#58a6ff" stroke-width="1.6"/>
    <text x="830" y="148" class="t" font-size="14">resolved_config.yaml</text>
    <text x="830" y="170" class="d" font-size="12">the frozen run — re-loaded verbatim</text>
    <text x="830" y="190" class="d" font-size="12">when a Slurm job re-runs on the node</text>
  </a>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/run_artifacts.py" target="_blank">
    <rect x="812" y="212" width="324" height="80" rx="10" fill="#1a1a1a" stroke="#58a6ff" stroke-width="1.6"/>
    <text x="830" y="240" class="t" font-size="14">cosmos_config.toml</text>
    <text x="830" y="262" class="d" font-size="12">launcher config consumed by</text>
    <text x="830" y="282" class="d" font-size="12">Cosmos-RL (mode, replicas, GRPO)</text>
  </a>
  <rect x="812" y="304" width="324" height="74" rx="10" fill="#1a1a1a" stroke="#2a2a2a"/>
  <text x="830" y="332" class="t" font-size="14">cosmos_config.toml</text>
  <text x="830" y="354" class="d" font-size="12">minimal HF config for the placeholder</text>
  <text x="830" y="372" class="d" font-size="12">model Cosmos loads on the trainer side</text>
  <!-- inspect tip -->
  <rect x="24" y="436" width="1112" height="56" rx="10" fill="#0d1117" stroke="#2a2a2a"/>
  <text x="44" y="470" class="d" font-size="14">Inspect the fully-resolved config before launching:  <tspan class="m" font-size="13">uv run --package alpagym-host -m alpagym_host.cli --cfg job</tspan></text>
</svg>
</div>

---

<a class="slide-id" id="34dc403b-5e99-4f9b-bfca-8fffd40d0324"></a>

## Bringing up the simulator

<div class="columns">
<div>

### Wizard lifecycle — [`alpasim_wizard.py`][wizard]
- `start_wizard()` spawns AlpaSim in the resolved checkout
- `wait_for_runtime_ready()` polls for the generated `runtime-server.yaml` **and** TCP connectivity
- A `finally` block guarantees the process is terminated

</div>
<div>

### Service discovery — [`endpoint_registry.py`][topo]
- A tiny **file-backed registry** (`FileTopologyRegistry`): endpoints written as YAML
- `publish_alpasim_runtime()` advertises `host:port`
- The runtime later `acquire_alpasim_runtime()`s it to open its gRPC channel
- No service mesh — just files in the run dir

</div>
</div>

> The AlpaSim checkout itself is resolved & cached (keyed by `repo_url@ref`) by [`alpasim_dependency.py`][dep], which also compiles protos and `uv sync`s the simulator.

---

<a class="slide-id" id="f8b44b57-e1e1-4b28-b2d6-2aeb307addd1"></a>

## Submitting to Slurm

[`slurm.py`][slurm] turns a prepared run into a cluster job:

<div class="columns">
<div>

### Submit — [`submit_slurm_job()`][submitjob]
- Writes an `sbatch` file and submits it
- The job body re-invokes `alpagym-host command=run` with `execution.resolved_config_path=...`
- The node reruns the **same** frozen config

</div>
<div>

### Containers — [`prepare_container_image()`][prepimg]
- Docker image URIs are converted to **`.sqsh`** via `enroot import`, cached on shared storage
- `build_cosmos_srun_command()` / `build_wizard_srun_command()` add `--container-image`, mounts & env
- Credentials validated before submit

</div>
</div>

---

<a class="slide-id" id="31fea2a9-493b-4d6c-ac98-934fd5836ca2"></a>

## What a run leaves behind

[`run_artifacts.py`][writeart] builds a timestamped run directory and writes the hand-off files:

| Artifact | Purpose |
|----------|---------|
| `resolved_config.yaml` | The frozen run — re-loaded verbatim by a Slurm node |
| `cosmos_config.toml` | Launcher config for Cosmos-RL (mode, replicas, GRPO) |
| `topology/`, `logs/`, `alpasim/` | Endpoint registry, logs, simulator output |

> One run = one self-contained directory. That's what makes a Slurm job reproducible from the login node.

---

<a class="slide-id" id="6feed53a-0d68-4358-b48a-230fc9d49dc9"></a>

<!-- _class: invert -->

# The RL **Runtime**

<p><code>packages/runtime</code> — the harness, inside the Cosmos-RL container</p>

---

<a class="slide-id" id="afd42948-1605-40ab-a79a-8f5814f242e2"></a>

## What is Cosmos-RL?

[**Cosmos-RL**](https://github.com/nvidia-cosmos/cosmos-rl) is NVIDIA's open-source, distributed **RL & post-training framework for Physical AI** — LLMs, world-foundation models and **vision-language-action** policies. AlpaGym is built *on top of* it.

<div class="columns">
<div>

### Decoupled, single-controller design
- A **controller** coordinates everything
- **Policy replicas** train; **rollout replicas** generate experience — scaled independently
- Replicas are wired by dynamic **NCCL** process groups; **Redis** is the control bus
- PyTorch-native · TP / FSDP / CP / PP parallelism · FP8

</div>
<div>

### Algorithms & extensibility
- SFT + RL: **GRPO**, DAPO (LLM); FlowGRPO, DiffusionNFT (world models); VLA GRPO/SFT
- A **registry / plugin** model: swap in your own model, rollout backend, trainer, reward fns
- [github.com/nvidia-cosmos/cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) · [docs][cosmosdocs]

</div>
</div>

> AlpaGym registers its driving-specific pieces and hands control to the launcher — the host starts it with `cosmos_rl.launcher.launch_all` (see [the control plane](#the-control-plane)).

---

<a class="slide-id" id="d799f86b-4944-42f7-a2f1-4ff24673c35b"></a>

## Cosmos-RL architecture

The [controller][cosmosctl] owns the dataset and orchestrates **policy** (trainer) and **rollout** (actor) replicas over a **Redis** bus; trained weights sync policy → rollout over **NCCL**.

<div style="display:flex;justify-content:center;margin-top:0.4rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="980" height="507" style="max-width:100%" font-family="Inter, sans-serif">
  <defs>
    <marker id="th" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#58a6ff"/></marker>
    <marker id="thr" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#58a6ff"/></marker>
    <marker id="nc" markerWidth="10" markerHeight="10" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#76b900"/></marker>
    <marker id="g" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#888"/></marker>
  </defs>
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <style>.t{fill:#f0f0f0;font-weight:700}.m{font-family:'JetBrains Mono',monospace;fill:#a3d44a}.d{fill:#a0a0a0}.blue{fill:#58a6ff}</style>
  <!-- Dataset -->
  <rect x="28" y="92" width="180" height="78" rx="11" fill="#1a1a1a" stroke="#2a2a2a"/>
  <text x="118" y="126" text-anchor="middle" class="t" font-size="15">Scene dataset</text>
  <text x="118" y="150" text-anchor="middle" class="d" font-size="12.5">prompts</text>
  <!-- Controller -->
  <a href="https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/dispatcher/controller.py" target="_blank">
    <rect x="338" y="52" width="486" height="158" rx="13" fill="#161616" stroke="#76b900" stroke-width="2"/>
  </a>
  <text x="358" y="84" class="t" font-size="18">Controller</text>
  <text x="358" y="106" class="d" font-size="13">single coordinator · dispatcher · data fetcher · shard mapper</text>
  <text x="358" y="126" class="d" font-size="12">starts its own Redis ▸</text>
  <!-- Redis sub-box -->
  <a href="https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/utils/redis_stream.py" target="_blank">
    <rect x="600" y="120" width="206" height="74" rx="9" fill="#0d1117" stroke="#58a6ff" stroke-width="1.4"/>
    <text x="703" y="148" text-anchor="middle" class="blue" font-size="14" font-weight="700">Redis streams</text>
    <text x="703" y="172" text-anchor="middle" class="d" font-size="11.5">command + rollout bus</text>
  </a>
  <!-- launcher note -->
  <a href="https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/launcher/launch_all.py" target="_blank">
    <text x="28" y="210" class="d" font-size="12">▸ <tspan class="m" font-size="11.5">launch_all.py</tspan> brings up the controller + all replicas</text>
  </a>
  <!-- Rollout replicas -->
  <a href="https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/rollout/rollout_base.py" target="_blank">
    <rect x="300" y="398" width="300" height="150" rx="13" fill="#1a1a1a" stroke="#76b900" stroke-width="1.8"/>
  </a>
  <text x="450" y="432" text-anchor="middle" class="t" font-size="17">Rollout replicas</text>
  <text x="450" y="456" text-anchor="middle" class="d" font-size="13">actors — generate + reward</text>
  <text x="450" y="478" text-anchor="middle" class="d" font-size="12.5">vLLM / TRT-LLM / custom backend</text>
  <text x="450" y="522" text-anchor="middle" class="m" font-size="12">@RolloutRegistry.register</text>
  <!-- Policy replicas -->
  <a href="https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/launcher/worker_entry.py" target="_blank">
    <rect x="700" y="398" width="300" height="150" rx="13" fill="#1a1a1a" stroke="#76b900" stroke-width="1.8"/>
  </a>
  <text x="850" y="432" text-anchor="middle" class="t" font-size="17">Policy replicas</text>
  <text x="850" y="456" text-anchor="middle" class="d" font-size="13">trainers — optimizer step</text>
  <text x="850" y="478" text-anchor="middle" class="d" font-size="12.5">GRPO · TP / FSDP / CP / PP</text>
  <text x="850" y="522" text-anchor="middle" class="m" font-size="12">@TrainerRegistry.register</text>
  <!-- dataset -> controller -->
  <path d="M208,131 L336,131" stroke="#888" stroke-width="2" marker-end="url(#g)"/>
  <text x="270" y="123" text-anchor="middle" class="d" font-size="11">prompts</text>
  <!-- controller <-> rollout (redis) -->
  <path d="M430,212 L430,396" stroke="#58a6ff" stroke-width="1.8" marker-end="url(#th)"/>
  <text x="392" y="310" text-anchor="middle" class="blue" font-size="11" transform="rotate(-90 392 310)">commands</text>
  <path d="M470,396 L470,212" stroke="#58a6ff" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#thr)"/>
  <text x="512" y="310" text-anchor="middle" class="blue" font-size="11" transform="rotate(-90 512 310)">rollouts + rewards</text>
  <!-- controller <-> policy (redis) -->
  <path d="M760,212 L850,396" stroke="#58a6ff" stroke-width="1.8" marker-end="url(#th)"/>
  <text x="716" y="306" class="blue" font-size="11">batches</text>
  <!-- policy -> rollout NCCL weight sync (thick) -->
  <path d="M698,475 L602,475" stroke="#76b900" stroke-width="5" marker-end="url(#nc)"/>
  <text x="650" y="462" text-anchor="middle" class="t" font-size="12">weight sync</text>
  <text x="650" y="538" text-anchor="middle" class="d" font-size="11">NCCL · IB / NVLink</text>
  <!-- legend -->
  <rect x="28" y="252" width="300" height="96" rx="10" fill="#0d1117" stroke="#2a2a2a"/>
  <line x1="48" y1="284" x2="92" y2="284" stroke="#58a6ff" stroke-width="2"/>
  <text x="104" y="288" class="d" font-size="12.5">Redis — control + rollout data queue</text>
  <line x1="48" y1="318" x2="92" y2="318" stroke="#76b900" stroke-width="5"/>
  <text x="104" y="322" class="d" font-size="12.5">NCCL — high-bandwidth weight sync</text>
  <!-- modes callout -->
  <a href="https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/launcher/launch_all.py" target="_blank">
    <rect x="844" y="232" width="288" height="120" rx="11" fill="#161616" stroke="#bc8cff" stroke-width="1.5"/>
  </a>
  <text x="864" y="262" fill="#bc8cff" font-size="14" font-weight="700">Placement modes</text>
  <text x="864" y="288" class="d" font-size="12.5"><tspan fill="#f0f0f0">disaggregated</tspan> — separate GPUs</text>
  <text x="864" y="310" class="d" font-size="12.5"><tspan fill="#f0f0f0">colocated</tspan> — share GPUs</text>
  <text x="864" y="336" class="d" font-size="12">AlpaGym runs <tspan class="m" font-size="11.5">colocated</tspan> (n_policy = n_rollout)</text>
</svg>
</div>

---

<a class="slide-id" id="0356e794-710f-41c5-9472-a24f677823b1"></a>

## How the Cosmos-RL loop runs

<div class="columns">
<div>

1. **Launch** — [`launch_all`][cosmoslaunch] reads `cosmos_config.toml`, places replicas, starts the controller (which spawns **Redis**)
2. **Register** — each replica runs the entrypoint, firing the `@…Registry.register` decorators, then calls `worker_entry.main(...)`
3. **Generate** — the controller dispatches prompts; rollout replicas produce trajectories and **rewards**

</div>
<div>

4. **Optimize** — batches go to policy replicas; the **GRPO** trainer takes a gradient step
5. **Weight sync** — updated weights broadcast policy → rollout over NCCL, so the next round is on-policy
6. **Repeat** until `max_num_steps`

</div>
</div>

> **GRPO** (Group Relative Policy Optimization) is a critic-free PPO variant: sample a *group* of trajectories per prompt and normalize their rewards into advantages — no value network. AlpaGym runs **colocated** (policy + rollout share GPUs).

---

<a class="slide-id" id="4cfeec69-1339-4877-a901-ab4f1622bf70"></a>

## Plugging into Cosmos-RL

Two ways to plug in: **decorator registries** (keyed by a string referenced in `cosmos_config.toml`) and **callbacks** handed to `worker_entry.main(...)`. AlpaGym wires all of it in [`cosmos/entrypoint.py`][entrypoint] — importing its modules fires the decorators, then it calls the launcher.

<div class="columns">
<div>

### Registered by string key
- `@RolloutRegistry.register("alpagym_rollout")` — [`rollout_backend.py`][rollout]
- `@TrainerRegistry.register("alpagym_trainer")` — [`trainer.py`][trainer]

</div>
<div>

### Passed to `worker_entry.main`
- **Dataset** of scenes — [`dataset.py`][dataset]
- **Data packer** — [`packer.py`][packer]
- **Reward fns** — [`reward_fn.py`][rewardfn]

</div>
</div>

> The actual driving model does **not** live in Cosmos's model slot — it runs inside the rollout backend's [inference engine](#policy-and-inference), loaded by the selected [policy bundle](#the-plugin-boundary).

---

<a class="slide-id" id="f4ac192b-fc5d-4a99-9371-9f14e67bb824"></a>

## The RL data flow

Scenes become prompts, the **actor** drives them into episodes, episodes are scored and persisted, and the learner consumes them.

<div style="display:flex;justify-content:center;margin-top:0.4rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="980" height="507" style="max-width:100%" font-family="Inter, sans-serif">
  <defs>
    <marker id="m" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#76b900"/></marker>
  </defs>
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <style>
    .t{fill:#f0f0f0;font-weight:700}.m{font-family:'JetBrains Mono',monospace;fill:#a3d44a}.d{fill:#a0a0a0}.lnk{fill:#58a6ff}
  </style>
  <text x="30" y="64" class="d" font-size="15"><tspan fill="#76b900" font-weight="700">①  Generate rollouts</tspan>  —  the actor collects experience by driving the simulator</text>
  <!-- TOP ROW -->
  <g>
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/dataset.py" target="_blank">
      <rect x="30" y="90" width="240" height="104" rx="12" fill="#1a1a1a" stroke="#2a2a2a"/>
      <text x="48" y="124" class="t" font-size="17">Scene dataset</text>
      <text x="48" y="148" class="m" font-size="12.5">cosmos/dataset.py</text>
      <text x="48" y="174" class="d" font-size="13">scene_ids → "prompts"</text>
    </a>
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/entrypoint.py" target="_blank">
      <rect x="300" y="90" width="240" height="104" rx="12" fill="#1a1a1a" stroke="#2a2a2a"/>
      <text x="318" y="124" class="t" font-size="17">Cosmos dispatcher</text>
      <text x="318" y="148" class="m" font-size="12.5">cosmos/entrypoint.py</text>
      <text x="318" y="174" class="d" font-size="13">queues + fans out prompts</text>
    </a>
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/rollout_backend.py" target="_blank">
      <rect x="570" y="90" width="240" height="104" rx="12" fill="#1a1a1a" stroke="#76b900" stroke-width="2"/>
      <text x="588" y="124" class="t" font-size="17">Rollout backend</text>
      <text x="588" y="148" class="m" font-size="12.5">cosmos/rollout_backend.py</text>
      <text x="588" y="174" class="d" font-size="13">the actor · prompt→scene_id</text>
    </a>
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/episode_runner/streaming_worker.py" target="_blank">
      <rect x="840" y="90" width="240" height="104" rx="12" fill="#1a1a1a" stroke="#76b900" stroke-width="2"/>
      <text x="858" y="124" class="t" font-size="17">run_batch()</text>
      <text x="858" y="148" class="m" font-size="12.5">episode_runner/streaming_worker.py</text>
      <text x="858" y="174" class="d" font-size="13">runs N×G sim episodes</text>
    </a>
    <path d="M270,142 L298,142" fill="none" stroke="#76b900" stroke-width="2.5" marker-end="url(#m)"/>
    <path d="M540,142 L568,142" fill="none" stroke="#76b900" stroke-width="2.5" marker-end="url(#m)"/>
    <path d="M810,142 L838,142" fill="none" stroke="#76b900" stroke-width="2.5" marker-end="url(#m)"/>
  </g>
  <!-- closed-loop note under run_batch -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/driver_server.py" target="_blank">
    <rect x="660" y="232" width="420" height="74" rx="10" fill="#0d1117" stroke="#58a6ff" stroke-width="1.3" stroke-dasharray="5 4"/>
    <text x="870" y="262" text-anchor="middle" class="lnk" font-size="14" font-weight="700">each episode = a closed loop with AlpaSim</text>
    <text x="870" y="286" text-anchor="middle" class="d" font-size="12.5">egodriver server · policy.step() · inference engine  (see next figure)</text>
  </a>
  <path d="M960,194 L960,230" fill="none" stroke="#58a6ff" stroke-width="2" marker-end="url(#m)"/>
  <!-- turn-down from run_batch to bottom row -->
  <path d="M960,306 C960,350 960,350 960,398" fill="none" stroke="#76b900" stroke-width="2.5" marker-end="url(#m)"/>
  <text x="975" y="356" class="d" font-size="12.5">EpisodeOutput</text>
  <text x="30" y="372" class="d" font-size="15"><tspan fill="#76b900" font-weight="700">②  Score &amp; train</tspan>  —  reward the episode, persist it, feed the learner  (read right → left)</text>
  <!-- BOTTOM ROW (right to left) -->
  <g>
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/rewards/compute.py" target="_blank">
      <rect x="840" y="398" width="240" height="104" rx="12" fill="#1a1a1a" stroke="#76b900" stroke-width="2"/>
      <text x="858" y="432" class="t" font-size="17">compute_reward()</text>
      <text x="858" y="456" class="m" font-size="12.5">rewards/compute.py</text>
      <text x="858" y="482" class="d" font-size="13">Σ reward terms → reward</text>
    </a>
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/disk.py" target="_blank">
      <rect x="570" y="398" width="240" height="104" rx="12" fill="#1a1a1a" stroke="#2a2a2a"/>
      <text x="588" y="432" class="t" font-size="17">JSON artifact</text>
      <text x="588" y="456" class="m" font-size="12.5">transport/disk.py</text>
      <text x="588" y="482" class="d" font-size="13">write_rollout_artifact()</text>
    </a>
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/reward_fn.py" target="_blank">
      <rect x="300" y="398" width="240" height="104" rx="12" fill="#1a1a1a" stroke="#2a2a2a"/>
      <text x="318" y="432" class="t" font-size="17">reward_fn</text>
      <text x="318" y="456" class="m" font-size="12.5">cosmos/reward_fn.py</text>
      <text x="318" y="482" class="d" font-size="13">reads episode.reward.total</text>
    </a>
    <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/trainer.py" target="_blank">
      <rect x="30" y="398" width="240" height="104" rx="12" fill="#1a1a1a" stroke="#58a6ff" stroke-width="2"/>
      <text x="48" y="432" class="t" font-size="17">step_training()</text>
      <text x="48" y="456" class="m" font-size="12.5">cosmos/trainer.py</text>
      <text x="48" y="482" class="d" font-size="13">GRPO update · stub today</text>
    </a>
    <path d="M838,450 L812,450" fill="none" stroke="#76b900" stroke-width="2.5" marker-end="url(#m)"/>
    <path d="M568,450 L542,450" fill="none" stroke="#76b900" stroke-width="2.5" marker-end="url(#m)"/>
    <path d="M298,450 L272,450" fill="none" stroke="#76b900" stroke-width="2.5" marker-end="url(#m)"/>
  </g>
  <text x="30" y="560" class="d" font-size="12.5">Registered with Cosmos-RL at import in <tspan class="m" font-size="12">cosmos/entrypoint.py</tspan>: the rollout backend, trainer, model, data packer and reward fns.</text>
</svg>
</div>

---

<a class="slide-id" id="55ab5d8c-230d-463e-900d-40bc190ec001"></a>

## The actor: rollout backend

[`AlpagymRollout`][rollout] is Cosmos's "actor". Its key method, [`rollout_generation()`](https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/rollout_backend.py):

<div class="columns">
<div>

### Setup (once per worker)
- `init_engine()` builds the batched **inference engine**, the **policy factory**, and the **egodriver** gRPC server
- Reads the AlpaSim endpoint from the topology registry

</div>
<div>

### Per `rollout_generation`
- Maps Cosmos payloads → **scene IDs**
- `submit_payload(...)` each to the [streaming worker][runbatch], then awaits their `Future`s
- Wraps the returned `.pt` paths into Cosmos `RolloutResult`s

</div>
</div>

> This is where the simulator, the policy and the GPU all meet — unpacked in [The Episode Loop](#the-episode-loop).

---

<a class="slide-id" id="b88cb15a-2cce-4449-b2ae-5bdbdb0a7f96"></a>

## Inside one node: compute & rollout flow

<div style="display:flex;justify-content:center;margin-top:0.4rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="980" height="507" style="max-width:100%" font-family="Inter, sans-serif">
  <defs>
    <marker id="pu" markerWidth="9" markerHeight="9" refX="6.5" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#bc8cff"/></marker>
    <marker id="b" markerWidth="9" markerHeight="9" refX="6.5" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#58a6ff"/></marker>
    <marker id="am" markerWidth="9" markerHeight="9" refX="6.5" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#fbbf24"/></marker>
    <marker id="g" markerWidth="9" markerHeight="9" refX="6.5" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#76b900"/></marker>
  </defs>
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <style>.t{fill:#f0f0f0;font-weight:700}.m{font-family:'JetBrains Mono',monospace;fill:#a3d44a}.d{fill:#a0a0a0}</style>
  <text x="22" y="22" class="d" font-size="13.5">Example: one 8-GPU node split <tspan fill="#f0f0f0" font-weight="700">1 policy / 3 rollout / 4 AlpaSim</tspan> (<a href="https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/conf/topology/slurm_full_node_1_3_4.yaml" target="_blank"><tspan class="m" font-size="12" text-decoration="underline">topology=slurm_full_node_1_3_4 ↗</tspan></a>)</text>
  <!-- CPU coordination strip -->
  <rect x="24" y="34" width="1112" height="28" rx="8" fill="#191919" stroke="#7a7a7a" stroke-width="1.2"/>
  <text x="580" y="52" text-anchor="middle" class="d" font-size="11"><tspan fill="#cfcfcf" font-weight="700">CPU only — coordination, no GPU:</tspan>   Cosmos-RL controller · Redis bus · AlpaSim controller procs · the StreamingRolloutWorker queue + threads</text>
  <!-- ZONE A: policy / training -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/trainer.py" target="_blank"><rect x="24" y="78" width="290" height="444" rx="12" fill="#1a1413" stroke="#f85149" stroke-width="1.8"/><text x="40" y="104" class="t" font-size="14">Policy</text></a>
  <text x="40" y="120" class="d" font-size="10.5">Cosmos role: the <tspan fill="#f0f0f0" font-weight="700">trainer</tspan> · GPU 0 · ×1</text>
  <rect x="40" y="132" width="110" height="24" rx="6" fill="#2a1414" stroke="#f85149"/><text x="95" y="148" text-anchor="middle" class="m" font-size="10.5" fill="#ff9a93">GPU 0</text>
  <text x="40" y="184" class="t" font-size="12" fill="#ff9a93">GRPO training step</text>
  <text x="40" y="206" class="d" font-size="11">• backprop — the heaviest</text>
  <text x="40" y="222" class="d" font-size="11">  GPU compute in the loop</text>
  <text x="40" y="248" class="d" font-size="11">• holds the master weights</text>
  <text x="40" y="270" class="d" font-size="11">• scales out: DP shards</text>
  <text x="40" y="286" class="d" font-size="10.5">  (<tspan class="m" font-size="10">n_init × dp_shard_size</tspan>)</text>
  <rect x="40" y="482" width="200" height="26" rx="7" fill="#2a1414" stroke="#f85149"/><text x="52" y="499" class="t" font-size="10.5" fill="#ff9a93">⚙ compute: TRAINING</text>
  <!-- ZONE B: rollout / inference -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/rollout_backend.py" target="_blank"><rect x="336" y="78" width="462" height="444" rx="12" fill="#121512" stroke="#76b900" stroke-width="2"/><text x="352" y="104" class="t" font-size="14">Rollout</text></a>
  <text x="352" y="120" class="d" font-size="10.5">Cosmos role: the <tspan fill="#f0f0f0" font-weight="700">actors</tspan> · GPUs 1-3 · ×3 (example)</text>
  <rect x="352" y="132" width="124" height="24" rx="6" fill="#16200f" stroke="#76b900"/><text x="414" y="148" text-anchor="middle" class="m" font-size="10.5">GPU 1 · replica</text>
  <rect x="484" y="132" width="124" height="24" rx="6" fill="#16200f" stroke="#76b900"/><text x="546" y="148" text-anchor="middle" class="m" font-size="10.5">GPU 2 · replica</text>
  <rect x="616" y="132" width="124" height="24" rx="6" fill="#16200f" stroke="#76b900"/><text x="678" y="148" text-anchor="middle" class="m" font-size="10.5">GPU 3 · replica</text>
  <rect x="352" y="166" width="438" height="174" rx="9" fill="#161616" stroke="#3a5a1a"/>
  <text x="366" y="185" class="t" font-size="11.5">inside ONE rollout replica — its N threads share ONE model:</text>
  <text x="366" y="200" class="d" font-size="9">StreamingRolloutWorker: a fixed pool of N=12 CPU threads (N = AlpaSim capacity; 3 shown)</text>
  <rect x="366" y="208" width="170" height="38" rx="7" fill="#1b1712" stroke="#f0883e"/><text x="376" y="224" class="t" font-size="10">thread 0 · scene A</text><text x="376" y="238" class="d" font-size="9">waiting on AlpaSim (gRPC)</text>
  <rect x="366" y="250" width="170" height="38" rx="7" fill="#1b1712" stroke="#f0883e"/><text x="376" y="266" class="t" font-size="10">thread 1 · scene B</text><text x="376" y="280" class="d" font-size="9">waiting on the model</text>
  <rect x="366" y="292" width="170" height="38" rx="7" fill="#1b1712" stroke="#f0883e"/><text x="376" y="308" class="t" font-size="10">thread 2 · scene C</text><text x="376" y="322" class="d" font-size="9">building the ModelInput</text>
  <path d="M536,227 L558,250" stroke="#76b900" stroke-width="1.6" marker-end="url(#g)"/>
  <path d="M536,269 L558,269" stroke="#76b900" stroke-width="1.6" marker-end="url(#g)"/>
  <path d="M536,311 L558,288" stroke="#76b900" stroke-width="1.6" marker-end="url(#g)"/>
  <text x="540" y="243" class="d" font-size="8" fill="#a3d44a">infer()</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/inference/inference_engine.py" target="_blank"><rect x="560" y="224" width="222" height="88" rx="7" fill="#16200f" stroke="#76b900" stroke-width="2"/><text x="671" y="242" text-anchor="middle" class="t" font-size="11" fill="#cfe3a0">shared model — 1 copy</text><text x="671" y="258" text-anchor="middle" class="d" font-size="9">the only GPU worker here:</text><text x="671" y="272" text-anchor="middle" class="d" font-size="9">stacks the threads' requests</text><text x="671" y="286" text-anchor="middle" class="d" font-size="9">into ONE forward pass, then</text><text x="671" y="300" text-anchor="middle" class="d" font-size="9">resolves each thread's Future</text></a>
  <text x="366" y="360" class="d" font-size="9.5"><tspan fill="#f0f0f0" font-weight="700">What a thread does</tspan> — a thin controller, not a compute unit:</text>
  <text x="366" y="376" class="d" font-size="9.5">pull a scene (from Cosmos) → <tspan class="m" font-size="9">simulate()</tspan> starts the session → then <tspan fill="#f0f0f0" font-weight="700">every</tspan></text>
  <text x="366" y="390" class="d" font-size="9.5"><tspan fill="#f0f0f0" font-weight="700">tick</tspan>: AlpaSim asks for an action → build input → <tspan class="m" font-size="9">infer()</tspan> on the shared model</text>
  <text x="366" y="404" class="d" font-size="9.5">(block on Future) → return trajectory. Between ticks it just <tspan fill="#f0f0f0" font-weight="700">waits</tspan> on AlpaSim.</text>
  <text x="366" y="418" class="d" font-size="9.5">The shared model batches all active rollouts' <tspan class="m" font-size="9">infer()</tspan> into 1 GPU forward.</text>
  <text x="366" y="434" class="d" font-size="9.5">(Each replica keeps its <tspan fill="#ff9a93" font-weight="700">OWN</tspan> weights, synced from policy via <tspan class="m" font-size="9">NCCL</tspan>.)</text>
  <text x="366" y="452" class="d" font-size="9.5"><tspan fill="#f0883e" font-weight="700">Why 36 threads (3×12) for 12 slots?</tspan> Deliberate over-provisioning:</text>
  <text x="366" y="466" class="d" font-size="9.5">threads are cheap (I/O-blocked), so AlpaSim's 12 slots are always kept full.</text>
  <rect x="352" y="482" width="278" height="26" rx="7" fill="#16200f" stroke="#76b900"/><text x="364" y="499" class="t" font-size="10.5" fill="#cfe3a0">⚙ compute: INFERENCE (batched forward)</text>
  <!-- ZONE C: AlpaSim / render+physics -->
  <a href="https://github.com/NVlabs/alpasim" target="_blank"><rect x="820" y="78" width="316" height="444" rx="12" fill="#10161f" stroke="#58a6ff" stroke-width="1.8"/><text x="836" y="104" class="t" font-size="14">AlpaSim</text></a>
  <text x="836" y="120" class="d" font-size="10">the environment · <tspan fill="#f0f0f0" font-weight="700">not a Cosmos replica</tspan> · GPUs 4-7</text>
  <rect x="836" y="132" width="68" height="24" rx="6" fill="#11202f" stroke="#58a6ff"/><text x="870" y="148" text-anchor="middle" class="m" font-size="10">GPU 4</text>
  <rect x="912" y="132" width="68" height="24" rx="6" fill="#11202f" stroke="#58a6ff"/><text x="946" y="148" text-anchor="middle" class="m" font-size="10">GPU 5</text>
  <rect x="988" y="132" width="68" height="24" rx="6" fill="#11202f" stroke="#58a6ff"/><text x="1022" y="148" text-anchor="middle" class="m" font-size="10">GPU 6</text>
  <rect x="1064" y="132" width="60" height="24" rx="6" fill="#11202f" stroke="#58a6ff"/><text x="1094" y="148" text-anchor="middle" class="m" font-size="10">GPU 7</text>
  <text x="836" y="167" class="d" font-size="9">every rollout flows through all of these each tick (instances × each):</text>
  <rect x="836" y="172" width="288" height="38" rx="7" fill="#11202f" stroke="#3a4a5a"/><text x="848" y="188" class="t" font-size="10.5" fill="#cfe3ff">sensorsim / NRE — render cameras (GPU)</text><text x="848" y="202" class="d" font-size="9">4 GPU instances handling 3 rollouts each = 12</text>
  <rect x="836" y="214" width="288" height="38" rx="7" fill="#11202f" stroke="#3a4a5a"/><text x="848" y="230" class="t" font-size="10.5" fill="#cfe3ff">physics — vehicle dynamics (GPU)</text><text x="848" y="244" class="d" font-size="9">4 GPU instances handling 3 rollouts each = 12</text>
  <rect x="836" y="256" width="288" height="38" rx="7" fill="#11202f" stroke="#3a4a5a"/><text x="848" y="272" class="t" font-size="10.5">controller — per-rollout control (CPU)</text><text x="848" y="286" class="d" font-size="9">6 CPU procs handling 2 rollouts each = 12</text>
  <text x="836" y="312" class="d" font-size="9">a rollout needs all 3 at once, so the tightest service</text>
  <text x="836" y="325" class="d" font-size="9">caps the total (all three = 12 here) — the bottleneck:</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/alpasim_configs/src/alpagym_alpasim_configs/configs/topology/alpagym_4gpu.yaml" target="_blank"><text x="836" y="341" class="t" font-size="10.5" fill="#cfe3ff" text-decoration="underline">12 concurrent rollouts ↗</text></a>
  <text x="836" y="362" class="d" font-size="9">Rollouts arrive via <tspan class="m" font-size="8.5">simulate()</tspan> on this ONE endpoint;</text>
  <text x="836" y="375" class="d" font-size="9">the 12 slots are shared first-come-first-served</text>
  <text x="836" y="388" class="d" font-size="9">across all replicas. AlpaSim then drives each</text>
  <text x="836" y="401" class="d" font-size="9">session back tick by tick, calling its egodriver.</text>
  <rect x="836" y="482" width="288" height="26" rx="7" fill="#11202f" stroke="#58a6ff"/><text x="848" y="499" class="t" font-size="10.5" fill="#cfe3ff">⚙ compute: NEURAL RENDER + PHYSICS</text>
  <!-- inter-zone arrows -->
  <path d="M316,212 L334,212" stroke="#bc8cff" stroke-width="2" marker-end="url(#pu)"/>
  <text x="300" y="200" class="d" font-size="9" fill="#cbb3ff" text-anchor="middle">weights</text>
  <path d="M334,300 L316,300" stroke="#fbbf24" stroke-width="2" marker-end="url(#am)"/>
  <text x="300" y="320" class="d" font-size="9" fill="#e8c98a" text-anchor="middle">.pt data</text>
  <path d="M800,212 L818,212" stroke="#58a6ff" stroke-width="2" marker-end="url(#b)"/>
  <path d="M818,300 L800,300" stroke="#58a6ff" stroke-width="2" marker-end="url(#b)"/>
  <text x="809" y="195" class="d" font-size="9" fill="#9cc6ff" text-anchor="middle">gRPC</text>
  <!-- footnote / legend -->
  <text x="22" y="544" class="d" font-size="10.5"><tspan fill="#f85149">▮ training</tspan>   <tspan fill="#76b900">▮ inference</tspan>   <tspan fill="#58a6ff">▮ render + physics</tspan>   <tspan fill="#7a7a7a">▮ CPU coordination</tspan>   ·   arrows: <tspan fill="#bc8cff">weights/NCCL</tspan> · <tspan fill="#58a6ff">gRPC</tspan> · <tspan fill="#fbbf24">rollout .pt</tspan></text>
  <text x="22" y="562" class="d" font-size="10.5">Only <tspan fill="#f0f0f0" font-weight="700">policy</tspan> + <tspan fill="#f0f0f0" font-weight="700">rollout</tspan> are Cosmos-RL replicas; <tspan fill="#f0f0f0" font-weight="700">AlpaSim</tspan> is the external environment they drive over gRPC.</text>
  <text x="22" y="580" class="d" font-size="10.5">Rollout actors are forward-only copies of the policy, kept in sync via NCCL. "×3" is this example; the default colocates 1 policy + 1 rollout.</text>
</svg>
</div>

---

<a class="slide-id" id="7b8921e8-50eb-4eeb-934b-4facd76ba423"></a>

## The trainer: GRPO <span class="tag tag-green">on main</span>

[`AlpagymGRPOTrainer`][trainer] subclasses Cosmos-RL's `GRPOTrainer` and runs a **real** optimization step — [`step_training()`][trainer] filters rollouts, builds minibatches and takes a gradient step. *(The earlier `AlpagymFakeTrainer` no-op stub is gone.)*

<div class="columns">
<div>

### Replay training, not re-sampling
- Rollout workers record a durable **replay contract** ([`replay.py`][replay]) — the *selected action* plus a model-family payload — into each artifact
- The trainer unpacks it via the [data packer][packer] and **recomputes logprobs** for that exact recorded action
- Applies the **GRPO / PPO surrogate** (`_compute_ppo_surrogate`) over those logprobs + group-relative advantages, with ratio clipping; optional **KL penalty** vs a refreshed reference model (`kl_beta`)

</div>
<div>

### Wired into Cosmos
- Updated policy weights **sync** policy → rollout (NCCL), so the next round is on-policy
- GRPO knobs live under `cosmos.train.train_policy` in [`conf/default.yaml`][defaults] — `grpo_ratio_clip_*`, `kl_beta`, `grpo_optimization_iterations`

</div>
</div>

---

<a class="slide-id" id="568902ab-9616-4821-bb01-2ba44300fd59"></a>

<!-- _class: invert -->

# The Episode **Loop**

<p>Where the simulator and the policy talk, step by step</p>

---

<a class="slide-id" id="491c8939-e5e0-40de-b62a-76184209b281"></a>

## What is AlpaSim?

**AlpaSim** is the **closed-loop driving simulator** AlpaGym drives — its own repo, [`NVlabs/alpasim`](https://github.com/NVlabs/alpasim). AlpaGym treats it as the *environment*: it never imports it, it talks to it over **gRPC**.

<div class="columns">
<div>

### What it provides
- Re-simulates a recorded **scene**: renders the **camera** observations and the ego state each step
- **Advances the ego car** along the trajectory the policy returns, then asks again
- Computes episode **metrics**
- Exposes a gRPC `RuntimeService` (the `simulate()` RPC) and calls back into our **egodriver**

</div>
<div>

### How AlpaGym uses it
- The host launches it through **Wizard** (AlpaSim's own launcher) — see [the control plane](#the-control-plane)
- The runtime opens the published gRPC endpoint and runs one **session** per scene × generation
- Conversion lives in [`proto_conversion.py`][proto]; the server is [`driver_server.py`][driver]

</div>
</div>

> **Closed-loop vs open-loop:** open-loop just replays ground-truth and scores predictions; **closed-loop** lets the policy *actually steer*, so mistakes compound over the episode — the realistic setting for evaluation and RL.

---

<a class="slide-id" id="3b4ecbbf-8f57-4159-8900-5320ae2538b6"></a>

## Inside AlpaSim — the *communication* view

A central **Runtime** (the *client*) calls each **gRPC microservice** in turn (①–⑤) per tick, logs the world state, and repeats. <span class="tag tag-orange">AlpaGym</span> plugs in as the **Driver ③**.

<div style="display:flex;justify-content:center;margin-top:0.2rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="860" height="445" style="max-width:100%" font-family="Inter, sans-serif">
  <defs>
    <marker id="ga" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#76b900"/></marker>
    <marker id="oa" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#f0883e"/></marker>
    <marker id="gy" markerWidth="10" markerHeight="10" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#888"/></marker>
  </defs>
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <style>
    .t{fill:#f0f0f0;font-weight:700}
    .m{font-family:'JetBrains Mono',monospace;fill:#a3d44a}
    .d{fill:#a0a0a0}
    .lbl{fill:#cfe0b0;font-family:'JetBrains Mono',monospace}
    .badge{fill:#76b900}
    .bnum{fill:#0a0a0a;font-weight:700;font-family:'JetBrains Mono',monospace}
  </style>
  <!-- WIZARD banner -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/wizard" target="_blank">
    <rect x="70" y="14" width="1020" height="56" rx="12" fill="#161616" stroke="#bc8cff" stroke-width="1.5" stroke-dasharray="6 4"/>
    <text x="580" y="40" text-anchor="middle" class="t" font-size="17">Wizard <tspan class="m" fill="#bc8cff" font-size="12">src/wizard</tspan> — configures the run &amp; launches every microservice</text>
    <text x="580" y="59" text-anchor="middle" class="d" font-size="12.5">tells the Runtime each service's address · every service scales to many gRPC-server replicas</text>
  </a>
  <!-- launch arrow -->
  <path d="M580,72 L580,248" stroke="#888" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#gy)"/>
  <text x="589" y="150" class="d" font-size="11.5">launches &amp; wires addresses</text>
  <!-- RUNTIME hub -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/runtime" target="_blank">
    <rect x="460" y="250" width="240" height="120" rx="16" fill="#161616" stroke="#76b900" stroke-width="2.6"/>
    <text x="580" y="288" text-anchor="middle" class="t" font-size="24">Runtime</text>
    <text x="580" y="312" text-anchor="middle" class="m" font-size="13">src/runtime</text>
    <text x="580" y="336" text-anchor="middle" class="d" font-size="13">orchestrator · central gRPC client</text>
    <text x="580" y="356" text-anchor="middle" class="d" font-size="12">owns world state · drives the loop</text>
  </a>
  <!-- (6) self-loop: log + advance -->
  <path d="M460,272 C402,260 402,302 460,294" fill="none" stroke="#76b900" stroke-width="2" marker-end="url(#ga)"/>
  <circle cx="408" cy="283" r="11" class="badge"/><text x="408" y="287" text-anchor="middle" class="bnum" font-size="12">6</text>
  <text x="392" y="316" text-anchor="end" class="d" font-size="11.5">log state, advance, repeat</text>
  <!-- TRAFFICSIM -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/trafficsim" target="_blank">
    <rect x="96" y="96" width="252" height="96" rx="12" fill="#161616" stroke="#76b900" stroke-width="1.8"/>
    <text x="222" y="126" text-anchor="middle" class="t" font-size="18">Trafficsim</text>
    <text x="222" y="148" text-anchor="middle" class="m" font-size="12.5">src/trafficsim</text>
    <text x="222" y="168" text-anchor="middle" class="d" font-size="12">actuates non-ego actors</text>
    <text x="222" y="184" text-anchor="middle" fill="#f0883e" font-size="11">neural traffic sim · coming soon</text>
  </a>
  <!-- NRE / SENSORSIM -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/grpc" target="_blank">
    <rect x="812" y="96" width="252" height="96" rx="12" fill="#161616" stroke="#76b900" stroke-width="1.8"/>
    <text x="938" y="124" text-anchor="middle" class="t" font-size="17">NRE / Sensorsim</text>
    <text x="938" y="146" text-anchor="middle" class="m" font-size="12">Sensorsim.render()</text>
    <text x="938" y="166" text-anchor="middle" class="d" font-size="12">neural rendering engine —</text>
    <text x="938" y="182" text-anchor="middle" class="d" font-size="12">renders the ego camera frames</text>
  </a>
  <!-- DRIVER (highlight: AlpaGym seam) -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/driver" target="_blank">
    <rect x="884" y="262" width="250" height="104" rx="12" fill="#1c1410" stroke="#f0883e" stroke-width="2.6"/>
    <text x="1009" y="290" text-anchor="middle" class="t" font-size="20">Driver</text>
    <text x="1009" y="312" text-anchor="middle" class="m" font-size="12.5">src/driver</text>
    <text x="1009" y="332" text-anchor="middle" class="d" font-size="12">runs the ego policy network</text>
    <text x="1009" y="353" text-anchor="middle" fill="#f0883e" font-weight="700" font-size="11.5">&#8592; AlpaGym's egodriver plugs in here</text>
  </a>
  <!-- CONTROLLER -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/controller" target="_blank">
    <rect x="648" y="470" width="250" height="96" rx="12" fill="#161616" stroke="#76b900" stroke-width="1.8"/>
    <text x="773" y="500" text-anchor="middle" class="t" font-size="18">Controller</text>
    <text x="773" y="522" text-anchor="middle" class="m" font-size="12.5">src/controller</text>
    <text x="773" y="542" text-anchor="middle" class="d" font-size="12">vehicle model — dynamics,</text>
    <text x="773" y="558" text-anchor="middle" class="d" font-size="12">uncorrected egomotion</text>
  </a>
  <!-- PHYSICS -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/physics" target="_blank">
    <rect x="262" y="470" width="250" height="96" rx="12" fill="#161616" stroke="#76b900" stroke-width="1.8"/>
    <text x="387" y="500" text-anchor="middle" class="t" font-size="18">Physics</text>
    <text x="387" y="522" text-anchor="middle" class="m" font-size="12.5">src/physics</text>
    <text x="387" y="542" text-anchor="middle" class="d" font-size="12">ground constraints —</text>
    <text x="387" y="558" text-anchor="middle" class="d" font-size="12">keeps actors on the road</text>
  </a>
  <!-- EVAL (offline) -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/eval" target="_blank">
    <rect x="36" y="262" width="200" height="104" rx="12" fill="#0d1117" stroke="#888" stroke-width="1.5" stroke-dasharray="6 4"/>
    <text x="136" y="290" text-anchor="middle" class="t" fill="#c8c8c8" font-size="17">Eval</text>
    <text x="136" y="312" text-anchor="middle" class="m" fill="#888" font-size="12">src/eval</text>
    <text x="136" y="332" text-anchor="middle" class="d" font-size="11.5">driving logs &#8658; AD metrics</text>
    <text x="136" y="351" text-anchor="middle" fill="#888" font-size="11">offline · outside the loop</text>
  </a>
  <!-- SPOKES (Runtime is the client; numbered per-tick call order) -->
  <!-- (1) trafficsim -->
  <path d="M482,252 L322,193" stroke="#76b900" stroke-width="2.2" marker-end="url(#ga)"/>
  <circle cx="400" cy="222" r="11" class="badge"/><text x="400" y="226" text-anchor="middle" class="bnum" font-size="12">1</text>
  <text x="408" y="244" text-anchor="middle" class="lbl" font-size="11">world state &#8596; non-ego actuation</text>
  <!-- (2) NRE -->
  <path d="M678,252 L838,193" stroke="#76b900" stroke-width="2.2" marker-end="url(#ga)"/>
  <circle cx="760" cy="222" r="11" class="badge"/><text x="760" y="226" text-anchor="middle" class="bnum" font-size="12">2</text>
  <text x="752" y="244" text-anchor="middle" class="lbl" font-size="11">world state &#8596; camera frames</text>
  <!-- (3) driver (orange seam) -->
  <path d="M700,304 L882,310" stroke="#f0883e" stroke-width="2.6" marker-end="url(#oa)"/>
  <circle cx="792" cy="307" r="11" fill="#f0883e"/><text x="792" y="311" text-anchor="middle" class="bnum" font-size="12">3</text>
  <text x="792" y="292" text-anchor="middle" class="lbl" fill="#f3b48a" font-size="11">sensor obs &#8596; ego actuation</text>
  <!-- (4) controller -->
  <path d="M662,370 L760,468" stroke="#76b900" stroke-width="2.2" marker-end="url(#ga)"/>
  <circle cx="712" cy="420" r="11" class="badge"/><text x="712" y="424" text-anchor="middle" class="bnum" font-size="12">4</text>
  <text x="724" y="424" text-anchor="start" class="lbl" font-size="11">actuation &#8596; egomotion</text>
  <!-- (5) physics -->
  <path d="M498,370 L400,468" stroke="#76b900" stroke-width="2.2" marker-end="url(#ga)"/>
  <circle cx="449" cy="420" r="11" class="badge"/><text x="449" y="424" text-anchor="middle" class="bnum" font-size="12">5</text>
  <text x="437" y="424" text-anchor="end" class="lbl" font-size="11">ego+non-ego &#8596; constrained</text>
  <!-- eval (offline log stream) -->
  <path d="M460,336 L238,322" stroke="#888" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#gy)"/>
  <text x="350" y="345" text-anchor="middle" class="d" font-size="10.5">log stream</text>
</svg>
</div>

<div style="position:absolute;left:4.5rem;right:9rem;bottom:1.05rem;font-size:0.72rem;color:#666;line-height:1.4">Source: public repo <a href="https://github.com/NVlabs/alpasim">NVlabs/alpasim</a> · <a href="https://github.com/NVlabs/alpasim/blob/main/docs/DESIGN.md">DESIGN.md</a> — tap any box to open it on GitHub.</div>

---

<a class="slide-id" id="2b9d330c-5905-451a-91b8-6a876204a704"></a>

## Inside AlpaSim — the *dependency* view

The same services as **per-tick data flow** — *who feeds whom*. A clockwise cycle: world state → sensors & traffic → **Driver** → controller → physics → **Runtime**; **Eval** taps the logs, off-loop.

<div style="display:flex;justify-content:center;margin-top:0.2rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="860" height="445" style="max-width:100%" font-family="Inter, sans-serif">
  <defs>
    <marker id="ga" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#76b900"/></marker>
    <marker id="oa" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#f0883e"/></marker>
    <marker id="gy" markerWidth="10" markerHeight="10" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#888"/></marker>
  </defs>
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <style>
    .t{fill:#f0f0f0;font-weight:700}
    .m{font-family:'JetBrains Mono',monospace;fill:#a3d44a}
    .d{fill:#a0a0a0}
    .lbl{fill:#cfe0b0;font-family:'JetBrains Mono',monospace}
  </style>
  <!-- AlpaSim container -->
  <rect x="60" y="150" width="1040" height="420" rx="22" fill="#181818" stroke="#2f2f2f" stroke-width="1.5"/>
  <text x="1078" y="556" text-anchor="end" fill="#76b900" opacity="0.15" font-size="50" font-weight="700">AlpaSim</text>
  <!-- DRIVER / ego policy (external seam, highlighted) -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/driver" target="_blank">
    <rect x="430" y="40" width="300" height="86" rx="14" fill="#1c1410" stroke="#f0883e" stroke-width="2.6"/>
    <text x="580" y="70" text-anchor="middle" class="t" font-size="19">Driver · ego policy</text>
    <text x="580" y="90" text-anchor="middle" class="m" font-size="12.5">src/driver</text>
    <text x="580" y="110" text-anchor="middle" fill="#f0883e" font-weight="700" font-size="11">AlpaGym's egodriver — cameras + ego history &#8594; trajectory</text>
  </a>
  <!-- NRE / Sensor sim (top-left) -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/grpc" target="_blank">
    <rect x="95" y="180" width="240" height="84" rx="12" fill="#161616" stroke="#76b900" stroke-width="1.8"/>
    <text x="215" y="208" text-anchor="middle" class="t" font-size="15.5">NRE / Sensor sim</text>
    <text x="215" y="229" text-anchor="middle" class="m" font-size="11.5">Sensorsim.render()</text>
    <text x="215" y="248" text-anchor="middle" class="d" font-size="11">renders ego camera frames</text>
  </a>
  <!-- Controller (top-right) -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/controller" target="_blank">
    <rect x="825" y="180" width="240" height="84" rx="12" fill="#161616" stroke="#76b900" stroke-width="1.8"/>
    <text x="945" y="208" text-anchor="middle" class="t" font-size="15.5">Controller</text>
    <text x="945" y="229" text-anchor="middle" class="m" font-size="11.5">src/controller</text>
    <text x="945" y="248" text-anchor="middle" class="d" font-size="11">vehicle dynamics &#8594; egomotion</text>
  </a>
  <!-- Trafficsim (mid-left) -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/trafficsim" target="_blank">
    <rect x="95" y="300" width="240" height="84" rx="12" fill="#161616" stroke="#76b900" stroke-width="1.8"/>
    <text x="215" y="328" text-anchor="middle" class="t" font-size="15.5">Trafficsim</text>
    <text x="215" y="349" text-anchor="middle" class="m" font-size="11.5">src/trafficsim</text>
    <text x="215" y="368" text-anchor="middle" fill="#f0883e" font-size="10.5">non-ego actors · coming soon</text>
  </a>
  <!-- Physics (mid-right) -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/physics" target="_blank">
    <rect x="825" y="300" width="240" height="84" rx="12" fill="#161616" stroke="#76b900" stroke-width="1.8"/>
    <text x="945" y="328" text-anchor="middle" class="t" font-size="15.5">Physics</text>
    <text x="945" y="349" text-anchor="middle" class="m" font-size="11.5">src/physics</text>
    <text x="945" y="368" text-anchor="middle" class="d" font-size="11">ground constraints</text>
  </a>
  <!-- Runtime (bottom-center) -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/runtime" target="_blank">
    <rect x="430" y="436" width="300" height="104" rx="14" fill="#161616" stroke="#76b900" stroke-width="2.6"/>
    <text x="580" y="472" text-anchor="middle" class="t" font-size="20">Runtime</text>
    <text x="580" y="496" text-anchor="middle" class="m" font-size="12.5">src/runtime</text>
    <text x="580" y="518" text-anchor="middle" class="d" font-size="11.5">owns world state · drives the loop</text>
  </a>
  <!-- Eval (bottom-left, offline) -->
  <a href="https://github.com/NVlabs/alpasim/tree/main/src/eval" target="_blank">
    <rect x="95" y="436" width="240" height="84" rx="12" fill="#0d1117" stroke="#888" stroke-width="1.5" stroke-dasharray="6 4"/>
    <text x="215" y="464" text-anchor="middle" class="t" fill="#c8c8c8" font-size="14.5">Eval · metrics</text>
    <text x="215" y="485" text-anchor="middle" class="m" fill="#888" font-size="11.5">src/eval</text>
    <text x="215" y="503" text-anchor="middle" class="d" font-size="10.5">offline · outside the loop</text>
  </a>
  <!-- ===== DATA-FLOW EDGES (clockwise cycle) ===== -->
  <!-- world-state feedback: Runtime -> rail -> Sensor sim & Trafficsim -->
  <path d="M430,514 C320,566 150,566 74,524" fill="none" stroke="#76b900" stroke-width="2"/>
  <path d="M74,524 L74,222 L93,222" fill="none" stroke="#76b900" stroke-width="2" marker-end="url(#ga)"/>
  <path d="M74,342 L93,342" fill="none" stroke="#76b900" stroke-width="2" marker-end="url(#ga)"/>
  <text x="62" y="312" text-anchor="middle" transform="rotate(-90 62 312)" fill="#76b900" font-size="11.5" font-weight="600">world state</text>
  <!-- Sensor sim -> Driver : camera frames -->
  <path d="M260,180 C320,148 382,120 428,104" fill="none" stroke="#76b900" stroke-width="2.2" marker-end="url(#ga)"/>
  <text x="352" y="136" text-anchor="middle" class="lbl" font-size="11">camera frames</text>
  <!-- Driver -> Controller : trajectory -->
  <path d="M732,104 C778,120 840,148 900,180" fill="none" stroke="#f0883e" stroke-width="2.4" marker-end="url(#oa)"/>
  <text x="808" y="136" text-anchor="middle" class="lbl" fill="#f3b48a" font-size="11">trajectory</text>
  <!-- Controller -> Physics : ego actuation -->
  <path d="M945,264 L945,298" stroke="#76b900" stroke-width="2.2" marker-end="url(#ga)"/>
  <text x="955" y="285" text-anchor="start" class="lbl" font-size="11">ego actuation</text>
  <!-- Trafficsim -> Physics : non-ego actuation -->
  <path d="M335,342 L823,342" stroke="#76b900" stroke-width="2.2" marker-end="url(#ga)"/>
  <text x="580" y="331" text-anchor="middle" class="lbl" font-size="11">non-ego actuation</text>
  <!-- Physics -> Runtime : constrained state -->
  <path d="M925,384 C900,442 800,476 732,486" fill="none" stroke="#76b900" stroke-width="2.2" marker-end="url(#ga)"/>
  <text x="824" y="430" text-anchor="middle" class="lbl" font-size="11">constrained state</text>
  <!-- Runtime -> Eval : logs (dashed) -->
  <path d="M430,498 L337,485" stroke="#888" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#gy)"/>
  <text x="384" y="478" text-anchor="middle" class="d" font-size="10.5">logs</text>
</svg>
</div>

<div style="position:absolute;left:4.5rem;right:9rem;bottom:1.05rem;font-size:0.72rem;color:#666;line-height:1.4">Source: public repo <a href="https://github.com/NVlabs/alpasim">NVlabs/alpasim</a> — mirrors the "Dependencies" panel of <a href="https://github.com/NVlabs/alpasim/blob/main/docs/DESIGN.md">DESIGN.md</a>.</div>

---

<a class="slide-id" id="6fe26d12-94a9-4548-ba4e-bf98a35e5c49"></a>

## Dispatching rollouts — the streaming worker

On `main`, [`StreamingRolloutWorker`][runbatch] runs **one `simulate(1)` per rollout**, not a batch:

1. `submit_payload(payload)` — **idempotent** by `prompt_idx`; creates a per-payload `Future` and enqueues `rollouts_per_payload` sibling jobs
2. A `PriorityQueue` feeds **N `alpagym-sim-{i}` threads** (N = the AlpaSim runtime's capacity)
3. Each thread fires `simulate(1)` (pinning a `session_uuid`), drains the `SessionRecord`, computes [reward](#rewards), writes a `.pt` artifact
4. **Retries preempt** fresh work (priority 0); after `n_target` siblings land, the `Future` resolves

> Cosmos **prefetches** (`enqueue_prefetch_payloads`) so the next batch's sims start while the current one is still in flight. A batch only exists at the `rollout_generation` boundary — under the hood, every rollout is an independent queue-driven job.

---

<a class="slide-id" id="f809ad36-586f-4123-9378-3a06947c5759"></a>

## The simulator ⇄ egodriver cycle

AlpaSim pushes observations and asks `drive()`; the egodriver snapshots them, steps the policy, and returns the chosen trajectory.

<div style="display:flex;justify-content:center;margin-top:0.4rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="980" height="507" style="max-width:100%" font-family="Inter, sans-serif">
  <defs>
    <marker id="c" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#76b900"/></marker>
    <marker id="r" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#888"/></marker>
    <marker id="b" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#58a6ff"/></marker>
  </defs>
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <style>.h{fill:#f0f0f0;font-weight:700}.m{font-family:'JetBrains Mono',monospace}.lbl{fill:#d8d8d8}.ret{fill:#888;font-style:italic}.d{fill:#a0a0a0}</style>
  <!-- lifelines -->
  <g stroke="#333" stroke-width="1.4" stroke-dasharray="4 4">
    <line x1="110" y1="120" x2="110" y2="566"/>
    <line x1="355" y1="120" x2="355" y2="566"/>
    <line x1="600" y1="120" x2="600" y2="566"/>
    <line x1="830" y1="120" x2="830" y2="566"/>
    <line x1="1052" y1="120" x2="1052" y2="566"/>
  </g>
  <!-- headers -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/episode_runner/streaming_worker.py" target="_blank">
    <rect x="14" y="74" width="192" height="44" rx="9" fill="#1a1a1a" stroke="#76b900" stroke-width="1.6"/>
    <text x="110" y="95" text-anchor="middle" class="h" font-size="13">StreamingRolloutWorker</text>
    <text x="110" y="111" text-anchor="middle" class="d m" font-size="10.5">alpagym-sim-i thread</text>
  </a>
  <rect x="267" y="74" width="176" height="44" rx="9" fill="#1a1a1a" stroke="#2a2a2a"/>
  <text x="355" y="100" text-anchor="middle" class="h" font-size="14">AlpaSim</text>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/driver_server.py" target="_blank">
    <rect x="512" y="74" width="176" height="44" rx="9" fill="#1a1a1a" stroke="#76b900" stroke-width="1.6"/>
    <text x="600" y="95" text-anchor="middle" class="h" font-size="14">EgodriverServer</text>
    <text x="600" y="111" text-anchor="middle" class="d m" font-size="10.5">driver_server.py</text>
  </a>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/policy.py" target="_blank">
    <rect x="742" y="74" width="176" height="44" rx="9" fill="#1a1a1a" stroke="#76b900" stroke-width="1.6"/>
    <text x="830" y="95" text-anchor="middle" class="h" font-size="14">AlpamayoPolicy</text>
    <text x="830" y="111" text-anchor="middle" class="d m" font-size="10.5">policy.py</text>
  </a>
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/inference/inference_engine.py" target="_blank">
    <rect x="964" y="74" width="184" height="44" rx="9" fill="#1a1a1a" stroke="#76b900" stroke-width="1.6"/>
    <text x="1056" y="95" text-anchor="middle" class="h" font-size="14">InferenceEngine</text>
    <text x="1056" y="111" text-anchor="middle" class="d m" font-size="10.5">inference_engine.py</text>
  </a>
  <!-- simulate() -->
  <path d="M112,150 L353,150" stroke="#76b900" stroke-width="2" marker-end="url(#c)"/>
  <text x="115" y="143" class="lbl m" font-size="12.5">simulate(1)  ·  one rollout per call, on a sim thread</text>
  <!-- start_session -->
  <path d="M357,182 L598,182" stroke="#76b900" stroke-width="2" marker-end="url(#c)"/>
  <text x="360" y="175" class="lbl m" font-size="12.5">start_session() ⇒ policy_factory() builds a Policy</text>
  <!-- loop box -->
  <rect x="322" y="200" width="800" height="276" rx="10" fill="none" stroke="#58a6ff" stroke-width="1.3" stroke-dasharray="6 4"/>
  <rect x="322" y="200" width="150" height="26" rx="0" fill="#58a6ff"/>
  <text x="332" y="218" fill="#0a0a0a" font-size="13" font-weight="700">loop  per sim tick</text>
  <!-- submit_* -->
  <path d="M353,254 L598,254" stroke="#76b900" stroke-width="2" marker-end="url(#c)"/>
  <text x="356" y="247" class="lbl m" font-size="12">submit_image / submit_egomotion / submit_route  →  TickBuffer</text>
  <!-- drive -->
  <path d="M353,294 L598,294" stroke="#76b900" stroke-width="2.4" marker-end="url(#c)"/>
  <text x="356" y="287" class="lbl m" font-size="12.5" font-weight="700">drive(DriveRequest)</text>
  <!-- step -->
  <path d="M602,328 L828,328" stroke="#76b900" stroke-width="2" marker-end="url(#c)"/>
  <text x="606" y="321" class="lbl m" font-size="12">step(PolicyInput)  ·  from tick snapshot</text>
  <!-- infer -->
  <path d="M832,362 L1050,362" stroke="#58a6ff" stroke-width="2" marker-end="url(#b)"/>
  <text x="836" y="355" class="lbl m" font-size="12">infer(ModelInput) → Future</text>
  <!-- model output return -->
  <path d="M1050,396 L832,396" stroke="#888" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#r)"/>
  <text x="1048" y="389" text-anchor="end" class="ret m" font-size="12">ModelOutput  (batched on GPU, unbound)</text>
  <!-- policy output return -->
  <path d="M828,430 L602,430" stroke="#888" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#r)"/>
  <text x="824" y="423" text-anchor="end" class="ret m" font-size="12">PolicyOutput  (selector picks 1 trajectory)</text>
  <!-- drive response return -->
  <path d="M598,464 L357,464" stroke="#888" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#r)"/>
  <text x="594" y="457" text-anchor="end" class="ret m" font-size="12">DriveResponse  ⇒ AlpaSim steps the car</text>
  <!-- close_session -->
  <path d="M353,506 L598,506" stroke="#76b900" stroke-width="2" marker-end="url(#c)"/>
  <text x="356" y="499" class="lbl m" font-size="12">close_session() ⇒ freeze SessionRecord</text>
  <!-- simulate returns -->
  <path d="M353,540 L114,540" stroke="#888" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#r)"/>
  <text x="350" y="533" text-anchor="end" class="ret m" font-size="12">simulate() returns — this rollout&#39;s SessionRecord</text>
  <text x="110" y="566" class="d" font-size="11.5">then the worker: drain SessionRecord → EpisodeOutput → compute_reward → torch.save .pt → resolve Future</text>
</svg>
</div>

---

<a class="slide-id" id="ddf4c0db-3dbe-499e-abba-6ea04b40114a"></a>

## Per-session state & the tick buffer

> A **tick** is one simulation step — the simulator gathers the latest sensors, asks the policy for a control, applies it, and advances time.

<div class="columns">
<div>

### Staging a tick — [`tick_buffer.py`][tick]
AlpaSim sends cameras, egomotion and route in **separate** RPCs. The `TickBuffer` accumulates them for the current tick until `drive()` fires, then the servicer snapshots it.

### Cameras — [`cameras.py`][cameras]
A fixed name→index map puts the 7 cameras in the order the model expects, regardless of arrival order.

</div>
<div>

### Sessions & records — [`driver_server.py`][driver]
- One `_Session` per scenario: its own policy, tick buffer and lock
- `drive()` → snapshot → `policy.step()` → record I/O → `DriveResponse`
- On `close_session()` the data is frozen into an immutable **`SessionRecord`** the runner drains

</div>
</div>

---

<a class="slide-id" id="ee076ffe-3b71-4ed3-bfca-7b1049eb9db4"></a>

## The protobuf boundary

All gRPC ⇄ Python translation lives in one file, [`proto_conversion.py`][proto] — the single place where wire formats meet AlpaGym types.

<div class="columns">
<div>

### Inbound
- `policy_input_from_tick_buffer()` → [`PolicyInput`][rtypes]
- `calibration_from_proto()` (sent once at session start)
- `ground_truth_from_proto()`

</div>
<div>

### Outbound
- `drive_response_from_policy_output()` → `DriveResponse`
- `build_simulation_request_proto()` → the `simulate()` call

</div>
</div>

> Keeping conversion in one module means the rest of the runtime works with clean dataclasses, never raw protobufs.

---

<a class="slide-id" id="204fa82e-aea1-455d-852b-fd29852a066a"></a>

<!-- _class: invert -->

# Policy and **Inference**

<p>How the car turns pixels into a trajectory</p>

---

<a class="slide-id" id="8dbbc1ad-74b3-42e5-9f55-8e6fe3c12458"></a>

## One policy step

[`AlpamayoPolicy.step()`](https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/policy.py) is four phases:

<div class="columns">
<div>

1. **Preprocess** — fold the new observation into [buffers][buffers]; build a `ModelInput` (camera frames, ego-history in **rig frame**, route)
2. **Infer** — `inference_engine.infer(...)` returns a `Future`; block on it

</div>
<div>

3. **Select** — a [selector][selectors] picks one trajectory from the sampled candidates
4. **Postprocess** — convert the chosen trajectory from **rig frame → world frame** ([`output_trajectory.py`][outtraj]) and emit a `PolicyOutput`

</div>
</div>

> The policy never touches the GPU directly — it hands work to the shared inference engine and waits.

---

<a class="slide-id" id="69be8255-d2c1-4e4d-9aef-d5497cca9212"></a>

## Inference: many sessions, one GPU batch

The [`InferenceEngine`][infeng] decouples *who asks* from *how it runs*: per-session threads enqueue requests and block on futures; a single worker thread batches them through the model.

<div style="display:flex;justify-content:center;margin-top:0.4rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="980" height="507" style="max-width:100%" font-family="Inter, sans-serif">
  <defs>
    <marker id="c" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#76b900"/></marker>
    <marker id="r" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#888"/></marker>
  </defs>
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <style>.t{fill:#f0f0f0;font-weight:700}.m{font-family:'JetBrains Mono',monospace;fill:#a3d44a}.d{fill:#a0a0a0}.ret{fill:#888;font-style:italic}</style>
  <text x="30" y="48" class="d" font-size="15">Many concurrent sessions, <tspan fill="#f0f0f0" font-weight="700">one GPU batch</tspan> — opportunistic batching keeps latency low and the GPU busy.</text>
  <!-- producer threads -->
  <text x="30" y="92" fill="#76b900" font-size="13" font-weight="700">PRODUCERS — one policy thread per session</text>
  <g>
    <rect x="24" y="104" width="244" height="78" rx="11" fill="#1a1a1a" stroke="#2a2a2a"/>
    <text x="40" y="132" class="t" font-size="14">Policy thread · session 0</text>
    <text x="40" y="156" class="m" font-size="12">infer(ModelInput) → Future</text>
    <text x="40" y="174" class="ret" font-size="11.5">blocks on future.result()</text>
    <rect x="24" y="196" width="244" height="78" rx="11" fill="#1a1a1a" stroke="#2a2a2a"/>
    <text x="40" y="224" class="t" font-size="14">Policy thread · session 1</text>
    <text x="40" y="248" class="m" font-size="12">infer(ModelInput) → Future</text>
    <text x="40" y="266" class="ret" font-size="11.5">blocks on future.result()</text>
    <text x="146" y="300" text-anchor="middle" class="d" font-size="20">⋮</text>
    <rect x="24" y="312" width="244" height="78" rx="11" fill="#1a1a1a" stroke="#2a2a2a"/>
    <text x="40" y="340" class="t" font-size="14">Policy thread · session k</text>
    <text x="40" y="364" class="m" font-size="12">infer(ModelInput) → Future</text>
    <text x="40" y="382" class="ret" font-size="11.5">blocks on future.result()</text>
  </g>
  <!-- queue -->
  <rect x="312" y="120" width="92" height="270" rx="10" fill="#0d1117" stroke="#58a6ff" stroke-width="1.4"/>
  <text x="358" y="112" text-anchor="middle" fill="#58a6ff" font-size="12.5" font-weight="700">queue</text>
  <g fill="#1a1a1a" stroke="#76b900" stroke-width="1.2">
    <rect x="328" y="138" width="60" height="34" rx="6"/>
    <rect x="328" y="180" width="60" height="34" rx="6"/>
    <rect x="328" y="222" width="60" height="34" rx="6"/>
    <rect x="328" y="264" width="60" height="34" rx="6"/>
  </g>
  <text x="358" y="160" text-anchor="middle" class="m" font-size="11" fill="#a3d44a">req</text>
  <text x="358" y="202" text-anchor="middle" class="m" font-size="11" fill="#a3d44a">req</text>
  <text x="358" y="244" text-anchor="middle" class="m" font-size="11" fill="#a3d44a">req</text>
  <text x="358" y="286" text-anchor="middle" class="m" font-size="11" fill="#a3d44a">req</text>
  <text x="358" y="330" text-anchor="middle" class="d" font-size="20">⋮</text>
  <!-- enqueue arrows -->
  <path d="M270,143 L310,150" stroke="#76b900" stroke-width="2" marker-end="url(#c)"/>
  <path d="M270,235 L310,230" stroke="#76b900" stroke-width="2" marker-end="url(#c)"/>
  <path d="M270,351 L310,300" stroke="#76b900" stroke-width="2" marker-end="url(#c)"/>
  <!-- worker -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/inference/inference_engine.py" target="_blank">
    <rect x="446" y="120" width="318" height="270" rx="12" fill="#161616" stroke="#76b900" stroke-width="2"/>
  </a>
  <text x="466" y="148" class="t" font-size="16">run_loop()  ·  single worker thread</text>
  <text x="466" y="168" class="m" font-size="11.5">inference_engine.py</text>
  <text x="466" y="202" class="d" font-size="13.5">1 · block for the first request</text>
  <text x="466" y="230" class="d" font-size="13.5">2 · drain more, up to <tspan fill="#a3d44a" class="m" font-size="12.5">max_batch_size</tspan></text>
  <text x="466" y="258" class="d" font-size="13.5">3 · <tspan fill="#a3d44a" class="m" font-size="12.5">BatchedModelInput.stack()</tspan></text>
  <text x="466" y="286" class="d" font-size="13.5">4 · one model forward pass →</text>
  <text x="466" y="314" class="d" font-size="13.5">5 · <tspan fill="#a3d44a" class="m" font-size="12.5">.unbind()</tspan> back to per-call outputs</text>
  <text x="466" y="342" class="d" font-size="13.5">6 · <tspan fill="#a3d44a" class="m" font-size="12.5">future.set_result(...)</tspan> for each</text>
  <text x="466" y="372" class="ret" font-size="12">one failure fails the whole in-flight batch</text>
  <!-- dequeue arrow -->
  <path d="M406,250 L444,250" stroke="#76b900" stroke-width="2.4" marker-end="url(#c)"/>
  <!-- model -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/inference_model.py" target="_blank">
    <rect x="812" y="170" width="324" height="170" rx="12" fill="#1a1a1a" stroke="#bc8cff" stroke-width="1.8"/>
  </a>
  <text x="974" y="206" text-anchor="middle" class="t" font-size="16">inference model (GPU)</text>
  <text x="974" y="232" text-anchor="middle" class="m" font-size="12" fill="#bc8cff">sample_trajectories_from_data()</text>
  <text x="974" y="262" text-anchor="middle" class="d" font-size="13">alpamayo_r1 · plugin models</text>
  <text x="974" y="292" text-anchor="middle" class="d" font-size="13">one forward for the whole batch →</text>
  <text x="974" y="318" text-anchor="middle" class="d" font-size="13">returns BatchedModelOutput</text>
  <!-- worker <-> model -->
  <path d="M766,230 L810,230" stroke="#76b900" stroke-width="2" marker-end="url(#c)"/>
  <text x="788" y="222" text-anchor="middle" class="ret" font-size="10.5">stack</text>
  <path d="M810,290 L766,290" stroke="#888" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#r)"/>
  <text x="788" y="308" text-anchor="middle" class="ret" font-size="10.5">unbind</text>
  <!-- return to threads -->
  <path d="M520,392 L520,440 L146,440 L146,392" stroke="#888" stroke-width="1.8" stroke-dasharray="6 4" fill="none" marker-end="url(#r)"/>
  <text x="333" y="460" text-anchor="middle" class="ret" font-size="12.5">future.result() unblocks each waiting policy thread with its own ModelOutput</text>
</svg>
</div>

---

<a class="slide-id" id="a681b382-66cd-4070-a879-d2d879968823"></a>

## Models plug in as policies

A policy package ships an `InferenceModel` and registers a bundle under a `policy.model.kind` string, so the rollout machinery stays **model-agnostic** ([`factory.py`][factory] → [`registry.py`][registry]).

| Kind | Where | What it is |
|------|------|------------|
| `alpamayo_r1` | [`policies/alpamayo_r1`][r1] | VLM that **reasons** (chain-of-thought) before predicting a trajectory — the reference policy |
| *your policy* | external package | any policy package plugs in the same way (see [The Plugin Boundary](#the-plugin-boundary)) |
| *test double* | tests | a deterministic stand-in exercises the plumbing with no GPU or weights |

> A bundle's model code is loaded from an HF directory or tarball via [`model_bundle.py`][modelbundle]; how a kind resolves to a package is [The Plugin Boundary](#the-plugin-boundary).

---

<a class="slide-id" id="428786bc-f3ab-450c-b680-0380e5da12f3"></a>

## Buffers, selectors & coordinate frames

<div class="columns">
<div>

### Observation buffers — [`buffers.py`][buffers]
Per-session, fixed-capacity **GPU-resident** ring buffers for decoded camera frames and a monotonic ego-pose history. Frames are GPU-decoded into on-device storage (`frame_device`) and served as chronologically-ordered views.

### Selectors — [`selectors.py`][selectors]
- `IdentitySelector` — always the first candidate
- `ClosestToPreviousSelector` — the candidate closest to last step's plan, for **smooth, continuous** driving

</div>
<div>

### Rig ⇄ world frames
- Models think in the **rig frame** (ego-centric, origin at the latest pose)
- The simulator wants **world frame** (absolute)
- [`geometry_utils.py`][geom] handles quaternions, rotation matrices and interpolation; [`output_trajectory.py`][outtraj] does the final rig→world transform and prepends the current pose

</div>
</div>

---

<a class="slide-id" id="7b6d2fec-202a-4187-ab7f-cc3114e229d8"></a>

<!-- _class: invert -->

# **Rewards**

---

<a class="slide-id" id="0d4be1f8-518f-4cfa-9928-fcf712a19c9f"></a>

## Scoring an episode

[`compute_reward()`][compute] is a simple, transparent **weighted sum** of independent terms — each term reads from the episode and contributes a scaled value plus a report.

<div class="columns">
<div>

### Term kinds
- **`metric`** — pull a value straight from `episode.metrics.aggregated`
- **`distance_to_gt`** — compare the executed trajectory to the recorded ground truth

Configured in [`conf/reward/*.yaml`](https://github.com/NVlabs/alpagym/tree/main/packages/host/src/alpagym_host/conf/reward); validated by `RewardConfig`.

</div>
<div>

### distance-to-GT — [`distance_to_gt.py`][dist]
1. Align executed & GT trajectories at a common anchor time
2. Interpolate GT onto executed timestamps
3. **XY RMSE** over the overlap window → scaled reward

Useful for imitation / trajectory-tracking objectives.

</div>
</div>

> Reward is computed during the rollout and stored on the episode; Cosmos later reads `episode.reward.total` back via [`reward_fn.py`][rewardfn].

---

<a class="slide-id" id="b1935183-9365-4de9-99be-45062fe44a0f"></a>

<!-- _class: invert -->

# Episode **Transport**

<p>Getting a finished episode from the rollout worker to the trainer</p>

---

<a class="slide-id" id="eb62b7d4-d1af-4a76-953a-6b53d32627da"></a>

## The transport seam

The rollout → trainer hand-off goes through one egress protocol — `EpisodeWriter` ([`base.py`][txbase]) — with **two implementations** chosen by `transport=`: **disk** (the default, a JSON artifact) or GPU-direct **NCCL** (sender on the rollout worker, receiver on the policy worker).

<div style="display:flex;justify-content:center;margin-top:0.4rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="980" height="507" style="max-width:100%" font-family="Inter, sans-serif">
  <defs>
    <marker id="g" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#888"/></marker>
    <marker id="gr" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#76b900"/></marker>
  </defs>
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <style>.t{fill:#f0f0f0;font-weight:700}.m{font-family:'JetBrains Mono',monospace;fill:#a3d44a}.d{fill:#a0a0a0}.lnk{fill:#58a6ff}</style>
  <!-- Rollout (sender) -->
  <rect x="34" y="250" width="210" height="116" rx="13" fill="#1a1a1a" stroke="#76b900" stroke-width="1.8"/>
  <text x="139" y="296" text-anchor="middle" class="t" font-size="16">Rollout replica</text>
  <text x="139" y="320" text-anchor="middle" class="d" font-size="13">sender</text>
  <text x="139" y="344" text-anchor="middle" class="d" font-size="12.5">episode tensors</text>
  <!-- seam -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/base.py" target="_blank">
    <rect x="300" y="262" width="150" height="92" rx="11" fill="#0d1117" stroke="#58a6ff" stroke-width="1.4"/>
    <text x="375" y="296" text-anchor="middle" class="lnk" font-size="13.5" font-weight="700">EpisodeWriter</text>
    <text x="375" y="318" text-anchor="middle" class="d" font-size="11.5">transport/base.py</text>
    <text x="375" y="338" text-anchor="middle" class="m" font-size="11">transport={disk|nccl}</text>
  </a>
  <path d="M244,308 L298,308" stroke="#888" stroke-width="2" marker-end="url(#g)"/>
  <!-- DISK lane (top, default) -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/disk.py" target="_blank">
    <rect x="540" y="96" width="290" height="92" rx="12" fill="#1a1a1a" stroke="#2a2a2a"/>
    <text x="685" y="130" text-anchor="middle" class="t" font-size="15">JSON artifact on disk</text>
    <text x="685" y="154" text-anchor="middle" class="m" font-size="12">transport/disk.py</text>
    <text x="685" y="174" text-anchor="middle" class="d" font-size="11.5">write → Lustre → read</text>
  </a>
  <path d="M392,260 C392,142 430,142 538,142" fill="none" stroke="#888" stroke-width="2" marker-end="url(#g)"/>
  <path d="M830,142 C900,142 900,250 916,300" fill="none" stroke="#888" stroke-width="2" marker-end="url(#g)"/>
  <rect x="690" y="60" width="86" height="26" rx="13" fill="rgba(118,185,0,0.15)"/>
  <text x="733" y="78" text-anchor="middle" fill="#76b900" font-size="12" font-weight="700">DEFAULT</text>
  <!-- NCCL lane (bottom, in progress) -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/nccl/README.md" target="_blank">
    <rect x="540" y="420" width="290" height="92" rx="12" fill="#161616" stroke="#76b900" stroke-width="1.8"/>
    <text x="685" y="454" text-anchor="middle" class="t" font-size="15">NCCL — GPU → GPU tensors</text>
    <text x="685" y="478" text-anchor="middle" class="m" font-size="12">transport/nccl/</text>
    <text x="685" y="498" text-anchor="middle" class="d" font-size="11.5">no disk round-trip</text>
  </a>
  <path d="M392,356 C392,466 430,466 538,466" fill="none" stroke="#76b900" stroke-width="2.4" marker-end="url(#gr)"/>
  <path d="M830,466 C900,466 900,360 916,318" fill="none" stroke="#76b900" stroke-width="2.4" marker-end="url(#gr)"/>
  <rect x="652" y="384" width="166" height="26" rx="13" fill="rgba(240,136,62,0.15)"/>
  <text x="735" y="402" text-anchor="middle" fill="#f0883e" font-size="12" font-weight="700">OPT-IN · IN PROGRESS</text>
  <!-- Trainer (receiver) -->
  <rect x="916" y="250" width="210" height="116" rx="13" fill="#1a1a1a" stroke="#76b900" stroke-width="1.8"/>
  <text x="1021" y="296" text-anchor="middle" class="t" font-size="16">Policy / trainer</text>
  <text x="1021" y="320" text-anchor="middle" class="d" font-size="13">receiver</text>
  <text x="1021" y="344" text-anchor="middle" class="d" font-size="12.5">training step</text>
  <!-- title hint -->
  <text x="34" y="60" class="d" font-size="13">The rollout → trainer episode hand-off goes through one swappable <tspan fill="#f0f0f0" font-weight="700">transport</tspan> seam:</text>
</svg>
</div>

A **handle** is the string the writer returns and the reader resolves — for disk, just the JSON file path; for NCCL, the rendezvous manifest key.

---

<a class="slide-id" id="75fce661-f3df-4850-b05c-ffd086f10fab"></a>

## Disk — the default transport <span class="tag tag-green">default</span>

`DiskEpisodeWriter.write` ([`disk.py`][disk]) serializes the whole `EpisodeOutput` (reward + replay payload) to one **`.json`** artifact; the trainer reads it back with `read_episode_json`. This is the default hand-off (`transport=disk`) unless you opt into NCCL.

<div style="display:flex;justify-content:center;margin-top:0.4rem">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 600" width="980" height="507" style="max-width:100%" font-family="Inter, sans-serif">
  <defs>
    <marker id="g" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#888"/></marker>
    <marker id="gr" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#76b900"/></marker>
  </defs>
  <rect x="0" y="0" width="1160" height="600" fill="#111111"/>
  <style>.t{fill:#f0f0f0;font-weight:700}.m{font-family:'JetBrains Mono',monospace;fill:#a3d44a}.d{fill:#a0a0a0}.lnk{fill:#58a6ff}</style>
  <text x="30" y="52" class="d" font-size="15"><tspan fill="#76b900" font-weight="700">On main · synchronous · filesystem-bound</tspan> — one <tspan fill="#f0f0f0" font-weight="700">torch.save .pt</tspan> blob per rollout, read back by the trainer.</text>
  <!-- Rollout writer -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/disk.py" target="_blank">
    <rect x="34" y="210" width="250" height="150" rx="13" fill="#1a1a1a" stroke="#76b900" stroke-width="2"/>
    <text x="159" y="250" text-anchor="middle" class="t" font-size="17">Rollout worker</text>
    <text x="159" y="276" text-anchor="middle" class="m" font-size="12">write_rollout_artifact</text>
    <text x="159" y="306" text-anchor="middle" class="d" font-size="13">torch.save(episode) → path</text>
    <text x="159" y="330" text-anchor="middle" class="d" font-size="12.5">whole EpisodeOutput + reward</text>
  </a>
  <!-- disk cylinder -->
  <g>
    <ellipse cx="580" cy="206" rx="120" ry="26" fill="#0d1117" stroke="#2a2a2a"/>
    <path d="M460,206 L460,360 A120,26 0 0 0 700,360 L700,206" fill="#0d1117" stroke="#2a2a2a"/>
    <ellipse cx="580" cy="206" rx="120" ry="26" fill="#161616" stroke="#2a2a2a"/>
    <text x="580" y="262" text-anchor="middle" class="t" font-size="16">episode artifact</text>
    <text x="580" y="288" text-anchor="middle" class="m" font-size="11.5" fill="#a3d44a">{scene}-{uuid}.pt</text>
    <text x="580" y="316" text-anchor="middle" class="d" font-size="12">on shared filesystem</text>
    <text x="580" y="338" text-anchor="middle" class="d" font-size="12">.tmp + atomic rename</text>
  </g>
  <!-- Trainer -->
  <a href="https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/packer.py" target="_blank">
    <rect x="876" y="210" width="250" height="150" rx="13" fill="#1a1a1a" stroke="#58a6ff" stroke-width="2"/>
    <text x="1001" y="250" text-anchor="middle" class="t" font-size="17">Trainer worker</text>
    <text x="1001" y="276" text-anchor="middle" class="m" font-size="12">read_rollout_artifact</text>
    <text x="1001" y="306" text-anchor="middle" class="d" font-size="13">torch.load(path)</text>
    <text x="1001" y="330" text-anchor="middle" class="d" font-size="12.5">packer + reward_fn</text>
  </a>
  <!-- arrows -->
  <path d="M286,272 C360,272 380,250 458,250" fill="none" stroke="#76b900" stroke-width="2.4" marker-end="url(#gr)"/>
  <text x="372" y="240" text-anchor="middle" class="d" font-size="12">torch.save</text>
  <path d="M702,250 C780,250 800,272 874,272" fill="none" stroke="#888" stroke-width="2.4" marker-end="url(#g)"/>
  <text x="788" y="240" text-anchor="middle" class="d" font-size="12">torch.load ×2</text>
  <!-- handle -->
  <path d="M876,320 C800,330 740,330 702,330" fill="none" stroke="#888" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#g)"/>
  <text x="788" y="352" text-anchor="middle" class="d" font-size="11">path string = the handle</text>
  <!-- cost callout -->
  <rect x="34" y="430" width="1092" height="120" rx="12" fill="#0d1117" stroke="#2a2a2a"/>
  <text x="56" y="466" class="t" font-size="15">The cost</text>
  <text x="56" y="496" class="d" font-size="13.5">• One <tspan fill="#f0f0f0">torch.save</tspan> of the entire <tspan class="m" font-size="12.5">EpisodeOutput</tspan> per rollout (reward + replay payload: camera frames, trace, tokens).</text>
  <text x="56" y="522" class="d" font-size="13.5">• The trainer <tspan fill="#f0f0f0">torch.loads it twice</tspan> — once in the data packer (replay + log-probs), once in the reward fn (<tspan class="m" font-size="12">reward.total</tspan>).</text>
  <text x="56" y="544" class="d" font-size="12.5">The <tspan fill="#f0f0f0">path string is the handle</tspan> Cosmos carries back; reward is computed once (rollout), only re-read on the trainer. NCCL would remove this disk hop →</text>
</svg>
</div>

---

<a class="slide-id" id="48fb5ff0-7bf8-44ab-acf9-9d3ae81ff103"></a>

## NCCL — the opt-in fast path

Set `transport=nccl` to ship episode **tensors GPU → GPU** instead of through disk, with only a tiny **manifest** on the side. Three separate channels keep the heavy data path off the slow one:

<div class="columns">
<div class="card">

### Control — `TCPStore`
The reconstruction manifest + rendezvous keys ([`comm_init.py`][txcomm], [`rendezvous.py`][txrdv]). A manifest-driven rendezvous gates every blocking transfer, so there is **one commit point and no deadlock**.

</div>
<div class="card">

### Data — `NCCL`
Tensor payloads **only**, one send per tensor on a dedicated CUDA stream ([`sender.py`][txsender] → [`receiver.py`][txreceiver]).

### Discard — `Redis`
Cosmos publishes discards; the sender frees the GPU buffer.

</div>
</div>

> Opt-in, and `cosmos.mode=disaggregated`-only (colocated always uses disk); single-host today. Full design, env handling and constraints: [NCCL&nbsp;README][nccldesign] · [`transport_env.py`][transportenv].

---

<a class="slide-id" id="c4ac9bff-1ccf-4372-8cd0-b3c8d35fe93d"></a>

<!-- _class: invert -->

# The Plugin **Boundary**

<p>How policies attach to the core — and how to add your own</p>

---

<a class="slide-id" id="9e7ce598-0a1f-4fae-b46f-62ec1cfbd941"></a>

## Policies are plugins

The core never imports a specific driving model. A policy is discovered as an **installed plugin** and selected by a `policy.model.kind` string — so you add or swap a policy without touching core code.

<div class="columns">
<div>

### The mechanism
- `policy.model.kind` is an **opaque string** — [`config.py`][config]
- Policies are discovered as **installed plugins** — [`plugins.py`][plugins]
- Each model lives in its **own package**

</div>
<div>

### Why it's shaped this way
- The core builds and runs with **no policy package** installed
- Adding a policy = ship a package + declare an entry point; no edits to `packages/*`
- Cosmos-RL registries stay at the Cosmos boundary — not the policy-selection mechanism

</div>
</div>

> One string in, one installed plugin out — the rest of the harness stays policy-agnostic.

---

<a class="slide-id" id="1a644019-99d6-46e2-bde4-31aadd190a6e"></a>

## How it works — entry-point discovery

Two **entry-point groups** keep policy names out of the core. The core discovers whatever plugins are *installed*; each policy package advertises itself from its own `pyproject.toml`.

<div class="columns">
<div>

### The two groups
- `alpagym.policy_bundles` → a [`PolicyBundle`][registry] of runtime hooks
- `alpagym.configs` → Hydra config search paths

A `PluginRegistry` ([`plugins.py`][plugins]) reads the installed table; `get_policy_bundle(kind)` loads only the one you selected.

</div>
<div>

### `alpamayo_r1` as the worked example
```toml
[project.entry-points."alpagym.policy_bundles"]
alpamayo_r1 = "alpagym_alpamayo_r1.bundle:get_bundle"
[project.entry-points."alpagym.configs"]
alpamayo_r1 = "alpagym_alpamayo_r1.configs"
```

Install the package → `policy.model.kind=alpamayo_r1` resolves. Nothing in `packages/*` imports it directly.

</div>
</div>

---

<a class="slide-id" id="b7113de2-8c82-48ae-b52b-7da813c7b558"></a>

## The PolicyBundle contract

The selected plugin returns a frozen [`PolicyBundle`][registry] of five hooks; the factory and trainer call **only the bundle** — never `import` a model package directly.

<div class="columns">
<div>

### The five hooks
- `install_runtime_bridge()` — register the model's code with the runtime
- `load_inference_model(…)` — build the GPU model
- `build_model_inputs(…)` — replay payload → model kwargs
- `build_data_packer(…)` · `setup_tokenizer(…)`

</div>
<div>

### The payoff
- **Fail fast** if the selected bundle isn't installed — [`plugins.py`][plugins]
- `check-policy-bundles` validates the wiring — no GPU, no checkpoint, no AlpaSim — [`check_policy_bundles.py`][checkbundles]
- The core builds and runs with **no policy package** installed
- Cosmos-RL registries stay at the Cosmos boundary — not the policy-selection mechanism

</div>
</div>

> Each model ships as a package-owned plugin (e.g. [`alpamayo_r1`][r1bundle]) — no per-model branch in the core.

---

<a class="slide-id" id="098750a7-78fe-4dfb-b5ef-de379052b1de"></a>

<!-- _class: invert -->

# Engineering **Culture**

---

<a class="slide-id" id="f9d33ea9-1c40-4afd-9f9c-9af0f1bb40cd"></a>

## How AlpaGym is written

The [`CONTRIBUTING.md`][contributing] guide captures the public contribution contract. Target reader: *a researcher who will modify the code, not just run it.*

<div class="columns">
<div>

### Code principles
- **Readability over flexibility** — abstractions must reduce net complexity (YAGNI)
- **Locality of behavior** — few hops; inline over gratuitous wrappers
- **One way to do it** — opinionated, no legacy paths
- **Fail fast** — raise on bad input, no silent fallbacks
- **Config in one place** — Hydra, no shadow constants

</div>
<div>

### Tests & review
- Tests prove **behavior**, not config defaults or implementation details
- Authors read every changed line before requesting review
- Contributions are signed off with the Developer Certificate of Origin
- The guide is a **living document** — update it when a rule is missing or unclear

</div>
</div>

---

<a class="slide-id" id="a2a55c82-e5a4-480b-a8cf-d509bfc3a1a1"></a>

<!-- _class: invert -->

# Where To **Look**

---

<a class="slide-id" id="465fed22-98f1-4822-95a3-5875f776fe04"></a>

## The reference map

<div class="columns">
<div>

### Control plane (`host`)
- Entry & lifecycle — [`cli.py`][cli]
- Config schema — [`config.py`][config]
- Simulator bring-up — [`alpasim_wizard.py`][wizard]
- Endpoint registry — [`endpoint_registry.py`][topo]
- Slurm & containers — [`slurm.py`][slurm]
- Run artifacts — [`run_artifacts.py`][writeart]

### RL runtime (`cosmos`)
- Registry & entry — [`entrypoint.py`][entrypoint]
- Actor — [`rollout_backend.py`][rollout]
- Trainer — [`trainer.py`][trainer]
- Replay contract — [`replay.py`][replay]
- Framework — [Cosmos-RL][cosmosdocs] · [controller][cosmosctl]
- Transport — [`base.py`][txbase] · [NCCL README][nccldesign]

</div>
<div>

### Episode & sim
- Rollout worker — [`streaming_worker.py`][runbatch]
- Egodriver server — [`driver_server.py`][driver]
- Proto boundary — [`proto_conversion.py`][proto]

### Policy & inference
- Policy step — [`policy.py`][policy]
- Inference engine — [`inference_engine.py`][infeng]
- Model bundle loader — [`model_bundle.py`][modelbundle]

### Plugins & rewards
- Plugin registry — [`plugins.py`][plugins]
- Policy bundle — [`registry.py`][registry] · [`alpamayo_r1`][r1bundle]
- Reward sum — [`compute.py`][compute]

</div>
</div>

---

<a class="slide-id" id="95cf011a-fdd0-403f-803f-fb811d639021"></a>

<!-- _class: lead -->

# Thank you

Start with the [smoke run](#run-it), then follow the [reference map](#where-to-look) into the code.

[Repo (`NVlabs/alpagym`)][readme] · [README][readme] · [Contributing][contributing] · links follow `main`

<!-- LINK REFERENCES (github.com/NVlabs/alpagym public main) -->
[readme]: https://github.com/NVlabs/alpagym/blob/main/README.md
[contributing]: https://github.com/NVlabs/alpagym/blob/main/CONTRIBUTING.md
[cli]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/cli.py
[execute_run]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/run_lifecycle.py
[config]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/config.py
[writeart]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/run_artifacts.py
[wizard]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/alpasim_wizard.py
[topo]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/endpoint_registry.py
[dep]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/alpasim_dependency.py
[slurm]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/slurm.py
[submitjob]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/slurm.py
[prepimg]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/slurm.py
[defaults]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/conf/default.yaml
[entrypoint]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/entrypoint.py
[trainer]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/trainer.py
[rollout]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/rollout_backend.py
[dataset]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/dataset.py
[packer]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/packer.py
[rewardfn]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/reward_fn.py
[runbatch]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/episode_runner/streaming_worker.py
[driver]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/driver_server.py
[proto]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/proto_conversion.py
[tick]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/tick_buffer.py
[cameras]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/cameras.py
[infeng]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/inference/inference_engine.py
[inftypes]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/inference/types.py
[factory]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/factory.py
[policy]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/policy.py
[buffers]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/buffers.py
[selectors]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/selectors.py
[outtraj]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/output_trajectory.py
[geom]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/geometry_utils.py
[modelbundle]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/model_bundle.py
[registry]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/registry.py
[checkbundles]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/check_policy_bundles.py
[plugins]: https://github.com/NVlabs/alpagym/blob/main/packages/plugins/src/alpagym_plugins/plugins.py
[r1]: https://github.com/NVlabs/alpagym/blob/main/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/inference_model.py
[r1bundle]: https://github.com/NVlabs/alpagym/blob/main/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/bundle.py
[compute]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/rewards/compute.py
[dist]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/rewards/distance_to_gt.py
[rtypes]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/types.py
[replay]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/replay.py
[disk]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/disk.py
[txbase]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/base.py
[txsender]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/nccl/sender.py
[txreceiver]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/nccl/receiver.py
[txrdv]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/nccl/rendezvous.py
[txcomm]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/nccl/comm_init.py
[transportenv]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/transport_env.py
[nccldesign]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/nccl/README.md
[runtimereadme]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/README.md
[cosmosdocs]: https://nvidia-cosmos.github.io/cosmos-rl/
[cosmosctl]: https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/dispatcher/controller.py
[cosmoslaunch]: https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/launcher/launch_all.py


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
