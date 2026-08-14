# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import re
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar, cast

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


@dataclass
class ArtifactPaths:
    """Paths generated for one host-owned run directory."""

    run_dir: Path
    artifacts_dir: Path
    policy_model_bundle_dir: Path
    resolved_config_path: Path
    cosmos_config_path: Path
    submit_script_path: Path
    log_dir: Path
    topology_registry_dir: Path
    alpasim_log_dir: Path
    alpasim_scene_ids_path: Path
    perf_dir: Path


@dataclass
class PerfConfig:
    """Flat performance-instrumentation knobs surfaced through `RunConfig.perf`.

    `enabled` gates the whole package. When False every public API short-circuits
    without calling `perf_counter_ns`. Defaults live in `conf/default.yaml`, the
    single source of truth, so an incomplete resolved config fails loudly here
    instead of being silently backfilled from code.
    """

    enabled: bool = MISSING
    sample_every_n: int = MISSING
    resource_sample_interval_s: float | None = MISSING
    max_samples_per_series: int = MISSING
    flush_every_n_updates: int = MISSING
    flush_interval_s: float = MISSING
    collect_cpu: bool = MISSING
    collect_gpu: bool = MISSING

    def __post_init__(self) -> None:
        """Reject non-positive counts and durations before any run work starts."""
        if self.sample_every_n <= 0:
            raise ValueError(f"perf.sample_every_n must be > 0, got {self.sample_every_n}")
        if self.max_samples_per_series <= 0:
            raise ValueError(
                f"perf.max_samples_per_series must be > 0, got {self.max_samples_per_series}"
            )
        if self.flush_every_n_updates <= 0:
            raise ValueError(
                f"perf.flush_every_n_updates must be > 0, got {self.flush_every_n_updates}"
            )
        if not math.isfinite(self.flush_interval_s) or self.flush_interval_s <= 0.0:
            raise ValueError(
                f"perf.flush_interval_s must be a finite value > 0, got {self.flush_interval_s}"
            )
        if self.resource_sample_interval_s is not None and (
            not math.isfinite(self.resource_sample_interval_s)
            or self.resource_sample_interval_s <= 0.0
        ):
            raise ValueError(
                "perf.resource_sample_interval_s must be a finite value > 0 when set, "
                f"got {self.resource_sample_interval_s}"
            )


@dataclass
class DatasetConfig:
    """Dataset selection for a local alpagym run."""

    scene_ids: list[str] | None = None
    test_suite_id: str | None = None


@dataclass
class DiffusionSamplingConfig:
    """Diffusion sampler overrides forwarded to the model.

    Each inference adapter consumes the subset of fields it understands;
    unset fields are omitted from the kwargs dict passed to the model.
    """

    noise_level: float | None = None
    temperature: float | None = None
    int_method: str | None = None
    inference_step: int | None = None


@dataclass
class SamplingParamsConfig:
    """Sampling knobs forwarded to the Alpamayo inference engine."""

    top_p: float
    top_k: int | None
    temperature: float
    num_traj_samples: int
    num_traj_sets: int
    max_generation_length: int | None = None
    diffusion_kwargs: DiffusionSamplingConfig = field(default_factory=DiffusionSamplingConfig)
    # Seed stochastic sampling, run per-row forwards, and enable deterministic runtime settings.
    force_determinism: bool = False
    # Re-run the forward when it returns a non-finite trajectory. The AR1.5 VLM
    # prefill intermittently emits an all-NaN trajectory (scene-state dependent,
    # not a fixed input), and re-running the prefill lands finite. The R1 preset
    # turns this on; determinism lowers the NaN frequency but does not eliminate it.
    retry_on_nonfinite: bool = False


class TrajectorySelectorKind(StrEnum):
    """Trajectory selector strategies supported by the Alpamayo policy."""

    identity = "identity"
    closest_to_previous = "closest_to_previous"


class CosmosRLMode(StrEnum):
    """Cosmos-RL launcher placement modes used by AlpaGym."""

    colocated = "colocated"
    disaggregated = "disaggregated"


class LoggingLevel(StrEnum):
    """Python logging levels supported by AlpaGym processes."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ModelConfig:
    """NN identity, device placement, and model I/O contract."""

    kind: str
    path: str
    device: str
    dtype: str
    use_cameras: list[str]
    num_context_frames: int
    num_historical_waypoints: int
    num_future_waypoints: int
    step_dt_us: int
    # Target `[H, W]` the policy resizes JPEG frames to before pushing
    # them into the per-camera ring.
    input_size: list[int]
    # Policy-specific knobs the selected bundle interprets. Opaque to the host
    # schema so adding a policy needs no schema change.
    bundle_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceConfig:
    """Trajectory sampling, batching, and replay-trace settings."""

    max_batch_size: int
    # Emits logprobs and model extras needed for replay training. Current Cosmos
    # trainer runs require true; rollout-only inference can set false once it has
    # its own entrypoint.
    return_trace_for_rl: bool
    sampling: SamplingParamsConfig


@dataclass
class AlpamayoPolicyConfig:
    """Authored policy settings consumed by the rollout backend."""

    kind: str  # always "alpamayo"
    model: ModelConfig
    inference: InferenceConfig
    trajectory_selector: TrajectorySelectorKind


@dataclass
class CosmosRLLaunchConfig:
    """Cosmos-RL launcher process settings."""

    policy_replicas: int
    rollout_replicas: int
    controller_port: int


@dataclass
class CosmosRLTrainPolicyConfig:
    """Cosmos-RL trainer policy scheduling and GRPO hyperparameters.

    AlpaGym-facing field names describe trainer behavior directly. Generated
    Cosmos-RL configs translate these fields to the names expected by
    Cosmos-RL's `GrpoConfig`.
    """

    allowed_outdated_steps: int
    on_policy: bool
    mini_batch: int
    grpo_ratio_clip_low: float
    grpo_ratio_clip_high: float
    grpo_optimization_iterations: int
    kl_beta: float
    reference_reset_interval: int
    trainer_type: str = "alpagym_grpo"
    ppo_value_loss_coef: float = 0.5
    ppo_value_clip_range: float | None = None
    ppo_normalize_advantages: bool = True
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95


@dataclass
class CosmosRLTrainCkptConfig:
    """Cosmos-RL checkpoint settings.

    Mirrors the fields of upstream ``cosmos_rl.policy.config.CheckpointConfig``
    that AlpaGym uses; the rest fall back to upstream defaults. Saves land
    under ``{run_dir}/cosmos/<timestamp>/checkpoints/step_{N}/`` (cosmos
    resume bundle) and ``{run_dir}/cosmos/<timestamp>/safetensors/step_{N}/``
    (HF-compatible weights). ``<timestamp>`` is appended by cosmos to
    ``train.output_dir`` at startup.
    """

    enable_checkpoint: bool = False
    save_freq: int = 20
    save_freq_in_epoch: int = 0
    export_safetensors: bool = True
    max_keep: int = 5


@dataclass
class CosmosRLTrainConfig:
    """Cosmos-RL train table settings."""

    train_batch_per_replica: int
    max_num_steps: int | None
    num_epochs: int
    optm_lr: float
    optm_warmup_steps: int
    train_policy: CosmosRLTrainPolicyConfig
    # LR schedule after warmup. decay_type in {sqrt, cosine, linear, none};
    # decay_ratio is the fraction of total steps spent decaying (1.0 = whole run);
    # lr decays to optm_min_lr_factor * optm_lr (0.0 = down to zero). The
    # "none" / 0.0 defaults keep a constant LR; they are plain str/float (not
    # Python None) so the generated Cosmos-RL TOML stays serializable. Override
    # to e.g. cosine + 1.0 to enable decay.
    optm_decay_type: str = "none"
    optm_decay_ratio: float = 0.0
    optm_min_lr_factor: float = 0.0
    # Defaults match `conf/default.yaml`; field-level defaults let callers and
    # tests construct `CosmosRLTrainConfig` without supplying `ckpt`.
    ckpt: CosmosRLTrainCkptConfig = field(default_factory=CosmosRLTrainCkptConfig)


@dataclass
class CosmosRLPolicyParallelismConfig:
    """Cosmos-RL policy worker parallelism settings."""

    tp_size: int
    cp_size: int
    ep_size: int
    dp_shard_size: int
    pp_size: int
    pp_micro_batch_size: int
    dp_replicate_size: int


@dataclass
class CosmosRLPolicyConfig:
    """Cosmos-RL policy table settings."""

    parallelism: CosmosRLPolicyParallelismConfig


@dataclass
class CosmosRLRolloutParallelismConfig:
    """Cosmos-RL rollout worker parallelism settings."""

    tp_size: int
    pp_size: int


@dataclass
class CosmosRLRolloutConfig:
    """Cosmos-RL rollout table settings."""

    n_generation: int
    batch_size: int
    parallelism: CosmosRLRolloutParallelismConfig
    # Forwarded to cosmos-rl's RolloutConfig.prefetch_rollout. When true,
    # cosmos-rl's _prefetch_loop calls the streaming backend's
    # `enqueue_prefetch_payloads` hook ahead of the next
    # `rollout_generation` call. Requires dp_shard_size == 1.
    prefetch_rollout: bool = True


@dataclass
class CosmosRLLoggingConfig:
    """Cosmos-RL trainer metric logging settings."""

    logger: list[str]
    log_training_metrics_every_n_steps: int
    project_name: str
    experiment_name: str


@dataclass
class CosmosRLConfig:
    """Cosmos-RL launch settings for local smoke runs."""

    mode: CosmosRLMode
    launch: CosmosRLLaunchConfig
    train: CosmosRLTrainConfig
    policy: CosmosRLPolicyConfig
    rollout: CosmosRLRolloutConfig
    logging: CosmosRLLoggingConfig


@dataclass
class RewardTermConfig:
    """One scaled scalar term in the episode reward."""

    kind: str
    scale: float
    metric_name: str | None = None

    def __post_init__(self) -> None:
        """Validate term-kind-specific required and forbidden fields."""
        if self.kind == "metric":
            if self.metric_name is None:
                raise ValueError("RewardTermConfig.kind='metric' requires metric_name")
        elif self.kind == "distance_to_gt":
            if self.metric_name is not None:
                raise ValueError("RewardTermConfig.kind='distance_to_gt' must not set metric_name")
        else:
            raise ValueError(f"Unknown RewardTermConfig.kind: {self.kind!r}")


@dataclass
class RewardConfig:
    """Total reward as a sum of scaled scalar terms."""

    terms: list[RewardTermConfig]

    def __post_init__(self) -> None:
        """Reject empty term lists; a reward needs at least one contribution."""
        if not self.terms:
            raise ValueError("RewardConfig requires at least one term")


@dataclass
class AlpaSimWizardArgs:
    """AlpaSim Wizard startup overrides authored by AlpaGym."""

    deploy: str
    topology: str
    driver_source: str
    force_gt_duration_us: int
    control_timestep_us: int
    n_sim_steps: int
    driver: str | None = None
    # Name of an alpasim `renderer` Hydra config group to activate. Leave `None`
    # to use the alpasim default NRE renderer.
    renderer: str | None = None
    # Free-form catch-all for any other Hydra-style alpasim Wizard overrides
    # (e.g. `services.renderer.environments=[...]` or deep numeric tweaks
    # under `runtime.simulation_config.*`). Shell-split before being appended
    # to the wizard argv. Keep this as the escape hatch; promote anything we
    # set unconditionally to a dedicated field above.
    extra_overrides: str = ""

    def __post_init__(self) -> None:
        """Reject empty required Wizard override values."""
        if not self.deploy:
            raise ValueError("AlpaSimWizardArgs.deploy must be non-empty")
        if not self.topology:
            raise ValueError("AlpaSimWizardArgs.topology must be non-empty")
        if not self.driver_source:
            raise ValueError("AlpaSimWizardArgs.driver_source must be non-empty")
        if self.control_timestep_us <= 0:
            raise ValueError("AlpaSimWizardArgs.control_timestep_us must be positive")
        if self.n_sim_steps <= 0:
            raise ValueError("AlpaSimWizardArgs.n_sim_steps must be positive")
        if self.driver is not None and not self.driver:
            raise ValueError("AlpaSimWizardArgs.driver must be non-empty when set")
        if self.renderer is not None and not self.renderer:
            raise ValueError("AlpaSimWizardArgs.renderer must be non-empty when set")


@dataclass
class AlpaSimConfig:
    """Host-managed AlpaSim Wizard startup settings."""

    startup_timeout_s: float
    simulation_timeout_s: float
    wizard_args: AlpaSimWizardArgs
    simulation_domain: str = "av"
    repo_url: str | None = None
    repo_ref: str | None = None
    repo_path: str | None = None
    # Optional directory for cached AlpaSim checkouts. Defaults to XDG_CACHE_HOME.
    checkout_cache_dir: str | None = None

    def __post_init__(self) -> None:
        """Reject configs that pin both an explicit repo_path and a remote repo.

        checkout_cache_dir is deliberately not part of this check: a deploy preset
        sets it unconditionally, and resolve_alpasim_checkout simply ignores it when
        repo_path is set, so rejecting it would break `repo_path=` overrides on those
        presets.
        """
        if self.repo_path is not None and (self.repo_url is not None or self.repo_ref is not None):
            raise ValueError("AlpaSimConfig.repo_path is mutually exclusive with repo_url/repo_ref")


class TransportKind(StrEnum):
    """Transport implementations the host can wire for one run."""

    disk = "disk"
    nccl = "nccl"


@dataclass
class TransportConfig:
    """Selection of the transport that carries completed rollout artifacts."""

    kind: TransportKind = TransportKind.disk
    nccl_env: dict[str, str] = field(default_factory=dict)
    nccl_read_device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate transport settings at config-load time."""
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", self.nccl_read_device):
            raise ValueError(
                "TransportConfig.nccl_read_device must be 'cpu', 'cuda', or 'cuda:<index>'; "
                f"got {self.nccl_read_device!r}"
            )


class SlurmLayout(StrEnum):
    """Supported Slurm host layouts for AlpaGym runs."""

    all_in_one = "all_in_one"
    separate_nodes = "separate_nodes"


@dataclass
class SlurmTopologyConfig:
    """Base schema for Slurm host topology variants."""

    kind: SlurmLayout = MISSING


@dataclass
class AllInOneSlurmTopologyConfig(SlurmTopologyConfig):
    """One Slurm node split between Cosmos and AlpaSim GPUs."""

    kind: SlurmLayout = SlurmLayout.all_in_one
    alpasim_gpus: int = 4


@dataclass
class SeparateNodesSlurmTopologyConfig(SlurmTopologyConfig):
    """Disjoint full-node Cosmos and AlpaSim Slurm topology."""

    kind: SlurmLayout = SlurmLayout.separate_nodes
    cosmos_nodes: int = 1
    alpasim_nodes: int = 1


@dataclass
class SlurmConfig:
    """Slurm settings for AlpaGym execution."""

    job_name: str
    partition: str | None
    account: str | None
    time: str
    nodes: int
    gpus_per_node: int
    topology: SlurmTopologyConfig
    exclusive: bool
    cpus_per_task: int | None
    container_image: str | None
    container_cache_root: str | None
    container_workdir: str
    uv_cache_dir: str | None = None
    container_mounts: list[str] = field(default_factory=list)
    export_env: list[str] = field(default_factory=list)
    qos: str | None = None
    mem: str | None = None


class ExecutionBackend(StrEnum):
    """Supported host execution backends."""

    local_process = "local_process"
    slurm = "slurm"

    @property
    def wizard_run_method(self) -> str:
        """Return the AlpaSim Wizard run method for this backend."""
        match self:
            case ExecutionBackend.local_process:
                return "DOCKER_COMPOSE"
            case ExecutionBackend.slurm:
                return "SLURM"

    @property
    def is_slurm_run(self) -> bool:
        """Return whether this backend runs through Slurm."""
        return self is ExecutionBackend.slurm


@dataclass
class ExecutionConfig:
    """Host execution settings for local and Slurm runs.

    `resolved_config_path` is only set when executing a run that was already
    prepared. A null value means `command=run` should prepare a new run from the
    Hydra-composed config before executing it.
    """

    backend: ExecutionBackend
    resolved_config_path: str | None
    slurm: SlurmConfig


@dataclass
class RunConfigSchema:
    """Root host config schema registered with Hydra."""

    command: str
    run_root: str
    logging_level: LoggingLevel
    execution: ExecutionConfig
    dataset: DatasetConfig
    policy: AlpamayoPolicyConfig
    reward: RewardConfig
    cosmos: CosmosRLConfig
    alpasim: AlpaSimConfig
    # Parent dir for this deploy's writable caches: the uv cache
    # (execution.slurm.uv_cache_dir) and the AlpaSim checkout cache
    # (alpasim.checkout_cache_dir) are placed in subdirs of this, sharing one
    # writable location.
    cache_root_dir: str | None
    expected_valid_steps: int
    perf: PerfConfig
    # kw_only=True keeps RunConfig's `artifact_paths` (no default) valid
    # under dataclass inheritance; otherwise a defaulted field on the parent
    # would force every subclass field to also carry a default.
    transport: TransportConfig = field(default_factory=TransportConfig, kw_only=True)


@dataclass
class RunConfig(RunConfigSchema):
    """Run config loaded from a host-written resolved config artifact.

    Inherits all fields from `RunConfigSchema` and adds generated artifact paths.
    """

    artifact_paths: ArtifactPaths


RunConfigT = TypeVar("RunConfigT", bound=RunConfig)


def register_config_schema() -> None:
    """Register the root host config schema with Hydra's ConfigStore."""
    OmegaConf.register_new_resolver(
        "alpasim_grpc_repo_ref",
        _resolve_alpasim_grpc_repo_ref,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "current_alpagym_project_root",
        lambda: str(alpagym_project_root()),
        replace=True,
    )
    OmegaConf.register_new_resolver("add", lambda a, b: a + b, replace=True)
    OmegaConf.register_new_resolver("floordiv", lambda a, b: a // b, replace=True)
    config_store = ConfigStore.instance()
    config_store.store(name="config_schema", node=RunConfigSchema)
    config_store.store(
        group="execution/slurm/topology",
        name="all_in_one",
        node=AllInOneSlurmTopologyConfig,
        package="execution.slurm.topology",
    )
    config_store.store(
        group="execution/slurm/topology",
        name="separate_nodes",
        node=SeparateNodesSlurmTopologyConfig,
        package="execution.slurm.topology",
    )


def _resolve_alpasim_grpc_repo_ref() -> str:
    """Read the AlpaSim grpc dependency rev pinned by the workspace."""
    workspace_pyproject = alpagym_project_root() / "pyproject.toml"
    workspace_project = tomllib.loads(workspace_pyproject.read_text(encoding="utf-8"))
    return workspace_project["tool"]["uv"]["sources"]["alpasim-grpc"]["rev"]


def alpagym_project_root() -> Path:
    """Return the current AlpaGym project checkout root."""
    return Path(__file__).resolve().parents[4]


def load_run_config(path: str | Path) -> RunConfig:
    """Load a host-written resolved config artifact as typed config.

    Args:
        path: Path to `resolved_config.yaml`.

    Returns:
        Typed run config, including generated artifact paths.
    """
    raw_data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return merge_run_config_schema(RunConfig, raw_data)


def merge_run_config_schema(schema_type: type[RunConfigT], raw_data: object) -> RunConfigT:
    """Merge raw run config data into a typed schema."""
    raw_config = OmegaConf.create(cast(Any, raw_data))
    raw_config.execution.slurm.topology = OmegaConf.merge(
        _structured_slurm_topology(raw_config.execution.slurm.topology),
        raw_config.execution.slurm.topology,
    )
    merged_config = OmegaConf.merge(OmegaConf.structured(schema_type), raw_config)
    run_config = OmegaConf.to_object(merged_config)
    if not isinstance(run_config, schema_type):
        raise TypeError(type(run_config))
    return cast(RunConfigT, run_config)


def _structured_slurm_topology(raw_topology: object) -> object:
    """Return the structured schema matching a raw Slurm topology discriminator."""
    topology = OmegaConf.create(cast(Any, raw_topology))
    match SlurmLayout(topology.kind):
        case SlurmLayout.all_in_one:
            return OmegaConf.structured(AllInOneSlurmTopologyConfig)
        case SlurmLayout.separate_nodes:
            return OmegaConf.structured(SeparateNodesSlurmTopologyConfig)
