# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the runtime policy bundle checker."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from alpagym_runtime.policies import check_policy_bundles
from alpagym_runtime.policies.registry import PolicyBundle


def _setup_tokenizer(config: object) -> None:
    """Accept a tokenizer config."""
    del config


def _install_runtime_bridge() -> None:
    """Accept runtime bridge installation."""


def _build_data_packer(run_config: object, cosmos_role: str | None) -> None:
    """Accept data packer construction."""
    del run_config, cosmos_role


def _load_inference_model(run_config: object, device: object, dtype: object) -> None:
    """Accept inference model loading."""
    del run_config, device, dtype


def _build_model_inputs(run_config: object) -> None:
    """Accept trainer input builder construction."""
    del run_config


def _bundle(
    *,
    setup_tokenizer=_setup_tokenizer,
    build_data_packer=_build_data_packer,
    install_runtime_bridge=_install_runtime_bridge,
    load_inference_model=_load_inference_model,
    build_model_inputs=_build_model_inputs,
) -> PolicyBundle:
    """Build a minimal policy bundle for checker tests."""
    return PolicyBundle(
        setup_tokenizer=setup_tokenizer,
        build_data_packer=build_data_packer,
        install_runtime_bridge=install_runtime_bridge,
        load_inference_model=load_inference_model,
        build_model_inputs=build_model_inputs,
    )


def test_check_policy_bundles_loads_each_installed_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The all-bundles check loads and validates each installed entry point."""
    monkeypatch.setattr(
        check_policy_bundles.policy_bundles,
        "get_entry_points",
        lambda: [
            SimpleNamespace(name="b_policy"),
            SimpleNamespace(name="a_policy"),
        ],
    )
    loaded: list[str] = []

    def fake_get_policy_bundle(kind: str) -> PolicyBundle:
        loaded.append(kind)
        return _bundle()

    monkeypatch.setattr(
        check_policy_bundles, "get_policy_bundle", fake_get_policy_bundle
    )

    checked = check_policy_bundles.check_policy_bundles()

    assert checked == ["a_policy", "b_policy"]
    assert loaded == ["a_policy", "b_policy"]


def test_check_policy_bundles_rejects_non_callable_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loaded bundles must expose callable hooks."""
    monkeypatch.setattr(
        check_policy_bundles.policy_bundles,
        "get_entry_points",
        lambda: [SimpleNamespace(name="installed")],
    )
    monkeypatch.setattr(
        check_policy_bundles,
        "get_policy_bundle",
        lambda kind: _bundle(load_inference_model=None),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="load_inference_model"):
        check_policy_bundles.check_policy_bundles()


def test_policy_bundle_rejects_non_callable_optional_export_hook() -> None:
    with pytest.raises(TypeError, match="export_model_checkpoint"):
        PolicyBundle(
            setup_tokenizer=_setup_tokenizer,
            build_data_packer=_build_data_packer,
            install_runtime_bridge=_install_runtime_bridge,
            load_inference_model=_load_inference_model,
            build_model_inputs=_build_model_inputs,
            export_model_checkpoint="not-callable",  # type: ignore[arg-type]
        )
