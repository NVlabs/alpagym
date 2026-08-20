# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from alpagym_g1_vla.provenance import canonical_tree_sha256, file_sha256


def test_canonical_tree_hash_is_order_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "b.py").write_text("b = 2\n", encoding="utf-8")
    (tree / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tree / "ignored.txt").write_text("ignored\n", encoding="utf-8")

    first = canonical_tree_sha256(tree, format_name="test.v1", suffix=".py")
    (tree / "ignored.txt").write_text("changed\n", encoding="utf-8")
    assert canonical_tree_sha256(tree, format_name="test.v1", suffix=".py") == first

    (tree / "a.py").write_text("a = 3\n", encoding="utf-8")
    assert canonical_tree_sha256(tree, format_name="test.v1", suffix=".py") != first


def test_tree_and_file_hash_reject_symlinks(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    target = tree / "target.py"
    target.write_text("pass\n", encoding="utf-8")
    link = tree / "alias.py"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlinks"):
        canonical_tree_sha256(tree, format_name="test.v1", suffix=".py")
    with pytest.raises(FileNotFoundError, match="not regular"):
        file_sha256(link)
