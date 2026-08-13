"""权重解析顺序：显式参数 > VGGT_WEIGHT_PATH > HuggingFace Hub。"""
import os

import pytest

from diffusion_policy_3d.model.vision.vggt.models.vggt_enc import resolve_vggt_weight


def test_explicit_path_wins(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.safetensors"
    explicit.write_bytes(b"x")
    other = tmp_path / "env.safetensors"
    other.write_bytes(b"x")
    monkeypatch.setenv("VGGT_WEIGHT_PATH", str(other))
    assert resolve_vggt_weight(str(explicit)) == str(explicit)


def test_env_var_used(tmp_path, monkeypatch):
    weight = tmp_path / "w.safetensors"
    weight.write_bytes(b"x")
    monkeypatch.setenv("VGGT_WEIGHT_PATH", str(weight))
    assert resolve_vggt_weight() == str(weight)


def test_missing_path_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("VGGT_WEIGHT_PATH", str(tmp_path / "nope.safetensors"))
    with pytest.raises(FileNotFoundError):
        resolve_vggt_weight()
