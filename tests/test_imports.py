"""每个对外模块都必须可导入 -- 裁剪仓库时最容易漏掉的就是残留的跨模块引用。"""
import importlib

import pytest

MODULES = [
    "diffusion_policy_3d.policy.dp",
    "diffusion_policy_3d.policy.base_policy",
    "diffusion_policy_3d.model.vision.vggt.models.vggt_enc",
    "diffusion_policy_3d.model.vision.pointnet_extractor",
    "diffusion_policy_3d.model.diffusion.conditional_unet1d",
    "diffusion_policy_3d.model.diffusion.ema_model",
    "diffusion_policy_3d.model.common.normalizer",
    "diffusion_policy_3d.dataset.metaworld_dataset",
    "diffusion_policy_3d.env",
    "diffusion_policy_3d.env_runner.metaworld_runner",
    "diffusion_policy_3d.common.replay_buffer",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_dp_policy_exposes_encoder():
    from diffusion_policy_3d.policy.dp import DP, DPEncoder

    assert hasattr(DP, "predict_action")
    assert hasattr(DPEncoder, "forward")


def test_no_quantization_by_default():
    """vggtq 必须默认关闭：论文之前的版本默认 True，会静默量化编码器。"""
    import inspect

    from diffusion_policy_3d.policy.dp import DP, DPEncoder

    for cls in (DP, DPEncoder):
        assert inspect.signature(cls.__init__).parameters["vggtq"].default is False
