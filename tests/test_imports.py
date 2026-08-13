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


def test_paper_defaults_are_opt_in():
    """两个历史陷阱：vggtq 和 flow_matching 曾默认 True。

    vggtq=True 会静默 int4 量化编码器，flow_matching=True 会把采样从 DDIM
    换成 MeanFlow。任何一个默认打开，旧 checkpoint 都会加载失败或跑出 0 成功率。
    """
    import inspect

    from diffusion_policy_3d.policy.dp import DP, DPEncoder

    assert inspect.signature(DP.__init__).parameters["vggtq"].default is False
    assert inspect.signature(DP.__init__).parameters["flow_matching"].default is False
    assert inspect.signature(DPEncoder.__init__).parameters["vggtq"].default is False
