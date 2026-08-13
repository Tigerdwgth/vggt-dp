"""配置必须能被 hydra 组合出来，且 _target_ 指向真实存在的类。"""
import pathlib

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "diffusion_policy_3d" / "config")
TASKS = sorted(p.stem for p in pathlib.Path(CONFIG_DIR, "task").glob("metaworld_*.yaml"))


def _compose(task):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="vggt_dp", overrides=["task=" + task])


def test_task_configs_exist():
    assert len(TASKS) >= 40, TASKS


@pytest.mark.parametrize("task", TASKS)
def test_task_config_composes(task):
    cfg = _compose(task)
    assert cfg.task.name
    assert cfg.task.dataset.zarr_path.endswith(".zarr")
    assert cfg.task.env_runner.eval_episodes > 0


def test_policy_targets_resolve():
    cfg = _compose("metaworld_hand-insert")
    for target in (cfg.policy._target_, cfg.task.env_runner._target_, cfg.task.dataset._target_):
        assert get_class(target) is not None


def test_paper_defaults():
    cfg = _compose("metaworld_hand-insert")
    assert cfg.policy.enc_type == "vggt"
    assert cfg.policy.vggtq is False
    assert cfg.policy.flow_matching is False
    assert cfg.policy.vggt_depth == 24
    assert cfg.n_obs_steps == 2 and cfg.horizon == 16 and cfg.n_action_steps == 8
