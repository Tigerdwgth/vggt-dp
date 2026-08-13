# https://github.com/haidog-yaqub/MeanFlow/blob/main/meanflow.py
import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np


class Normalizer:
    # minmax for raw image, mean_std for vae latent
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        """
        config: [mode, mean, std]
        """
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, ...) - can be any shape with batch dimension first
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    # Calculate mean over all dimensions except batch dimension
    dims_to_reduce = list(range(1, len(error.shape)))
    delta_sq = torch.mean(error ** 2, dim=dims_to_reduce, keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class MeanFlow:
    def __init__(
        self,
        channels=1,
        image_size=32,
        num_classes=10,
        normalizer=['minmax', None, None],
        # mean flow settings
        flow_ratio=0.50,
        # time distribution, mu, sigma
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        # set scale as none to disable CFG distill
        cfg_scale=2.0,
        # experimental
        cfg_uncond='v',
        jvp_api='autograd',
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None

        self.normer = Normalizer.from_list(normalizer)

        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale

        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    # fix: r should be always not larger than t
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, model, x, c=None):
        batch_size = x.shape[0]
        device = x.device

        t, r = self.sample_t_r(batch_size, device)

        # 调整维度以适应动作数据 (B, T, A) 而不是图像数据 (B, C, H, W)
        if len(x.shape) == 3:  # (B, T, A)
            t_ = rearrange(t, "b -> b 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1").detach().clone()
        else:  # 保持原来的图像格式
            t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

        e = torch.randn_like(x)
        x = self.normer.norm(x)

        z = (1 - t_) * x + t_ * e
        v = e - x
        v_hat = v  # 默认值

        if c is not None:
            assert self.cfg_ratio is not None
            # 对于动作生成，使用零向量作为无条件输入
            uncond = torch.zeros_like(c)
            # 生成随机mask，但只在batch维度上应用
            if len(c.shape) > 1:
                # c是多维的，如 (batch_size, condition_dim)
                cfg_mask = torch.rand(c.shape[0], device=c.device) < self.cfg_ratio
                cfg_mask_expanded = cfg_mask.unsqueeze(-1).expand_as(c)
                c = torch.where(cfg_mask_expanded, uncond, c)
            else:
                # c是一维的，如 (batch_size,)
                cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
                c = torch.where(cfg_mask, uncond, c)
                
            if self.w is not None:
                with torch.no_grad():
                    u_t = model(sample=z, timestep=t, r_timestep=t, global_cond=uncond)
                v_hat = self.w * v + (1 - self.w) * u_t
                if self.cfg_uncond == 'v':
                    # official JAX repo uses original v for unconditional items
                    # 使用batch维度的mask
                    if len(c.shape) > 1:
                        mask_to_use = cfg_mask  # 已经是(batch_size,)的形状
                    else:
                        mask_to_use = cfg_mask
                        
                    if len(x.shape) == 3:  # (B, T, A)
                        mask_to_use = rearrange(mask_to_use, "b -> b 1 1").bool()
                    else:  # (B, C, H, W)
                        mask_to_use = rearrange(mask_to_use, "b -> b 1 1 1").bool()
                    v_hat = torch.where(mask_to_use, v, v_hat)

        # forward pass
        model_partial = partial(model, global_cond=c)
        jvp_args = (
            lambda z, t, r: model_partial(sample=z, timestep=t, r_timestep=r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt

        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        mse_val = (stopgrad(error) ** 2).mean()
        return loss, mse_val

    @torch.no_grad()
    def sample_action(self, model, c: torch.Tensor, sample_steps=5, device='cuda', action_dim=None, horizon=None):
        model.eval()
        # 从dp.py中获取正确的维度信息
        batch_size = c.shape[0]
        
        # 如果没有提供action_dim和horizon，尝试从model或其他地方推断
        if action_dim is None:
            action_dim = 4  # 默认动作维度，可以从model.action_dim获取
        if horizon is None:
            horizon = 16   # 默认时间步长，可以从model.horizon获取
            
        z = torch.randn(batch_size, horizon, action_dim, device=device)
        
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i].item(), device=device)
            r = torch.full((z.size(0),), t_vals[i + 1].item(), device=device)
            v = model(sample=z, timestep=t, r_timestep=r, global_cond=c)
            t_ = t.view(-1, 1, 1)
            r_ = r.view(-1, 1, 1)
            z = z - (t_ - r_) * v
        return z
        
    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, classes=None,
                          sample_steps=5, device='cuda'):
        model.eval()

        if classes is None:
            c = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        else:
            c = torch.tensor(classes, device=device).repeat(n_per_class)

        z = torch.randn(c.shape[0], self.channels,
                        self.image_size, self.image_size,
                        device=device)

        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        # print(t_vals)

        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i].item(), device=device)
            r = torch.full((z.size(0),), t_vals[i + 1].item(), device=device)

            # print(f"t: {t[0].item():.4f};  r: {r[0].item():.4f}")

            t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

            v = model(sample=z, timestep=t, r_timestep=r, global_cond=c)
            z = z - (t_ - r_) * v

        z = self.normer.unnorm(z)
        return z