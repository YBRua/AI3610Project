import math
import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal


class Perturbator():
    def __init__(
            self,
            model: nn.Module,
            device: torch.device) -> None:
        self.model = model
        self.device = device
        self.perturb_dict = {}

    def perturb_model(self):
        raise NotImplementedError()

    def restore_perturbation(self):
        raise NotImplementedError()

    def step(self):
        return


class DeviceFaultPerturbator(Perturbator):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            hrs_proportion: float = 0.1,
            lrs_proportion: float = 0.1,
            scheduled: bool = False) -> None:
        super().__init__(model, device)
        self.hrs_proportion = hrs_proportion / 2
        self.lrs_proportion = lrs_proportion / 2
        self.scheduled = scheduled

    def _perturb_idx(self, x: torch.Tensor, proportion: float):
        idx = torch.zeros_like(x).view(-1)
        chosen = np.random.choice(
            np.arange(torch.numel(idx)),
            replace=False,
            size=int(torch.numel(idx) * proportion),
        )
        idx[chosen] = 1
        return idx.reshape(x.shape) == 1

    def step(self):
        if self.scheduled:
            self.lrs_proportion += 0.005
            self.hrs_proportion += 0.005

    def perturb_model(self):
        self.perturb_dict = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param_ = param.clone()

                pos, neg = param.clone(), param.clone() * -1
                pos[pos < 0] = 0
                neg[neg < 0] = 0
                pos_max = pos.max()
                neg_max = neg.max()
                pos_min = pos.min()
                neg_min = neg.min()

                pos_maxed = self._perturb_idx(pos, self.hrs_proportion)
                neg_maxed = self._perturb_idx(neg, self.hrs_proportion)
                pos_mined = self._perturb_idx(pos, self.lrs_proportion)
                neg_mined = self._perturb_idx(neg, self.lrs_proportion)

                param[pos_maxed] = pos_max
                param[neg_maxed] = - neg_max
                param[pos_mined] = pos_min
                param[neg_mined] = - neg_min

                self.perturb_dict[name] = (
                    param_, pos_maxed, neg_maxed, pos_mined, neg_mined)

    def restore_perturbation(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                saved = self.perturb_dict[name]
                param_, p_maxed, n_maxed, p_mined, n_mined = saved
                param[p_maxed] = param_[p_maxed]
                param[n_maxed] = param_[n_maxed]
                param[p_mined] = param_[p_mined]
                param[n_mined] = param_[n_mined]


class ScheduledExpGaussianPerturbator(Perturbator):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            mean: float = 0.0,
            std_start: float = 0.8,
            std_end: float = 1,
            steps: int = 3,
            eps: float = 1e-5) -> None:
        super().__init__(model, device)
        self.mean = mean
        self.eps = eps
        self.schedule = torch.log(
            torch.linspace(math.exp(std_start), math.exp(std_end), steps)
        ).tolist()
        self.current_step = 0

    def _exp_gaussian_like(self, x: torch.Tensor, std: float):
        sampler = Normal(self.mean, std)
        return torch.exp(sampler.sample(x.shape).to(x.device))

    def perturb_model(self):
        self.perturb_dict = {}
        if self.current_step < len(self.schedule):
            std = self.schedule[self.current_step]
        else:
            std = self.schedule[-1]
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                noise = self._exp_gaussian_like(param, std)
                self.perturb_dict[name] = noise
                param.mul_(noise + self.eps)

    def restore_perturbation(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.mul_(1 / (self.perturb_dict[name] + self.eps))

    def step(self):
        self.current_step += 1


class ExpGaussianPerturbator(Perturbator):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            mean: float = 0.0,
            std: float = 1.0,
            eps: float = 1e-5) -> None:
        super().__init__(model, device)
        self.mean = mean
        self.std = std
        self.sampler = Normal(mean, std)
        self.eps = eps

    def _exp_gaussian_like(self, x: torch.Tensor):
        return torch.exp(self.sampler.sample(x.shape).to(x.device))

    def perturb_model(self):
        self.perturb_dict = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                noise = self._exp_gaussian_like(param)
                self.perturb_dict[name] = noise
                param.mul_(noise + self.eps)

    def restore_perturbation(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.mul_(1 / (self.perturb_dict[name] + self.eps))
