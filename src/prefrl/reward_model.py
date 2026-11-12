"""Reward predictor ensemble fit to pairwise preferences (paper 2.2.3)."""

import numpy as np
import torch
from torch import nn


class PrefDB:
    """The preference database D of triples (segment 1, segment 2, mu), mu = P(segment 1 preferred) in {0, 0.5, 1}.

    Each triple gets a Poisson(1) count per ensemble member. That is the online form of "each predictor is trained on
    |D| triples sampled from D with replacement", and the triples a member draws zero times (a fraction 1/e) are
    exactly its held-out validation set.
    """

    def __init__(self, n_members, rng):
        self.n_members, self.rng = n_members, rng
        self.obs, self.act, self.mu, self.w = [], [], [], []

    def add(self, obs, act, mu):
        self.obs.append(obs), self.act.append(act), self.mu.append(mu)
        self.w.append(self.rng.poisson(1.0, self.n_members))

    def __len__(self):
        return len(self.mu)

    def arrays(self):
        return (np.stack(self.obs), np.stack(self.act), np.array(self.mu, np.float32), np.stack(self.w).astype(np.float32))


class RewardEnsemble(nn.Module):
    """r-hat(o, a): `n_members` MLPs with two hidden layers of 64 leaky-ReLU units (paper A.1)."""

    def __init__(self, obs_dim, n_actions, n_members=3, hidden=64, lr=1e-3, label_noise=0.1, l2_init=1e-4, val_ratio=(1.1, 1.5)):
        super().__init__()
        self.n_actions, self.label_noise, self.val_ratio = n_actions, label_noise, val_ratio
        self.nets = nn.ModuleList(
            nn.Sequential(
                nn.Linear(obs_dim + n_actions, hidden), nn.LeakyReLU(0.01),
                nn.Linear(hidden, hidden), nn.LeakyReLU(0.01),
                nn.Linear(hidden, 1),
            )
            for _ in range(n_members)
        )  # fmt: skip
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)  # members share no parameters, so one Adam = one each
        self.register_buffer("l2", torch.full((n_members,), l2_init))
        self.register_buffer("mean", torch.zeros(n_members))
        self.register_buffer("std", torch.ones(n_members))
        self.register_buffer("scale", torch.ones(()))

    def raw(self, obs, act):
        """Unnormalised reward of every member: obs (..., d), act (...) -> (members, ...)."""
        onehot = nn.functional.one_hot(torch.as_tensor(act), self.n_actions)
        x = torch.cat([torch.as_tensor(obs, dtype=torch.float32), onehot.float()], -1)
        return torch.stack([net(x).squeeze(-1) for net in self.nets])

    @torch.no_grad()
    def reward(self, obs, act):
        """The reward the agent optimises, for a flat batch: each member normalised on its own, then averaged
        (2.2.3), then rescaled to zero mean and unit standard deviation (2.2.1, A.1)."""
        r = (self.raw(obs, act) - self.mean[:, None]) / self.std[:, None]
        return (r.mean(0) / self.scale).numpy()

    @torch.no_grad()
    def set_normalisation(self, obs, act):
        """Recompute the normalisation constants on a flat batch of recent transitions."""
        r = self.raw(obs, act)
        self.mean.copy_(r.mean(1))
        self.std.copy_(r.std(1) + 1e-8)
        self.scale.copy_(((r - self.mean[:, None]) / self.std[:, None]).mean(0).std() + 1e-8)

    def prefer_first(self, obs, act):
        """Each member's P[segment 1 preferred]: obs (n, 2, k, d), act (n, 2, k) -> (members, n).

        Eq. 1 (Bradley-Terry on summed rewards), mixed with a `label_noise` chance of a uniformly random answer.
        """
        ret = self.raw(obs, act).sum(-1)  # (members, n, 2)
        return (1 - self.label_noise) * torch.sigmoid(ret[..., 0] - ret[..., 1]) + self.label_noise / 2

    def _losses(self, obs, act, mu, w):
        """Per-member (train, validation) cross-entropy on a batch, by bootstrap count w (n, members)."""
        p = self.prefer_first(obs, act)
        ce = -(mu * p.log() + (1 - mu) * (1 - p).log())  # (members, n)
        w, held_out = w.T, (w.T == 0).float()
        return (w * ce).sum(1) / w.sum(1).clamp(min=1), (held_out * ce).sum(1) / held_out.sum(1).clamp(min=1)

    def fit(self, db, epochs, batch_size, rng):
        """Train every member on its bootstrap sample of D. Returns the final mean (train loss, validation loss).

        After each epoch a member's l2 coefficient moves up when its validation loss exceeds 1.5x its training loss
        and down when it is below 1.1x (2.2.3).
        """
        obs, act, mu, w = (torch.as_tensor(x) for x in db.arrays())
        lo, hi = self.val_ratio
        for _ in range(epochs):
            for idx in torch.as_tensor(rng.permutation(len(mu))).split(batch_size):
                train, _ = self._losses(obs[idx], act[idx], mu[idx], w[idx])
                sq_norm = torch.stack([sum((p**2).sum() for p in net.parameters()) for net in self.nets])
                self.opt.zero_grad()
                (train + self.l2 * sq_norm).sum().backward()
                self.opt.step()
            with torch.no_grad():
                train, val = self._losses(obs, act, mu, w)
                has_val = (w == 0).any(0)
                self.l2[has_val & (val > hi * train)] *= 1.1
                self.l2[has_val & (val < lo * train)] /= 1.1
                self.l2.clamp_(1e-7, 1e-1)  # a runaway coefficient would flatten the member to zero
        return train.mean().item(), val[has_val].mean().item() if has_val.any() else float("nan")
