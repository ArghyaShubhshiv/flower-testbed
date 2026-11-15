"""
Modular Byzantine attacks (numpy-only).
  data   : poison local batches             (label_flip)
  update : tamper with returned delta       (sign_flip, scale, gaussian, zero)
  server : omniscient/colluding, crafted from benign updates, simulated in the
           strategy, replacing Byzantine clients' updates (alie)

Env (same on server + clients):
  ATTACK             none | label_flip | sign_flip | scale | gaussian | zero | alie
  NUM_BYZANTINE      clients with CLIENT_ID < NUM_BYZANTINE are Byzantine
  ATTACK_SCALE       strength (see DEFAULT_SCALE)
  ATTACK_START_ROUND first active round (default 1)
Only trainable params are attacked; BatchNorm buffers are left untouched.
"""
import os
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

DATA_ATTACKS = {}
UPDATE_ATTACKS = {}
SERVER_ATTACKS = {}

DEFAULT_SCALE = {"sign_flip": 1.0, "scale": 10.0, "gaussian": 1.0, "label_flip": 1.0,
                 "zero": 1.0, "alie": 1.0}


def _register(registry, name):
    def deco(fn):
        registry[name] = fn
        return fn
    return deco


@dataclass
class AttackConfig:
    name: str = "none"
    num_byzantine: int = 0
    scale: float = 1.0
    start_round: int = 1
    num_classes: int = 100

    @classmethod
    def from_env(cls):
        name = os.environ.get("ATTACK", "none").lower()
        if name not in ATTACK_NAMES:
            raise ValueError(f"Unknown ATTACK '{name}'. Choose from {ATTACK_NAMES}")
        scale = os.environ.get("ATTACK_SCALE")
        return cls(
            name=name,
            num_byzantine=int(os.environ.get("NUM_BYZANTINE", "0")) if name != "none" else 0,
            scale=float(scale) if scale else DEFAULT_SCALE.get(name, 1.0),
            start_round=int(os.environ.get("ATTACK_START_ROUND", "1")),
        )

    @property
    def enabled(self):
        return self.name != "none" and self.num_byzantine > 0

    def is_byzantine(self, client_id: int) -> bool:
        return self.enabled and client_id < self.num_byzantine

    def active(self, server_round: int) -> bool:
        return self.enabled and server_round >= self.start_round


# ---- data attacks
@_register(DATA_ATTACKS, "label_flip")
def label_flip(images, labels, cfg):
    """y -> (C-1-y)."""
    return images, (cfg.num_classes - 1) - labels


def poison_batch(images, labels, cfg: AttackConfig):
    fn = DATA_ATTACKS.get(cfg.name)
    return fn(images, labels, cfg) if fn else (images, labels)


# ---- update attacks: fn(deltas, mask, cfg, rng) -> deltas
def _apply(deltas, mask, f):
    return [f(d) if (m and np.issubdtype(d.dtype, np.floating)) else d
            for d, m in zip(deltas, mask)]


@_register(UPDATE_ATTACKS, "sign_flip")
def sign_flip(deltas, mask, cfg, rng):
    return _apply(deltas, mask, lambda d: -cfg.scale * d)


@_register(UPDATE_ATTACKS, "scale")
def scale(deltas, mask, cfg, rng):
    """Model-replacement style boosting."""
    return _apply(deltas, mask, lambda d: cfg.scale * d)


@_register(UPDATE_ATTACKS, "gaussian")
def gaussian(deltas, mask, cfg, rng):
    """Noise with std = scale * per-tensor std of the honest delta."""
    return _apply(deltas, mask,
                  lambda d: rng.normal(0.0, cfg.scale * (d.std() + 1e-8), d.shape).astype(d.dtype))


@_register(UPDATE_ATTACKS, "zero")
def zero(deltas, mask, cfg, rng):
    """Free-rider: zero update, keeps its num_examples weight."""
    return _apply(deltas, mask, np.zeros_like)


def poison_update(deltas, mask, cfg: AttackConfig, rng):
    fn = UPDATE_ATTACKS.get(cfg.name)
    return fn(deltas, mask, cfg, rng) if fn else deltas


# ---- server-side (omniscient) attacks: fn(benign, n_byz, mask, cfg) -> shared malicious update
@_register(SERVER_ATTACKS, "alie")
def alie(benign, n_byz, mask, cfg):
    """'A Little Is Enough' (Baruch et al. 2019): mean - z*std per coordinate."""
    n = len(benign) + n_byz
    s = n // 2 + 1 - n_byz
    z = NormalDist().inv_cdf(min(max((n - s) / n, 0.5001), 0.9999)) * cfg.scale
    out = []
    for i, m in enumerate(mask):
        stack = np.stack([b[i] for b in benign]).astype(np.float64)
        mu, sd = stack.mean(0), stack.std(0)
        vec = mu - z * sd if m and np.issubdtype(benign[0][i].dtype, np.floating) else mu
        out.append(vec.astype(benign[0][i].dtype))
    return out


ATTACK_NAMES = ["none"] + sorted({*DATA_ATTACKS, *UPDATE_ATTACKS, *SERVER_ATTACKS})
