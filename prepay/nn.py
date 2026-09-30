"""Feed-forward neural network (PyTorch), following the paper's architecture:

    input -> 128 -> 256 -> 1
    batch norm + ReLU between hidden layers, sigmoid on the output
    binary cross-entropy with a positive-class weight (paper: 10)
    SGD, learning rate 0.1 multiplied by 0.91 after every epoch, 30 epochs

Change from the paper: the epoch with the best validation PR-AUC is kept
(early-stopping style) instead of whatever the last epoch produced.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Net(nn.Module):
    def __init__(self, n_in: int, hidden: list[int]):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))  # sigmoid is applied inside the loss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NeuralNet:
    def __init__(self, cfg: dict, seed: int):
        self.cfg, self.seed = cfg, seed

    def fit(self, X, y, Xv, yv):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = self.device = _device(self.cfg["device"])
        print(f"      training on {dev}")
        self.n_in = X.shape[1]
        self.model = _Net(self.n_in, self.cfg["hidden"]).to(dev)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.cfg["lr"])
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.cfg["lr_decay"])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.cfg["pos_weight"], device=dev))
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y.astype(np.float32))
        bs = self.cfg["batch_size"]
        best_ap, best_state = -1.0, None
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.model.train()
            perm = torch.from_numpy(rng.permutation(len(y)))
            total = 0.0
            for i in range(0, len(y), bs):
                idx = perm[i : i + bs]
                if len(idx) < 2:  # batch norm needs >1 sample
                    continue
                xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            sched.step()
            ap = average_precision_score(yv, self.score(Xv))
            if ap > best_ap:
                best_ap, best_state, self.best_epoch = ap, copy.deepcopy(self.model.state_dict()), epoch
            if epoch == 1 or epoch % 5 == 0 or epoch == self.cfg["epochs"]:
                print(f"      epoch {epoch:3d}  loss {total / len(y):.4f}  val PR-AUC {ap:.4f}")
        self.model.load_state_dict(best_state)
        print(f"      kept epoch {self.best_epoch} (best val PR-AUC {best_ap:.4f})")
        return self

    # Pickle the weights as CPU tensors so a model trained on a GPU loads anywhere.
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("device", None)
        if "model" in state:
            state["model"] = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        return state

    def __setstate__(self, state):
        weights = state.pop("model", None)
        self.__dict__.update(state)
        self.device = torch.device("cpu")
        if weights is not None:
            self.model = _Net(self.n_in, self.cfg["hidden"])
            self.model.load_state_dict(weights)
            self.model.eval()

    @torch.no_grad()
    def score(self, X):
        self.model.eval()
        out = []
        for i in range(0, len(X), 65536):
            out.append(self.model(torch.from_numpy(X[i : i + 65536]).to(self.device)).float().cpu().numpy())
        return np.concatenate(out)
