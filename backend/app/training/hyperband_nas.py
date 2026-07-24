"""Hyperband neural architecture search (autoPyTorch-style BOHB for tabular regression)."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class NASConfig:
    max_runtime: int = 900
    min_budget_epochs: int = 8
    max_budget_epochs: int = 60
    eta: int = 3
    n_configs: int = 32
    seed: int = 42
    # Architecture search space. Defaults keep the original small space used by
    # the ΔΔG NAS (train_ddg); the dG production search widens these so NAS can
    # reach — and beat — the hand-picked [1024,512,256,128] tower.
    width_choices: tuple[int, ...] = (64, 128, 256, 480)
    min_layers: int = 2
    max_layers: int = 4

    def __post_init__(self) -> None:
        # Scale search intensity with wall-clock budget (~30 configs per 15 min)
        self.n_configs = max(self.n_configs, int(self.max_runtime / 30))
        self.max_budget_epochs = max(self.max_budget_epochs, min(80, self.max_runtime // 20))


@dataclass
class TrialResult:
    config: dict
    budget_epochs: int
    val_loss: float
    state_dict: dict
    metrics: dict = field(default_factory=dict)


def _sample_config(
    rng: np.random.Generator,
    input_dim: int,
    width_choices: tuple[int, ...] = (64, 128, 256, 480),
    min_layers: int = 2,
    max_layers: int = 4,
) -> dict:
    n_layers = int(rng.integers(min_layers, max_layers + 1))
    widths = [int(rng.choice(width_choices)) for _ in range(n_layers)]
    return {
        "widths": widths,
        "dropout": float(rng.uniform(0.1, 0.5)),
        "lr": float(10 ** rng.uniform(-4, -2.5)),
        "weight_decay": float(10 ** rng.uniform(-6, -3)),
        "batch_size": int(rng.choice([64, 128, 256])),
        "input_dim": input_dim,
    }


class _DynamicMLP(nn.Module):
    def __init__(self, input_dim: int, widths: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.BatchNorm1d(input_dim)]
        prev = input_dim
        for width in widths:
            layers.extend([nn.Linear(prev, width), nn.GELU(), nn.Dropout(dropout)])
            prev = width
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x)).squeeze(-1)


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    r, _ = pearsonr(pred, target) if len(pred) > 1 else (0.0, 1.0)
    return {"mae": mae, "rmse": rmse, "pearson_r": float(r)}


def _train_one(
    config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    device: str,
) -> TrialResult:
    torch.manual_seed(config.get("_seed", 0))
    model = _DynamicMLP(config["input_dim"], config["widths"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    criterion = nn.MSELoss()

    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=True)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_np = y_val.astype(np.float64)

    best_state = None
    best_val = float("inf")

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t).cpu().numpy()
        val_loss = float(np.mean((val_pred - y_val_np) ** 2))
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val_t).cpu().numpy()

    return TrialResult(
        config=config,
        budget_epochs=epochs,
        val_loss=best_val,
        state_dict=best_state or model.state_dict(),
        metrics=_metrics(val_pred, y_val_np),
    )


class HyperbandRegressor:
    """Successive-halving NAS over PyTorch MLPs (autoPyTorch BOHB methodology)."""

    def __init__(self, nas_config: NASConfig | None = None, device: str = "cpu") -> None:
        self.nas_config = nas_config or NASConfig()
        self.device = device
        self.best_trial: TrialResult | None = None
        self.search_history: list[dict] = []

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrialResult:
        cfg = self.nas_config
        rng = np.random.default_rng(cfg.seed)
        y_train = y_train.ravel().astype(np.float32)
        y_val = y_val.ravel().astype(np.float64)
        input_dim = x_train.shape[1]

        sample = lambda: _sample_config(
            rng, input_dim, cfg.width_choices, cfg.min_layers, cfg.max_layers
        )
        configs = [sample() for _ in range(cfg.n_configs)]
        for i, c in enumerate(configs):
            c["_seed"] = cfg.seed + i

        budgets = []
        b = cfg.min_budget_epochs
        while b <= cfg.max_budget_epochs:
            budgets.append(b)
            b = int(b * cfg.eta)

        deadline = time.time() + cfg.max_runtime
        survivors = configs
        round_idx = 0

        while time.time() < deadline:
            if not survivors:
                survivors = [sample() for _ in range(cfg.n_configs // 2)]
                for i, c in enumerate(survivors):
                    c["_seed"] = cfg.seed + round_idx * 1000 + i
                round_idx += 1

            for stage, budget in enumerate(budgets):
                if time.time() >= deadline or not survivors:
                    break

                results: list[TrialResult] = []
                for c in survivors:
                    if time.time() >= deadline:
                        break
                    results.append(
                        _train_one(c, x_train, y_train, x_val, y_val, budget, self.device)
                    )

                if not results:
                    break

                results.sort(key=lambda r: r.val_loss)
                keep = max(1, len(results) // cfg.eta)
                survivors = [r.config for r in results[:keep]]
                self.search_history.append({
                    "round": round_idx,
                    "stage": stage,
                    "budget_epochs": budget,
                    "n_trials": len(results),
                    "best_val_loss": results[0].val_loss,
                    "best_mae": results[0].metrics.get("mae"),
                })

                if results[0].val_loss < (self.best_trial.val_loss if self.best_trial else float("inf")):
                    self.best_trial = results[0]

            survivors = []  # start fresh round after full bracket

        # Final refit at max budget on best config
        if self.best_trial and time.time() < deadline:
            final_epochs = min(cfg.max_budget_epochs, self.best_trial.budget_epochs * cfg.eta)
            self.best_trial = _train_one(
                self.best_trial.config,
                x_train,
                y_train,
                x_val,
                y_val,
                final_epochs,
                self.device,
            )

        if self.best_trial is None:
            raise RuntimeError("Hyperband NAS found no valid model within time budget.")

        self._model = _DynamicMLP(
            self.best_trial.config["input_dim"],
            self.best_trial.config["widths"],
            self.best_trial.config["dropout"],
        )
        self._model.load_state_dict(self.best_trial.state_dict)
        self._model.to(self.device).eval()
        return self.best_trial

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.best_trial is None:
            raise RuntimeError("Model not fitted")
        x_t = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self._model(x_t).cpu().numpy()
