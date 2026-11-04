import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class StateEvolution:
    alpha: float
    nus: list
    L: int
    seed: int
    n_mc: int = 200000
    batch_size: int = 20000

    def __post_init__(self):
        self.nus = np.array(self.nus, dtype=float)
        self.rng = np.random.default_rng(self.seed)

    def solve_overlaps(self):
        q = np.zeros(len(self.nus))
        for f, nu in enumerate(self.nus):
            if self.alpha * nu**2 > 1:
                q[f] = (self.alpha * nu**2 - 1) / (nu * (1 + self.alpha * nu))
        return q

    def generalization_error(self, q):
        t = self.nus * q
        losses = np.empty(self.n_mc)
        for start in range(0, self.n_mc, self.batch_size):
            n = min(self.batch_size, self.n_mc - start)
            z = self.rng.normal(size=(n, self.L, len(q)))
            z[:, 0] *= np.sqrt(1 + t)
            w = np.exp(0.5 * np.sum(t / (1 + t) * z**2, axis=2))
            losses[start:start + n] = 0.5 * (1 - w[:, 0] / w.sum(axis=1))
        se = losses.std(ddof=1) / np.sqrt(self.n_mc) if self.n_mc > 1 else np.nan
        return losses.mean(), se

    def run(self):
        q = self.solve_overlaps()
        error, error_se = self.generalization_error(q)
        return q, error, error_se


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("alpha", type=float)
    parser.add_argument("L", type=int)
    parser.add_argument("output_dir")
    parser.add_argument("seed", type=int)
    parser.add_argument("--nus", type=float, nargs="+", required=True)
    parser.add_argument("--n-mc", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=20000)
    args = parser.parse_args()

    se = StateEvolution(args.alpha, args.nus, args.L, args.seed, args.n_mc, args.batch_size)
    q, error, error_se = se.run()
    strengths = "".join(f"_nu{f}_{nu}" for f, nu in enumerate(args.nus, 1))
    name = f"non_isotropic_gaussian_BO{strengths}_F{len(q)}_L{args.L}"
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {"alpha": args.alpha, "test_error": error}
    for f, overlap in enumerate(q, 1):
        data[f"rho{f}"] = overlap
    filename = output_dir / f"{name}_alpha_{args.alpha:.2f}.csv"
    pd.DataFrame([data]).to_csv(filename, index=False)
    print("q =", q, "error =", error, "MC error =", error_se)
    print("Saved:", filename)
