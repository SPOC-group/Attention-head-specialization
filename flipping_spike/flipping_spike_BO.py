import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq


@dataclass
class StateEvolution:
    alpha: float
    nu1: float
    nu2: float
    L: int
    seed: int
    n_mc: int = 200000
    batch_size: int = 20000
    n_quad: int = 100

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        z, w = np.polynomial.hermite.hermgauss(self.n_quad)
        self.z, self.w = np.sqrt(2) * z, w / np.sqrt(np.pi)

    def eta(self, q):
        t = self.nu2 * q
        return np.sum(self.w * np.tanh(t + np.sqrt(t) * self.z)**2)

    def equation(self, q):
        if q == 0:
            return self.alpha * self.nu2**2 - 1
        return self.alpha * self.nu2 * (1 - q) * self.eta(q) / q - 1

    def solve_overlaps(self):
        q1 = self.alpha * self.nu1 / (1 + self.alpha * self.nu1)
        q2 = 0.0
        if self.alpha * self.nu2**2 > 1:
            q2 = brentq(self.equation, 0, 1)
        return np.array([q1, q2])

    def generalization_error(self, q):
        t = np.array([self.nu1, self.nu2]) * q
        losses = np.empty(self.n_mc)
        for start in range(0, self.n_mc, self.batch_size):
            n = min(self.batch_size, self.n_mc - start)
            h = self.rng.normal(size=(n, self.L, 2)) * np.sqrt(t)
            h[:, 0] += t
            w = np.exp(h[:, :, 0]) * np.cosh(h[:, :, 1])
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
    parser.add_argument("nu1", type=float)
    parser.add_argument("nu2", type=float)
    parser.add_argument("L", type=int)
    parser.add_argument("output_dir")
    parser.add_argument("seed", type=int)
    parser.add_argument("--n-mc", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--n-quad", type=int, default=100)
    args = parser.parse_args()

    se = StateEvolution(args.alpha, args.nu1, args.nu2, args.L, args.seed,
                        args.n_mc, args.batch_size, args.n_quad)
    q, error, error_se = se.run()
    name = f"flipping_spike_BO_nu1_{args.nu1}_nu2_{args.nu2}_L{args.L}"
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "alpha": args.alpha,
        "test_error": error,
        "rho1": q[0],
        "rho2": q[1],
    }
    filename = output_dir / f"{name}_alpha_{args.alpha:.2f}.csv"
    pd.DataFrame([data]).to_csv(filename, index=False)
    print("q =", q, "error =", error, "MC error =", error_se)
    print("Saved:", filename)
