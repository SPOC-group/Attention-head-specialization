import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
import pandas as pd

"""Multi-head attention on the non-isotropic Gaussian single-location model."""


sigma = lambda Z: torch.softmax(Z, dim=1)


class ERM(nn.Module):
    def __init__(self, D, N, H, lambda_, initI, w_init=None):
        super().__init__()
        self.D = D
        self.N = N
        self.H = H
        self.lambda_ = lambda_
        self.w = nn.Parameter(w_init.clone() if initI else torch.randn(D, H))

    def forward(self, X):
        Z = X @ self.w / np.sqrt(self.D)
        return torch.einsum("nlh,nld->nd", sigma(Z), X) / self.H

    def loss(self, y_hat, y):
        return 0.5 * torch.sum((y - y_hat) ** 2) / self.D + self.lambda_ * 0.5 * torch.sum(self.w**2)


class Run:
    def __init__(self, D, alpha, nu_1, nu_2, F, L, H, lambda_, initI, tMax=200):
        self.D = D
        self.alpha = alpha
        self.N = int(alpha * D)
        self.nu_1 = nu_1
        self.nu_2 = nu_2
        self.F = F
        self.L = L
        self.H = H
        self.lambda_ = lambda_
        self.initI = initI
        self.tMax = tMax
        self.log_data = []

        if self.F < 1 or self.H < 1 or self.L < 1 or self.N < 1:
            raise ValueError("F, H, L and N must be positive")
        if self.lambda_ <= 0 or self.nu_2 < 0 or self.nu_1 < self.nu_2:
            raise ValueError("Require lambda>0 and nu_1>=nu_2>=0")
        self.nu = torch.tensor([self.nu_1]) if self.F == 1 else torch.linspace(self.nu_1, self.nu_2, self.F)

        self._generate_data()
        w_informed = self._informed_initialization()
        self.model = ERM(self.D, self.N, self.H, self.lambda_, self.initI, w_init=w_informed)
        if not self.initI:
            with torch.no_grad():
                w_0 = torch.randn(self.D, self.H)
                gram = self.k_star.T @ self.k_star
                w_0 -= self.k_star @ (torch.linalg.pinv(gram) @ (self.k_star.T @ w_0))
                self.model.w.copy_(w_0)

        self.optimiseur = optim.LBFGS(self.model.parameters(), lr=0.5, max_iter=20, history_size=10)

    def _sample_theta(self, n_samples):
        return torch.randn(n_samples, self.F) * torch.sqrt(self.nu).unsqueeze(0)

    def _sample_dataset(self, n_samples):
        X = torch.randn(n_samples, self.L, self.D)
        theta = self._sample_theta(n_samples)
        X[:, 0, :] += theta @ self.k_star.T / np.sqrt(self.D)
        return X, X[:, 0, :]

    def _generate_data(self):
        self.k_star = torch.randn(self.D, self.F)
        self.X, self.y = self._sample_dataset(self.N)
        self.X_test, self.y_test = self._sample_dataset(1000)

    def _informed_initialization(self):
        w_init = 0.01 * torch.randn(self.D, self.H)
        for f in range(min(self.F, self.H // 2)):
            amplitude = 0.1 * torch.sqrt(self.nu[f])
            w_init[:, 2 * f] += amplitude * self.k_star[:, f]
            w_init[:, 2 * f + 1] -= amplitude * self.k_star[:, f]
        return w_init

    def risk(self, y_hat, y):
        with torch.no_grad():
            return torch.mean(0.5 * (y - y_hat) ** 2).item()

    def run(self):
        losses = []
        for n in range(self.tMax):
            def closure():
                self.optimiseur.zero_grad()
                loss = self.model.loss(self.model(self.X), self.y)
                loss.backward()
                return loss

            self.optimiseur.step(closure)
            with torch.no_grad():
                losses.append(self.model.loss(self.model(self.X), self.y).item())
            if n > 10:
                diff = np.std(losses[-10:]) / np.abs(np.mean(losses[-10:]))
                if diff < 1e-10:
                    print(f"Convergence reached at iteration {n}.")
                    break
        self._log_final_metrics()

    def _log_final_metrics(self):
        with torch.no_grad():
            err = self.risk(self.model(self.X), self.y)
            err_test = self.risk(self.model(self.X_test), self.y_test)
            m = self.model.w.T @ self.k_star / self.D
            q = self.model.w.T @ self.model.w / self.D

        self.log_data.append({
            "alpha": self.alpha,
            "m": m.flatten().tolist(),
            "q": q.flatten().tolist(),
            "train_error": err,
            "test_error": err_test,
        })
        print(f"Final test error: {err_test}")

    def save_log(self, filename):
        pd.DataFrame(self.log_data).to_csv(filename, index=False)


if __name__ == "__main__":
    D = int(sys.argv[1])
    alpha = float(sys.argv[2])
    nu_1 = float(sys.argv[3])
    nu_2 = float(sys.argv[4])
    F = int(sys.argv[5])
    L = int(sys.argv[6])
    H = int(sys.argv[7])
    lambda_ = float(sys.argv[8])
    initI = int(sys.argv[9])
    output_dir = sys.argv[10]
    seed = int(sys.argv[11])

    torch.manual_seed(seed)
    np.random.seed(seed)

    experiment = Run(D, alpha, nu_1, nu_2, F, L, H, lambda_, initI)
    experiment.run()

    name = (
        f"non_isotropic_gaussian_empirics_lambda_{lambda_}_nu1_{nu_1}_nu2_{nu_2}_"
        f"F{F}_L{L}_H{H}_D{D}_initI{initI}"
    )
    output_dir = Path(output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment.save_log(output_dir / f"{name}_alpha_{alpha:.2f}_seed_{seed}.csv")
