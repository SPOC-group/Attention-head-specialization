import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
import pandas as pd

"""
Multi-head attention on single-location flipping spike model.
"""

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
        y_hat = torch.einsum('nlh,nld->nd', sigma(Z), X) / self.H
        return y_hat
    
    def loss(self, y_hat, y):
        return 0.5 * torch.sum((y - y_hat) ** 2) / self.D+ self.lambda_ * 0.5 * torch.sum(self.w ** 2) 

class Run:
    def __init__(self, D, alpha, nu_1, nu_2, L, H, lambda_, initI, tMax=200):
        self.D = D
        self.alpha = alpha
        self.N = int(alpha * D)
        self.nu_1 = nu_1
        self.nu_2 = nu_2
        self.L = L
        self.H = H
        self.lambda_ = lambda_
        self.initI = initI
        self.tMax = tMax
        
        self.log_data = []
        
        self._generate_data()
        w_informed = torch.randn(self.D, self.H) * 0.01
        signs = torch.ones(self.H)
        signs[1::2] = -1
        for h in range(self.H):
            w_informed[:, h] += 0.1 * (np.sqrt(self.nu_1) * self.k_star[:, 0]+ signs[h] * np.sqrt(self.nu_2) * self.k_star[:, 1])
        self.model = ERM(self.D, self.N, self.H, self.lambda_, self.initI, w_init=w_informed)
        if not initI:
            with torch.no_grad():
                w_0 = torch.randn(self.D, self.H)
                k2 = self.k_star[:, 1:2]
                w_0 = w_0 - k2 @ ((k2.T @ w_0) / (k2.T @ k2))
                self.model.w.copy_(w_0)
            
        self.optimiseur = optim.LBFGS(self.model.parameters(), lr=0.5, max_iter=20, history_size=10)

    def _generate_data(self):
        self.k_star = torch.randn(self.D, 2)
        self.X = torch.randn((self.N, self.L, self.D))
        theta_1 = torch.ones((self.N,)) * np.sqrt(self.nu_1)
        signs = torch.ones(self.N)
        signs[:self.N // 2] = -1
        signs = signs[torch.randperm(self.N)] 
        theta_2 = signs * np.sqrt(self.nu_2)
        self.X[:,0,:] += (torch.unsqueeze(theta_1, 1)*self.k_star[:,0] + torch.unsqueeze(theta_2, 1)*self.k_star[:,1])/np.sqrt(self.D)
        self.X_test = torch.randn((1000, self.L, self.D))
        theta_1 = torch.ones((1000,)) * np.sqrt(self.nu_1)
        signs = torch.ones(1000)
        signs[:1000 // 2] = -1
        signs = signs[torch.randperm(1000)] 
        theta_2 = signs * np.sqrt(self.nu_2)
        self.X_test[:,0,:] += (torch.unsqueeze(theta_1, 1)*self.k_star[:,0] + torch.unsqueeze(theta_2, 1)*self.k_star[:,1])/np.sqrt(self.D)
        self.y = self.X[:,0,:]
        self.y_test = self.X_test[:,0,:]

    def risk(self, y_hat, y):
        with torch.no_grad():
            return torch.mean(0.5*(y - y_hat) ** 2).item()

    def run(self):
        losses = []
        for n in range(self.tMax):
            def closure():
                self.optimiseur.zero_grad()
                y_hat = self.model(self.X)
                loss = self.model.loss(y_hat, self.y)
                loss.backward()
                return loss
            self.optimiseur.step(closure)
            with torch.no_grad():
                y_hat = self.model(self.X)
                losses.append(self.model.loss(y_hat, self.y).item())
            if n > 10:
                diff = np.std(losses[-10:]) / np.abs(np.mean(losses[-10:]))
                if diff < 1e-10:
                    print(f"Convergence reached at iteration {n}.")
                    break

        self._log_final_metrics()
    
    def _log_final_metrics(self):
        with torch.no_grad():
            y_hat = self.model(self.X)
            err = self.risk(y_hat, self.y)
            y_hat_test = self.model(self.X_test)
            err_test = self.risk(y_hat_test, self.y_test)
            m = self.model.w.T @ self.k_star/self.D
            q = self.model.w.T @ self.model.w/self.D

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
    L = int(sys.argv[5])
    H = int(sys.argv[6])
    lambda_ = float(sys.argv[7])
    initI = int(sys.argv[8])
    output_dir = sys.argv[9] 
    seed = int(sys.argv[10])

    torch.manual_seed(seed) 
    
    experiment = Run(D, alpha, nu_1, nu_2, L, H, lambda_, initI)
    experiment.run()
    
    name = f"flipping_spike_empirics_lambda_{lambda_}_nu1_{nu_1}_nu2_{nu_2}_L{L}_H{H}_D{D}_initI{initI}"
    output_dir = Path(output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{name}_alpha_{alpha:.2f}_seed_{seed}.csv"
    experiment.save_log(filename)
