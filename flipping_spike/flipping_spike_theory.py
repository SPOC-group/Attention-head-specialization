import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import optimize

def sym(A):
    return 0.5 * (A + A.T)

def spd_sqrt(A, eps=1e-8):
    A = sym(A)
    w, U = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    return (U * np.sqrt(w)) @ U.T

def spd_inv(A, eps=1e-12):
    A = sym(A)
    w, U = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    return (U * (1.0 / w)) @ U.T

def unpack_x(x, L, H):
    return x.reshape(L, H)   

def pack_Z(Ztok):
    return Ztok.reshape(-1)

def result_name(lam, nu1, nu2, L, H, phase):
    return f"flipping_spike_theory_lambda_{float(lam)}_nu1_{float(nu1)}_nu2_{float(nu2)}_L{L}_H{H}_{phase.upper()}"

def softmax(Z):
    m = np.max(Z)
    e = np.exp(Z-m)
    return e/np.sum(e)

sigma = softmax

def psi(Z, omega, V_inv, L, H):
    Ztok = unpack_x(Z, L, H)     # (L,H)
    Zhead = Ztok.T               # (H,L)

    a = np.zeros(L)
    for h in range(H):
        a += (1/H)*sigma(Zhead[h])
    delta = np.zeros(L)
    delta[0] = 1.0
    loss = -0.5*np.sum((delta - a)**2)  
    quad = -0.5*np.einsum("lh,hi,li->", Ztok - omega, V_inv, Ztok - omega) 
    return -(loss + quad)

def gradPsi(Z, omega, V_inv, L, H):
    Ztok = unpack_x(Z, L, H)     # (L,H)
    Zhead = Ztok.T               # (H,L)

    a = np.zeros(L)
    for h in range(H):
        a += (1/H)*sigma(Zhead[h])
    delta = np.zeros(L)
    delta[0] = 1.0

    gradLoss = np.zeros((H, L))
    for h in range(H):
        s = sigma(Zhead[h])
        gradLoss[h] = (1/H)*s*((delta-a) - s @ (delta-a))
    gradQuad = - (Ztok - omega) @ V_inv
    return -pack_Z(gradLoss.T + gradQuad)

@dataclass
class StateEvolution:
    alpha: float
    H: int
    L: int
    lam: float = 1.0
    nu1: float = 1.0                
    nu2: float = 1.0                
    seed: int = 0
    informed_init: bool = False

    n_mc: int = 1000
    n_mc_end: int = 5000
    mc_switch_iter: int = 50
    max_iter: int = 100
    tol: float = 1e-4
    damping: float = 0.5
    max_error: float = 1e3
    phase: str = "free"

    def __post_init__(self):
        if self.phase not in ("free", "ps", "mem", "spec"):
            raise ValueError("phase must be free, ps, mem or spec")
        if self.phase != "free" and self.H != 2:
            raise ValueError("The ps, mem and spec branches require H=2")
        if self.phase == "spec":
            self.informed_init = True
        if self.lam <= 0 or self.alpha < 0 or min(self.nu1, self.nu2) < 0:
            raise ValueError("Require lambda>0 and alpha, nu1, nu2>=0")
        if min(self.H, self.L, self.n_mc, self.max_iter) < 1 or not 0 <= self.damping < 1 or self.tol <= 0:
            raise ValueError("Invalid dimensions, iteration parameters or damping")
        if not self.max_error > self.tol:
            raise ValueError("max_error must exceed tol")
        self.rng = np.random.default_rng(self.seed)
        self.F = 2
        self.theta_plus = np.array([np.sqrt(self.nu1),  np.sqrt(self.nu2)])
        self.theta_minus = np.array([np.sqrt(self.nu1), -np.sqrt(self.nu2)])

        self.z_samples = self.rng.normal(size=(self.L, self.H, self.n_mc))

        self.Zs_plus_log = np.zeros((self.L * self.H, self.n_mc))
        self.Zs_minus_log = np.zeros((self.L * self.H, self.n_mc))
    
    def reset_mc_buffers(self, new_n_mc):
        old_n_mc = self.n_mc
        old_plus = self.Zs_plus_log.copy()
        old_minus = self.Zs_minus_log.copy()

        self.n_mc = new_n_mc
        self.z_samples = self.rng.normal(size=(self.L, self.H, self.n_mc))

        self.Zs_plus_log = np.zeros((self.L * self.H, self.n_mc))
        self.Zs_minus_log = np.zeros((self.L * self.H, self.n_mc))

        n_keep = min(old_n_mc, self.n_mc)
        self.Zs_plus_log[:, :n_keep] = old_plus[:, :n_keep]
        self.Zs_minus_log[:, :n_keep] = old_minus[:, :n_keep]

    
    def project_overlaps(self, m, q, V):
        if self.phase == "free":
            return m, sym(q), sym(V)
        m = np.array([[np.mean(m[:, 0]), 0.5 * (m[0, 1] - m[1, 1])],
                      [np.mean(m[:, 0]), 0.5 * (m[1, 1] - m[0, 1])]])
        q, V = 0.5 * (q + q[::-1, ::-1]), 0.5 * (V + V[::-1, ::-1])
        if self.phase in ("ps", "mem"):
            m[:, 1] = 0.0
        if self.phase == "ps":
            q = np.full_like(q, np.mean(q))
        return m, sym(q), sym(V)

    def init_state(self):
        m = np.zeros((self.H, self.F))
        a, b = (1.0, 0.5) if self.informed_init else (1e-2, 1e-2)
        m[:, 0] = a
        if self.H == 1:
            u = np.zeros(1)
        else:
            u = (-1.0) ** np.arange(self.H)
            u -= np.mean(u)
            u /= np.max(np.abs(u))
        m[:, 1] = b * u

        I = np.eye(self.H)
        one = np.ones((self.H, self.H))
        P0 = one / self.H
        Pp = I - P0
        q = one + 1e-8 * Pp
        V = 2.0 * P0 + 4.0 * Pp
        if self.phase == "mem":
            q += 1e-2 * Pp
        if self.phase == "spec":
            q += self.H * b**2 * Pp
        return self.project_overlaps(m, q, V)

    def omegas(self, m, q):
        noise = np.einsum("hi,lin->lhn", spd_sqrt(q), self.z_samples)
        omegas_plus = np.copy(noise)
        omegas_minus = np.copy(noise)
        omegas_plus[0] += (m @ self.theta_plus)[:, None]
        omegas_minus[0] += (m @ self.theta_minus)[:, None]
        return omegas_plus, omegas_minus
    
    def proxis(self, omega_plus, omega_minus, V):
        Zs_plus = np.zeros((self.L, self.H, self.n_mc))
        Zs_minus = np.zeros((self.L, self.H, self.n_mc))

        V_inv = spd_inv(V)
        for n in range(self.n_mc):
            Z_plus = optimize.fmin_l_bfgs_b(psi, self.Zs_plus_log[:, n], args=(omega_plus[:, :, n], V_inv, self.L, self.H),
                                           fprime=gradPsi, factr=0.0, pgtol=self.tol**2, maxls=50)[0]
            Z_minus = optimize.fmin_l_bfgs_b(psi, self.Zs_minus_log[:, n], args=(omega_minus[:, :, n], V_inv, self.L, self.H),
                                           fprime=gradPsi, factr=0.0, pgtol=self.tol**2, maxls=50)[0]
            self.Zs_plus_log[:, n] = Z_plus
            self.Zs_minus_log[:, n] = Z_minus
            Zs_plus[:, :, n] = unpack_x(Z_plus, self.L, self.H)
            Zs_minus[:, :, n] = unpack_x(Z_minus, self.L, self.H)
        return Zs_plus, Zs_minus

    def update_hat_overlaps(self, m, q, V, omegas_plus, omegas_minus, Zs_plus, Zs_minus):
        V_inv = spd_inv(V)
        Rs_plus = np.einsum("hi,lin->lhn", V_inv, Zs_plus - omegas_plus)
        Rs_minus = np.einsum("hi,lin->lhn", V_inv, Zs_minus - omegas_minus)

        m_hat = self.alpha/2 * (np.outer(np.mean(Rs_plus[0], axis=-1), self.theta_plus)
                              + np.outer(np.mean(Rs_minus[0], axis=-1), self.theta_minus))
        q_hat = self.alpha/(2*self.n_mc) * (np.einsum("lhn,lin->hi", Rs_plus, Rs_plus)
                                         + np.einsum("lhn,lin->hi", Rs_minus, Rs_minus))
        V_hat = -self.alpha/(2*self.n_mc) * np.einsum(
            "lhn,lin->hi", Rs_plus + Rs_minus, self.z_samples) @ spd_inv(spd_sqrt(q))
        return self.project_overlaps(m_hat, q_hat, V_hat)

    def update_overlaps(self, m_hat, q_hat, V_hat):
        IdV_inv = spd_inv(self.lam*np.eye(self.H) + V_hat)
        m = IdV_inv @ m_hat
        q = IdV_inv @ (m_hat@m_hat.T + q_hat) @ IdV_inv
        V = IdV_inv
        return self.project_overlaps(m, q, V)

    def update_state(self, m, q, V):
        omegas_plus, omegas_minus = self.omegas(m, q)
        Zs_plus, Zs_minus = self.proxis(omegas_plus, omegas_minus, V)
        m_hat, q_hat, V_hat = self.update_hat_overlaps(m, q, V, omegas_plus, omegas_minus, Zs_plus, Zs_minus)
        return self.update_overlaps(m_hat, q_hat, V_hat)

    def convergence_error(self, old, new):
        errors = [np.linalg.norm(b - a) / max(1.0, np.linalg.norm(a)) for a, b in zip(old, new)]
        return max(errors) if np.all(np.isfinite(errors)) else np.inf

    def training_loss(self, m, q, V):
        omegas_plus, omegas_minus = self.omegas(m, q)
        Zs_plus, Zs_minus = self.proxis(omegas_plus, omegas_minus, V)
        m_hat, q_hat, V_hat = self.update_hat_overlaps(m, q, V, omegas_plus, omegas_minus, Zs_plus, Zs_minus)
        m_new, q_new, V_new = self.update_overlaps(m_hat, q_hat, V_hat)
        err = self.convergence_error((m, q, V), (m_new, q_new, V_new))

        V_inv = spd_inv(V)
        losses, moreaus = np.zeros(self.n_mc), np.zeros(self.n_mc)
        for n in range(self.n_mc):
            for Zs, omegas in ((Zs_plus, omegas_plus), (Zs_minus, omegas_minus)):
                Z = Zs[:, :, n]
                losses[n] += 0.5 * psi(pack_Z(Z), Z, V_inv, self.L, self.H)
                moreaus[n] += 0.5 * psi(pack_Z(Z), omegas[:, :, n], V_inv, self.L, self.H)
        train_loss = self.alpha * np.mean(losses)
        free_entropy = 0.5 * np.trace(V_hat @ q - q_hat @ V) - np.sum(m_hat * m)
        free_entropy += 0.5 * np.trace(V_new @ (q_hat + m_hat @ m_hat.T)) - self.alpha * np.mean(moreaus)
        return train_loss, free_entropy, err

    def generalization_error(self, m, q, n_mc_err=10000):
        z_old = self.z_samples
        self.z_samples = self.rng.normal(0.0, 1.0, size=(self.L, self.H, n_mc_err))
        omegas_plus, omegas_minus = self.omegas(m, q)

        delta = np.zeros(self.L)
        delta[0] = 1.0
        gen_errs = np.zeros(n_mc_err)
        for n in range(n_mc_err):
            a_plus = np.mean([sigma(omegas_plus[:, h, n]) for h in range(self.H)], axis=0)
            a_minus = np.mean([sigma(omegas_minus[:, h, n]) for h in range(self.H)], axis=0)
            loss_plus = 0.5 * np.sum((delta - a_plus) ** 2)
            loss_minus = 0.5 * np.sum((delta - a_minus) ** 2)
            gen_errs[n] = 0.5 * (loss_plus + loss_minus)
        self.z_samples = z_old
        return np.mean(gen_errs)
    
    def run(self):
        self.diverged = False
        m, q, V = self.init_state()
        # Keep the initial evaluation to preserve the Monte Carlo draw order.
        gen_err = self.generalization_error(m, q)

        for t in range(self.max_iter):
            if t == self.mc_switch_iter and self.n_mc_end > self.n_mc:
                print(f"Switching Monte Carlo samples: {self.n_mc} -> {self.n_mc_end}")
                self.reset_mc_buffers(self.n_mc_end)

            m_old, q_old, V_old = m.copy(), q.copy(), V.copy()
            try:
                m_new, q_new, V_new = self.update_state(m, q, V)
                err = self.convergence_error((m, q, V), (m_new, q_new, V_new))
            except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
                print(f"Numerical failure: {exc}")
                err = np.inf
            if not np.isfinite(err) or err > self.max_error:
                self.diverged = True
                print(f"Stopping alpha={self.alpha}, iteration={t}: convergence error={err}, max_error={self.max_error}")
                break
            m = self.damping * m_old + (1.0 - self.damping) * m_new
            q = self.damping * q_old + (1.0 - self.damping) * q_new
            V = self.damping * V_old + (1.0 - self.damping) * V_new
            m, q, V = self.project_overlaps(m, q, V)

            print("iteration =", t)
            print("m =", m.flatten())
            print("q =", q.flatten())
            print("V =", V.flatten())
            print("convergence error =", err)
            if err < self.tol:
                break

        if not self.diverged:
            try:
                gen_err = self.generalization_error(m, q)
                self.train_loss, self.free_entropy, err = self.training_loss(m, q, V)
            except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
                print(f"Numerical failure during final evaluation: {exc}")
                err = np.inf
            self.diverged = bool(not np.isfinite(err) or err > self.max_error)
            if self.diverged:
                print(f"Stopping during final evaluation: convergence error={err}, max_error={self.max_error}")
        if self.diverged:
            gen_err = self.train_loss = self.free_entropy = np.nan
        self.converged = bool(not self.diverged and np.isfinite(err) and err < self.tol)
        return m, q, V, gen_err, err, t

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("alpha", type=float)
    parser.add_argument("nu1", type=float)
    parser.add_argument("nu2", type=float)
    parser.add_argument("L", type=int)
    parser.add_argument("H", type=int)
    parser.add_argument("lam", type=float)
    parser.add_argument("initI", type=int)
    parser.add_argument("output_dir")
    parser.add_argument("seed", type=int)
    parser.add_argument("--phase", choices=("free", "ps", "mem", "spec"), default="free")
    parser.add_argument("--n-mc", type=int, default=5000)
    parser.add_argument("--n-mc-end", type=int, default=5000)
    parser.add_argument("--mc-switch-iter", type=int, default=400)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--damping", type=float, default=0.7)
    parser.add_argument("--max-error", type=float, default=1e3)
    args = parser.parse_args()

    alpha, nu_1, nu_2, L, H, lambda_ = args.alpha, args.nu1, args.nu2, args.L, args.H, args.lam
    name = result_name(lambda_, nu_1, nu_2, L, H, args.phase)
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    se = StateEvolution(
        alpha=alpha, H=H, L=L, lam=lambda_, nu1=nu_1, nu2=nu_2,
        seed=args.seed, informed_init=bool(args.initI),
        n_mc=args.n_mc, n_mc_end=args.n_mc_end,
        mc_switch_iter=args.mc_switch_iter, max_iter=args.max_iter, tol=args.tol,
        damping=args.damping, max_error=args.max_error, phase=args.phase
    )
    print("maximum convergence error =", se.max_error)
    m, q, V, gen_err, err, t = se.run()

    print("\nFinal state:")
    print("m =", m.flatten())
    print("q =", q.flatten())
    print("generalization error =", gen_err)
    print("training error =", se.train_loss / alpha)
    print("convergence error =", err)
    print("iterations =", t)

    if se.diverged:
        raise SystemExit("SE diverged; no result was saved.")

    filename = output_dir / f"{name}_alpha_{alpha:.2f}.csv"
    data = {
        "alpha": alpha,
        "m": [m.flatten().tolist()],
        "q": [q.flatten().tolist()],
        "train_error": se.train_loss / alpha,
        "test_error": gen_err,
        "free_entropy": se.free_entropy,
    }
    pd.DataFrame(data).to_csv(filename, index=False)
