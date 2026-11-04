import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import optimize, linalg


def sym(A):
    return 0.5 * (A + A.T)


def spd_sqrt(A, eps=1e-12):
    w, U = np.linalg.eigh(sym(A))
    if w.min() < -eps * max(1.0, np.max(np.abs(w))):
        raise np.linalg.LinAlgError("Non-positive-semidefinite covariance")
    return (U * np.sqrt(np.maximum(w, 0.0))) @ U.T


def spd_inv(A):
    A = sym(A)
    return sym(linalg.cho_solve(linalg.cho_factor(A), np.eye(A.shape[0])))


def unpack_x(x, L, H):
    return x.reshape(L, H)


def pack_Z(Ztok):
    return Ztok.reshape(-1)


def result_name(lam, nu1, nu2, F, L, H):
    return (
        f"non_isotropic_gaussian_theory_lambda_{float(lam)}_nu1_{float(nu1)}_"
        f"nu2_{float(nu2)}_F{F}_L{L}_H{H}_FREE"
    )


def softmax(Z):
    m = np.max(Z)
    e = np.exp(Z - m)
    return e / np.sum(e)


sigma = softmax


def psi(Z, omega, V_inv, L, H):
    Ztok = unpack_x(Z, L, H)
    Zhead = Ztok.T

    a = np.zeros(L)
    for h in range(H):
        a += sigma(Zhead[h]) / H
    delta = np.zeros(L)
    delta[0] = 1.0
    loss = -0.5 * np.sum((delta - a) ** 2)
    quad = -0.5 * np.einsum("lh,hi,li->", Ztok - omega, V_inv, Ztok - omega)
    return -(loss + quad)


def gradPsi(Z, omega, V_inv, L, H):
    Ztok = unpack_x(Z, L, H)
    Zhead = Ztok.T

    a = np.zeros(L)
    for h in range(H):
        a += sigma(Zhead[h]) / H
    delta = np.zeros(L)
    delta[0] = 1.0

    gradLoss = np.zeros((H, L))
    for h in range(H):
        s = sigma(Zhead[h])
        gradLoss[h] = s * ((delta - a) - s @ (delta - a)) / H
    gradQuad = -(Ztok - omega) @ V_inv
    return -pack_Z(gradLoss.T + gradQuad)


def loss_hessian(Z, L, H):
    """Full token-major Hessian of the loss, including cross-head blocks."""
    probs = np.array([sigma(z) for z in unpack_x(Z, L, H).T])
    residual = probs.mean(axis=0)
    residual[0] -= 1.0
    jac = np.array([np.diag(p) - np.outer(p, p) for p in probs])
    G = jac.transpose(1, 2, 0).reshape(L, L * H) / H
    B = G.T @ G
    for h, p in enumerate(probs):
        v = p * (residual - p @ residual)
        block = np.diag(v) - np.outer(v, p) - np.outer(p, v)
        idx = h + H * np.arange(L)
        B[np.ix_(idx, idx)] += block / H
    return sym(B)


def hessPsi(Z, omega, V_inv, L, H):
    return loss_hessian(Z, L, H) + np.kron(np.eye(L), V_inv)


@dataclass
class StateEvolution:
    alpha: float
    H: int
    F: int
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

    def __post_init__(self):
        if self.lam <= 0 or self.alpha < 0 or self.nu2 < 0 or self.nu1 < self.nu2:
            raise ValueError("Require lambda>0, alpha>=0 and nu1>=nu2>=0")
        if min(self.H, self.F, self.L, self.n_mc, self.max_iter) < 1:
            raise ValueError("H, F, L, n_mc and max_iter must be positive")
        if not 0 <= self.damping < 1 or self.tol <= 0 or self.max_error <= self.tol:
            raise ValueError("Invalid damping or convergence tolerances")

        self.rng = np.random.default_rng(self.seed)
        nu = np.array([self.nu1]) if self.F == 1 else np.linspace(self.nu1, self.nu2, self.F)
        self.Lambda = np.diag(nu)
        self.z_samples = self.rng.normal(size=(self.L, self.H, self.n_mc))
        self.Zs_log = np.zeros((self.L * self.H, self.n_mc))

    def reset_mc_buffers(self, new_n_mc):
        # Keep each existing draw together with its proximal warm start.
        extra = new_n_mc - self.n_mc
        if extra > 0:
            self.z_samples = np.concatenate((self.z_samples,
                self.rng.normal(size=(self.L, self.H, extra))), axis=2)
            self.Zs_log = np.pad(self.Zs_log, ((0, 0), (0, extra)))
            self.n_mc = new_n_mc

    def init_state(self):
        m = np.zeros((self.H, self.F))
        scale = 0.5 if self.informed_init else 1e-3
        nu = np.diag(self.Lambda)
        amplitudes = scale * np.sqrt(nu / max(np.max(nu), 1e-12))
        n_seeded = min(self.F, self.H // 2)  # Opposite head pairs, one per seeded feature.
        for f in range(n_seeded):
            m[2 * f, f] = amplitudes[f] / np.sqrt(2.0)
            m[2 * f + 1, f] = -amplitudes[f] / np.sqrt(2.0)

        q = 0.4 * np.eye(self.H) + m @ m.T
        V = 0.4 * np.eye(self.H)
        return m, sym(q), sym(V)

    def omegas(self, m, q):
        Sigma1 = sym(q + m @ self.Lambda @ m.T)
        omegas = np.einsum("hi,lin->lhn", spd_sqrt(q), self.z_samples)
        omegas[0] = np.einsum("hi,in->hn", spd_sqrt(Sigma1), self.z_samples[0])
        return omegas

    def proxis(self, omegas, V):
        Zs = np.zeros((self.L, self.H, self.n_mc))
        V_inv = spd_inv(V)
        precision = np.kron(np.eye(self.L), V_inv)
        self.J_mean = np.zeros((self.L, self.H, self.H))
        grad_tol = max(1e-9, self.tol ** 2)
        for n in range(self.n_mc):
            args = (omegas[:, :, n], V_inv, self.L, self.H)
            Z, _, info = optimize.fmin_l_bfgs_b(
                psi, self.Zs_log[:, n], args=args, fprime=gradPsi,
                factr=0.0, pgtol=grad_tol, maxls=50,
            )
            if np.max(np.abs(info["grad"])) > max(1e-7, 10 * grad_tol):
                result = optimize.minimize(
                    psi, Z, args=args, jac=gradPsi, hess=hessPsi,
                    method="Newton-CG", options={"xtol": grad_tol, "maxiter": 100},
                )
                Z = result.x
            if not np.all(np.isfinite(Z)) or np.max(np.abs(gradPsi(Z, *args))) > max(1e-7, 10 * grad_tol):
                raise FloatingPointError(f"Proximal solve failed for sample {n}")
            B = loss_hessian(Z, self.L, self.H)
            # J = Vblock^-1 C Vblock^-1 - Vblock^-1, evaluated without
            # subtracting two large precisions; C is the FULL Hessian inverse.
            factor = linalg.cho_factor(sym(precision + B))
            J = sym(-B + B @ linalg.cho_solve(factor, B))
            for ell in range(self.L):
                sl = slice(ell * self.H, (ell + 1) * self.H)
                self.J_mean[ell] += J[sl, sl] / self.n_mc
            self.Zs_log[:, n] = Z
            Zs[:, :, n] = unpack_x(Z, self.L, self.H)
        return Zs

    def update_hat_overlaps(self, m, q, V, omegas, Zs):
        # Gaussian integration by parts, with implicit proximal derivatives.
        Rs = np.einsum("hi,lin->lhn", spd_inv(V), Zs - omegas)
        m_hat = self.alpha * self.J_mean[0] @ m @ self.Lambda
        q_hat = self.alpha / self.n_mc * np.einsum("lhn,lin->hi", Rs, Rs)
        V_hat = -self.alpha * self.J_mean.sum(axis=0)
        return m_hat, sym(q_hat), sym(V_hat)

    def update_overlaps(self, m_hat, q_hat, V_hat):
        IdV_inv = spd_inv(self.lam * np.eye(self.H) + V_hat)
        m = IdV_inv @ m_hat
        q = IdV_inv @ (m_hat @ m_hat.T + q_hat) @ IdV_inv
        V = IdV_inv
        return m, sym(q), sym(V)

    def update_state(self, m, q, V):
        omegas = self.omegas(m, q)
        Zs = self.proxis(omegas, V)
        return self.update_overlaps(*self.update_hat_overlaps(m, q, V, omegas, Zs))

    def convergence_error(self, old, new):
        errors = [np.linalg.norm(b - a) / max(1.0, np.linalg.norm(a)) for a, b in zip(old, new)]
        return max(errors) if np.all(np.isfinite(errors)) else np.inf

    def training_loss(self, m, q, V):
        omegas = self.omegas(m, q)
        Zs = self.proxis(omegas, V)
        m_hat, q_hat, V_hat = self.update_hat_overlaps(m, q, V, omegas, Zs)
        m_new, q_new, V_new = self.update_overlaps(m_hat, q_hat, V_hat)
        err = self.convergence_error((m, q, V), (m_new, q_new, V_new))

        V_inv = spd_inv(V)
        losses = np.zeros(self.n_mc)
        for n in range(self.n_mc):
            Z = Zs[:, :, n]
            losses[n] = psi(pack_Z(Z), Z, V_inv, self.L, self.H)
        self.train_err = np.mean(losses)
        train_loss = self.alpha * self.train_err
        return train_loss, err

    def generalization_error(self, m, q, n_mc_err=10000):
        z_old = self.z_samples
        self.z_samples = self.rng.normal(size=(self.L, self.H, n_mc_err))
        omegas = self.omegas(m, q)

        delta = np.zeros(self.L)
        delta[0] = 1.0
        gen_errs = np.zeros(n_mc_err)
        for n in range(n_mc_err):
            a = np.mean([sigma(omegas[:, h, n]) for h in range(self.H)], axis=0)
            gen_errs[n] = 0.5 * np.sum((delta - a) ** 2)
        self.z_samples = z_old
        return np.mean(gen_errs)

    def run(self):
        self.diverged = False
        self.train_err = self.train_loss = np.nan
        m, q, V = self.init_state()
        # Keep this draw to preserve the original Monte Carlo sequence.
        gen_err = self.generalization_error(m, q)

        for t in range(self.max_iter):
            if t == self.mc_switch_iter and self.n_mc_end > self.n_mc:
                print(f"Switching Monte Carlo samples: {self.n_mc} -> {self.n_mc_end}")
                self.reset_mc_buffers(self.n_mc_end)

            old = (m.copy(), q.copy(), V.copy())
            try:
                new = self.update_state(m, q, V)
                err = self.convergence_error((m, q, V), new)
            except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
                print(f"Numerical failure: {exc}")
                err = np.inf
            if not np.isfinite(err) or err > self.max_error:
                self.diverged = True
                print(
                    f"Stopping alpha={self.alpha}, iteration={t}: "
                    f"convergence error={err}, max_error={self.max_error}"
                )
                break

            m = self.damping * old[0] + (1.0 - self.damping) * new[0]
            q = sym(self.damping * old[1] + (1.0 - self.damping) * new[1])
            V = sym(self.damping * old[2] + (1.0 - self.damping) * new[2])

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
                self.train_loss, err = self.training_loss(m, q, V)
            except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
                print(f"Numerical failure during final evaluation: {exc}")
                err = np.inf
            self.diverged = bool(not np.isfinite(err) or err > self.max_error)
        if self.diverged:
            gen_err = self.train_err = self.train_loss = np.nan
        self.converged = bool(not self.diverged and err < self.tol)

        return m, q, V, gen_err, err, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("alpha", type=float)
    parser.add_argument("nu1", type=float)
    parser.add_argument("nu2", type=float)
    parser.add_argument("F", type=int)
    parser.add_argument("L", type=int)
    parser.add_argument("H", type=int)
    parser.add_argument("lam", type=float)
    parser.add_argument("initI", type=int)
    parser.add_argument("output_dir")
    parser.add_argument("seed", type=int)
    parser.add_argument("--n-mc", type=int, default=10000)
    parser.add_argument("--n-mc-end", type=int, default=5000)
    parser.add_argument("--mc-switch-iter", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--damping", type=float, default=0.7)
    parser.add_argument("--max-error", type=float, default=1e3)
    args = parser.parse_args()

    name = result_name(args.lam, args.nu1, args.nu2, args.F, args.L, args.H)
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    se = StateEvolution(
        alpha=args.alpha,
        H=args.H,
        F=args.F,
        L=args.L,
        lam=args.lam,
        nu1=args.nu1,
        nu2=args.nu2,
        seed=args.seed,
        informed_init=bool(args.initI),
        n_mc=args.n_mc,
        n_mc_end=args.n_mc_end,
        mc_switch_iter=args.mc_switch_iter,
        max_iter=args.max_iter,
        tol=args.tol,
        damping=args.damping,
        max_error=args.max_error,
    )
    print("maximum convergence error =", se.max_error)
    m, q, V, gen_err, err, t = se.run()

    print("\nFinal state:")
    print("m =", m.flatten())
    print("q =", q.flatten())
    print("V =", V.flatten())
    print("generalization error =", gen_err)
    print("per-sample training error =", se.train_err)
    print("convergence error =", err)
    print("iterations =", t)

    data = {
        "alpha": args.alpha,
        "m": [m.flatten().tolist()],
        "q": [q.flatten().tolist()],
        "train_error": se.train_err,
        "test_error": gen_err,
    }
    pd.DataFrame(data).to_csv(output_dir / f"{name}_alpha_{args.alpha:.2f}.csv", index=False)

    if not se.converged:
        raise SystemExit("SE did not converge: inspect the convergence error in the log.")
