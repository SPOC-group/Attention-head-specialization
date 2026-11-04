#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtri, softmax
from scipy.stats import qmc


def sym(a):
    return (a + np.swapaxes(a, -1, -2)) * 0.5


class NumericalFailure(RuntimeError):
    pass


def psd_sqrt(a):
    values, vectors = np.linalg.eigh(sym(a))
    scale = max(1.0, float(np.max(np.abs(values))))
    if values[0] < -1e-9 * scale:
        raise NumericalFailure("A covariance matrix is not positive semidefinite.")
    values[values < 1e-13 * scale] = 0.0
    return (vectors * np.sqrt(values)) @ vectors.T


def spd_inv(a):
    a = sym(a)
    if not np.all(np.isfinite(a)):
        raise NumericalFailure("A required positive-definite matrix contains nonfinite values.")
    try:
        np.linalg.cholesky(a)
        return sym(np.linalg.solve(a, np.eye(a.shape[-1])))
    except np.linalg.LinAlgError as exc:
        raise NumericalFailure("A required positive-definite matrix is singular or indefinite.") from exc


def inverse_sqrt(a):
    values, vectors = np.linalg.eigh(sym(a))
    if values[0] <= 0:
        raise NumericalFailure("V must be positive definite.")
    return (vectors / np.sqrt(values)) @ vectors.T


def complement(columns):
    if columns.shape[1] == 0:
        return np.eye(columns.shape[0])
    u, s, _ = np.linalg.svd(columns, full_matrices=True)
    if s[-1] < 1e-7 * s[0]:
        raise NumericalFailure("The requested specialized branch lost magnetization rank.")
    return u[:, columns.shape[1]:]


def loss_derivatives(z):
    n, L, H = z.shape
    p = softmax(z, axis=1)
    residual = p.mean(axis=2)
    residual[:, 0] -= 1.0
    b = p * (residual[:, :, None] - np.sum(p * residual[:, :, None], axis=1, keepdims=True))
    gradient = b / H
    ph = p.transpose(0, 2, 1)
    jac = ph[..., :, None] * np.eye(L) - ph[..., :, None] * ph[..., None, :]
    hessian = np.einsum("nhla,nkat->nlhtk", jac, jac).reshape(n, L * H, L * H) / H**2
    for h in range(H):
        bh, sh = b[:, :, h], p[:, :, h]
        second = bh[:, :, None] * np.eye(L)
        second -= bh[:, :, None] * sh[:, None, :] + sh[:, :, None] * bh[:, None, :]
        hessian[:, h::H, h::H] += second / H
    return 0.5 * np.sum(residual**2, axis=1), gradient, sym(hessian)


def proximal_batch(omega, V_inv, start, tol, max_iter):
    n, L, H = omega.shape
    z = omega.copy() if start is None else start.copy()
    D = np.kron(np.eye(L), V_inv)
    for _ in range(max_iter):
        loss, grad_loss, hess_loss = loss_derivatives(z)
        diff = z - omega
        grad = (grad_loss + diff @ V_inv).reshape(n, -1)
        hess = sym(hess_loss + D)
        residuals = np.max(np.abs(grad), axis=1)
        if np.max(residuals) <= tol:
            break
        minimum = np.linalg.eigvalsh(hess)[:, 0]
        shifted = hess + np.maximum(1e-7 - minimum, 0.0)[:, None, None] * np.eye(L * H)
        step = np.linalg.solve(shifted, grad[..., None])[..., 0].reshape(n, L, H)
        directional = np.sum(grad * step.reshape(n, -1), axis=1)
        base = loss + 0.5 * np.einsum("nlh,hk,nlk->n", diff, V_inv, diff)
        scale = np.ones(n)
        accepted = residuals <= tol
        candidate = z.copy()
        for _ in range(35):
            trial = z - scale[:, None, None] * step
            probs = softmax(trial, axis=1).mean(axis=2)
            probs[:, 0] -= 1
            delta = trial - omega
            objective = 0.5 * np.sum(probs**2, axis=1)
            objective += 0.5 * np.einsum("nlh,hk,nlk->n", delta, V_inv, delta)
            good = objective <= base - 1e-4 * scale * directional + 5e-15
            take = good & ~accepted
            candidate[take] = trial[take]
            accepted |= good
            if np.all(accepted):
                break
            scale[~accepted] *= 0.5
        if not np.all(accepted):
            raise NumericalFailure("Proximal Newton line search failed.")
        z = candidate
    else:
        raise NumericalFailure("Proximal solve did not converge; increase --prox-max-iter.")
    try:
        np.linalg.cholesky(hess)
        covariance = sym(np.linalg.inv(hess))
    except np.linalg.LinAlgError as exc:
        raise NumericalFailure("The converged proximal Hessian is not positive definite.") from exc
    envelope = loss + 0.5 * np.einsum("nlh,hk,nlk->n", diff, V_inv, diff)
    return z, grad_loss, covariance, envelope, float(np.max(residuals))


@dataclass
class State:
    m: np.ndarray
    q: np.ndarray
    V: np.ndarray

    def copy(self):
        return State(self.m.copy(), self.q.copy(), self.V.copy())


class GaussianSE:
    def __init__(self, args):
        self.args = args
        self.H, self.L = args.H, args.L
        self.nu = np.asarray(args.nus, dtype=float)
        self.F = len(self.nu)
        self.identity = np.eye(self.H)
        self.parallel = np.ones((self.H, self.H)) / self.H
        self.perp = self.identity - self.parallel
        if args.sampling == "sobol":
            uniforms = qmc.Sobol(self.H * self.L, scramble=True, seed=args.seed).random_base2(int(math.log2(args.n_mc)))
            self.noise = ndtri(np.clip(uniforms, 1e-15, 1 - 1e-15)).reshape(args.n_mc, self.L, self.H)
        else:
            self.noise = np.random.default_rng(args.seed).normal(size=(args.n_mc, self.L, self.H))

    def permutation_projection(self, matrix):
        common = float(np.sum(matrix) / self.H)
        contrast = float((np.trace(matrix) - common) / (self.H - 1))
        return common * self.parallel + contrast * self.perp

    def enforce(self, state, r):
        state.m[:, r:] = 0.0
        state.q, state.V = sym(state.q), sym(state.V)
        if r == 0:
            state.q = float(np.mean(state.q)) * np.ones((self.H, self.H))
            state.V = self.permutation_projection(state.V)
        return state

    def initial_state(self, r, parent=None, paired=False):
        if parent is not None:
            state = parent["state"].copy()
            direction = psd_sqrt(state.V) @ parent["critical_vector"]
            direction /= np.linalg.norm(direction)
            state.m[:, r - 1] = self.args.init_scale * direction
            state.q += np.outer(state.m[:, r - 1], state.m[:, r - 1])
        else:
            m = np.zeros((self.H, self.F))
            if paired:
                for f in range(r):
                    m[2 * f, f] = self.args.init_scale / np.sqrt(2)
                    m[2 * f + 1, f] = -self.args.init_scale / np.sqrt(2)
            elif r:
                patterns = np.sign(np.cos(np.pi * (np.arange(self.H)[:, None] + 0.5)
                                          * np.arange(1, r + 1)[None, :] / self.H))
                patterns -= patterns.mean(axis=0, keepdims=True)
                orthogonal, _ = np.linalg.qr(patterns)
                m[:, :r] = self.args.init_scale * orthogonal[:, :r]
            q0 = m @ m.T + 0.02 * self.parallel
            state = State(m, q0, self.identity / max(self.args.lam, 0.1))
        return self.enforce(state, r)

    def moments(self, state, r, warm=None):
        m, q, V = state.m, state.q, state.V
        Vi = spd_inv(V)
        fields = self.noise @ psd_sqrt(q)
        signal_cov = (m[:, :r] * self.nu[:r]) @ m[:, :r].T
        fields[:, 0] = self.noise[:, 0] @ psd_sqrt(q + signal_cov)
        z_all = np.empty_like(fields)
        qhat_mean = np.zeros_like(V)
        curvature_sum = np.zeros_like(V)
        curvature_11 = np.zeros_like(V)
        C11 = np.zeros_like(V)
        envelope = prox_error = 0.0
        n = self.args.n_mc
        D = np.kron(np.eye(self.L), Vi)
        for lo in range(0, n, self.args.batch_size):
            hi = min(lo + self.args.batch_size, n)
            start = None if warm is None else warm[lo:hi]
            z, loss_grad, C, M, error = proximal_batch(
                fields[lo:hi], Vi, start, self.args.opt_tol, self.args.prox_max_iter)
            z_all[lo:hi] = z
            qhat_mean += np.einsum("nlh,nlk->hk", loss_grad, loss_grad) / n
            A = sym(D - (D @ C) @ D)
            curvature_11 += A[:, :self.H, :self.H].sum(axis=0) / n
            C11 += C[:, :self.H, :self.H].sum(axis=0) / n
            for ell in range(self.L):
                sl = slice(ell * self.H, (ell + 1) * self.H)
                curvature_sum += A[:, sl, sl].sum(axis=0) / n
            envelope += float(M.sum()) / n
            prox_error = max(prox_error, error)
        if r == 0:
            qhat_mean = float(np.mean(qhat_mean)) * np.ones_like(qhat_mean)
            curvature_11 = self.permutation_projection(curvature_11)
            curvature_sum = self.permutation_projection(curvature_sum)
            C11 = self.permutation_projection(C11)
        return {"qhat_mean": sym(qhat_mean), "A11": sym(curvature_11),
                "Asum": sym(curvature_sum), "C11": sym(C11), "M": envelope,
                "prox_error": prox_error, "warm": z_all}

    def solve(self, alpha, r, initial=None):
        state = self.initial_state(r) if initial is None else self.enforce(initial.copy(), r)
        warm = None
        error = math.inf
        for iteration in range(1, self.args.max_iter + 1):
            stats = self.moments(state, r, warm)
            warm = stats["warm"]
            m, q, V = state.m, state.q, state.V
            Vi = spd_inv(V)
            mhat = -alpha * stats["A11"] @ (m * self.nu)
            qhat = alpha * stats["qhat_mean"]
            target_precision = self.args.lam * self.identity + alpha * stats["Asum"]
            m_residual = m - V @ mhat
            error_m = 0.0 if r == 0 else max(
                np.linalg.norm(m_residual[:, f]) / max(np.linalg.norm(m[:, f]), 1e-12) for f in range(r))
            error_q = np.linalg.norm(q - m @ m.T - V @ qhat @ V) / max(1.0, np.linalg.norm(q))
            error_V = np.linalg.norm(Vi - target_precision) / max(1.0, np.linalg.norm(Vi))
            error = max(error_m, error_q, error_V)
            if error <= self.args.tol:
                if r and min(np.linalg.norm(m[:, f]) for f in range(r)) < self.args.min_magnetization:
                    raise NumericalFailure(f"The {r}-SPEC branch collapsed to a lower-rank solution.")
                return self.result(alpha, r, state, stats, error, iteration)
            if r and iteration > 50 and min(np.linalg.norm(m[:, f]) for f in range(r)) < self.args.min_magnetization:
                raise NumericalFailure(f"The {r}-SPEC branch collapsed; its onset may be above this alpha.")
            eta = 1.0 - self.args.damping
            while eta > 1e-8:
                mixed_precision = sym((1 - eta) * Vi + eta * target_precision)
                try:
                    Vnew = spd_inv(mixed_precision)
                    break
                except NumericalFailure:
                    eta *= 0.5
            else:
                raise NumericalFailure("No positive-definite SE precision step was possible.")
            mtarget = Vnew @ mhat
            qtarget = mtarget @ mtarget.T + Vnew @ qhat @ Vnew
            state = self.enforce(State((1 - eta) * m + eta * mtarget,
                                       (1 - eta) * q + eta * qtarget, Vnew), r)
            if self.args.verbose and iteration % 100 == 0:
                print(f"    r={r} alpha={alpha:.8g} SE it={iteration} residual={error:.3e}", flush=True)
        raise NumericalFailure(f"SE did not converge at r={r}, alpha={alpha:.10g}: residual={error:.3e}; increase --max-iter or --damping.")

    def result(self, alpha, r, state, stats, error, iterations):
        W = inverse_sqrt(state.V)
        J = sym(alpha * (W @ stats["C11"] @ W - self.identity))
        basis = complement(np.ones((self.H, 1))) if r == 0 else complement(W @ state.m[:, :r])
        values, vectors = np.linalg.eigh(sym(basis.T @ J @ basis))
        kappa = float(values[-1])
        growth = float(self.nu[r] * kappa)
        common_growth = float(self.nu[r] * np.sum(J) / self.H) if r == 0 else math.nan
        if r == 0 and common_growth >= 1.0:
            raise NumericalFailure("The common-head PS mode is unstable; this is not an isolated head-specialization boundary.")
        learned_error = 0.0
        if r:
            learned = W @ state.m[:, :r]
            learned_error = float(np.linalg.norm(J @ learned - learned / self.nu[:r]) / max(np.linalg.norm(learned), 1e-12))
        potential = alpha * stats["M"] + 0.5 * self.args.lam * np.trace(state.q)
        potential -= 0.5 * np.trace(spd_inv(state.V) @ (state.q - state.m @ state.m.T))
        critical_vector = basis @ vectors[:, -1]
        if r == 0:
            critical_vector = np.ones(self.H)
            critical_vector[self.H // 2:] = -1
            critical_vector -= critical_vector.mean()
            critical_vector /= np.linalg.norm(critical_vector)
        return {"alpha": float(alpha), "r": r, "state": state.copy(), "J": J,
                "kappa": kappa, "growth": growth, "g": growth - 1,
                "critical_vector": critical_vector, "se_error": float(error),
                "iterations": iterations, "prox_error": stats["prox_error"],
                "learned_eigenvector_error": learned_error, "common_growth": common_growth,
                "free_energy": float(potential)}


class BoundarySearch:
    def __init__(self, args):
        self.args = args
        self.se = GaussianSE(args)
        self.cache = {r: {} for r in range(args.max_features)}
        self.rows, self.trace = [], []
        config = {k: v for k, v in vars(args).items() if k not in ("output_dir", "verbose")}
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        delta_tag = "" if args.delta is None else f"_delta_{args.delta:.6g}"
        self.stem = (f"alpha_critical_gaussian_F{len(args.nus)}_L{args.L}_H{args.H}"
                     f"_lam_{args.lam:.6g}{delta_tag}_seed_{args.seed}_{digest}")
        self.directory = Path(args.output_dir).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)

    def evaluate(self, alpha, r, parent=None):
        key = float(alpha)
        if key in self.cache[r]:
            return self.cache[r][key]
        if self.cache[r]:
            nearest = min(self.cache[r], key=lambda a: abs(a - alpha))
            candidates = [self.cache[r][nearest]["state"]]
        else:
            candidates = [self.se.initial_state(r, parent)]
            if r and self.args.branch_starts >= 2:
                candidates.append(self.se.initial_state(r))
            if r and self.args.branch_starts >= 3:
                candidates.append(self.se.initial_state(r, paired=True))
        successes, errors = [], []
        for candidate in candidates:
            try:
                successes.append(self.se.solve(alpha, r, candidate))
            except NumericalFailure as exc:
                errors.append(str(exc))
        if not successes and r and self.args.seed_retries:
            print(f"  Retrying r={r}, alpha={alpha:.8g} with a fresh informed initialization.", flush=True)
            successes.append(self.se.solve(alpha, r, self.se.initial_state(r)))
        if not successes:
            raise NumericalFailure("; ".join(errors))
        result = min(successes, key=lambda item: item["free_energy"])
        self.cache[r][key] = result
        row = {k: result[k] for k in ("alpha", "r", "growth", "kappa", "g", "se_error",
                                      "iterations", "prox_error", "learned_eigenvector_error",
                                      "common_growth", "free_energy")}
        row["magnetization_norms"] = json.dumps(np.linalg.norm(result["state"].m, axis=0).tolist())
        self.trace.append(row)
        self.write_csv(self.directory / (self.stem + "_evaluations.csv"), self.trace)
        print(f"r={r} alpha={alpha:.10g}  growth={result['growth']:.9g}  "
              f"g={result['g']:+.3e}  SE={result['se_error']:.2e}  it={result['iterations']}", flush=True)
        return result

    @staticmethod
    def row(r, status, low=math.nan, high=math.nan, result=None, message=""):
        found = status == "crossing_found"
        width = high - low
        return {"r_from": r, "r_to": r + 1, "feature": r + 1, "status": status,
                "alpha_c": 0.5 * (low + high) if found else (math.inf if status == "zero_signal" else math.nan),
                "bracket_low": low, "bracket_high": high,
                "alpha_error_bound": 0.5 * width if found else math.nan,
                "growth_at_alpha_c": result["growth"] if found and result else math.nan,
                "growth_last_evaluated": result["growth"] if result else math.nan,
                "se_error": result["se_error"] if result else math.nan,
                "se_iterations": result["iterations"] if result else 0,
                "message": message}

    def bisect(self, r, low, high):
        if not low["g"] < 0 <= high["g"]:
            raise NumericalFailure("Bisection requires converged stable and unstable endpoints.")
        print(f"Bisection r={r}: [{low['alpha']:.10g}, {high['alpha']:.10g}]", flush=True)
        for _ in range(self.args.max_root_iter):
            if high["alpha"] - low["alpha"] <= self.args.alpha_tol:
                break
            result = self.evaluate(0.5 * (low["alpha"] + high["alpha"]), r)
            if result["g"] < 0:
                low = result
            else:
                high = result
        lo, hi = low["alpha"], high["alpha"]
        if hi - lo > self.args.alpha_tol:
            return self.row(r, "root_tolerance_not_met", lo, hi,
                            message="Increase --max-root-iter."), high
        center = self.evaluate(0.5 * (lo + hi), r)
        row = self.row(r, "crossing_found", lo, hi, center)
        row.update({"growth_low": low["growth"], "growth_high": high["growth"],
                    "learned_eigenvector_error": center["learned_eigenvector_error"]})
        return row, high

    def search_branch(self, r, previous=None, parent=None):
        a = self.args
        if a.nus[r] == 0.0:
            return self.row(r, "zero_signal", message="An exactly zero-strength feature cannot be recovered."), None
        grid = np.linspace(a.alpha_min, a.alpha_max, a.n_alphas)
        if r == 0:
            first = self.evaluate(a.alpha_min, r)
            if first["g"] >= 0:
                zero = {"alpha": 0.0, "g": -1.0, "growth": 0.0}
                return self.bisect(r, zero, first)
            stable = first
        else:
            if previous is None or previous["status"] != "crossing_found":
                return self.row(r, "previous_boundary_not_found",
                                message="The preceding continuous onset was not resolved."), None
            onset = previous["bracket_high"]
            if onset >= a.alpha_max:
                return self.row(r, "outside_search_interval", onset, math.inf), None
            step = a.onset_step if a.onset_step is not None else (a.alpha_max - a.alpha_min) / (a.n_alphas - 1)
            start_alpha = min(a.alpha_max, onset + max(step, 4 * a.alpha_tol))
            current = self.evaluate(start_alpha, r, parent)
            for _ in range(a.max_onset_refine):
                if current["g"] < 0:
                    break
                if current["alpha"] - previous["bracket_low"] <= a.alpha_tol:
                    break
                candidate = 0.5 * (onset + current["alpha"])
                if candidate == current["alpha"]:
                    break
                current = self.evaluate(candidate, r)
            if current["g"] >= 0:
                return self.row(r, "no_stable_interval_resolved", previous["bracket_low"], current["alpha"], current,
                                "The onsets may coincide or be closer than the resolved branch interval. "
                                "No separate bisection boundary is claimed."), None
            stable = current
        for alpha in grid:
            if alpha <= stable["alpha"]:
                continue
            result = self.evaluate(float(alpha), r)
            if result["g"] >= 0:
                return self.bisect(r, stable, result)
            stable = result
        return self.row(r, "stable_up_to_alpha_max", a.alpha_max, math.inf, stable,
                        "No crossing found inside the tested interval; increase --alpha-max."), None

    @staticmethod
    def write_csv(path, rows):
        if not rows:
            return
        fields = list(dict.fromkeys(key for row in rows for key in row))
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

    def save(self):
        metadata = {"lambda": self.args.lam, "F": len(self.args.nus), "H": self.args.H,
                    "L": self.args.L, "nus": json.dumps(self.args.nus),
                    "nu_total": sum(self.args.nus), "delta": self.args.delta,
                    "seed": self.args.seed, "n_mc": self.args.n_mc,
                    "sampling": self.args.sampling, "alpha_tol": self.args.alpha_tol,
                    "se_tol": self.args.tol, "regularization_convention": "lambda"}
        metadata.update({f"nu{f + 1}": nu for f, nu in enumerate(self.args.nus)})
        selected = [dict(metadata, **row) for row in self.rows if row["r_from"] >= self.args.first_r]
        self.write_csv(self.directory / (self.stem + ".csv"), selected)
        report = {"arguments": vars(self.args), "boundaries": self.rows,
                  "precision_note": "alpha_error_bound is half the finite-sample bisection bracket, not a bound on Gaussian integration error."}
        def finite_json(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: finite_json(item) for key, item in value.items()}
            if isinstance(value, list):
                return [finite_json(item) for item in value]
            return value
        path = self.directory / (self.stem + ".json")
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(finite_json(report), indent=2) + "\n")
        os.replace(temporary, path)
        arrays = {}
        for row in self.rows:
            if row["status"] != "crossing_found":
                continue
            r = row["r_from"]
            for side in ("low", "high"):
                alpha = row[f"bracket_{side}"]
                result = self.cache[r].get(alpha)
                if result is None:
                    continue
                for name in ("m", "q", "V"):
                    arrays[f"r{r}_{side}_{name}"] = getattr(result["state"], name)
                arrays[f"r{r}_{side}_J"] = result["J"]
                arrays[f"r{r}_{side}_alpha"] = np.array(alpha)
        if arrays:
            path = self.directory / (self.stem + "_states.npz")
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, path)

    def run(self):
        previous = parent = None
        failure = False
        for r in range(self.args.max_features):
            print(f"\nSearching {r}-SPEC -> {r + 1}-SPEC; next strength={self.args.nus[r]:.10g}", flush=True)
            try:
                row, next_parent = self.search_branch(r, previous, parent)
            except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
                row, next_parent = self.row(r, "numerical_failure", message=str(exc)), None
            self.rows.append(row)
            self.save()
            print(f"Boundary r={r}: {row['status']}; alpha_c={row['alpha_c']}; "
                  f"bracket=[{row['bracket_low']}, {row['bracket_high']}]", flush=True)
            if row["status"] in ("numerical_failure", "root_tolerance_not_met", "no_stable_interval_resolved"):
                failure = True
            previous, parent = row, next_parent
        print("\nSummary CSV:", self.directory / (self.stem + ".csv"), flush=True)
        return 1 if failure else 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    signals = p.add_mutually_exclusive_group(required=True)
    signals.add_argument("--nus", nargs="+", type=float)
    signals.add_argument("--nu-total", type=float)
    p.add_argument("--F", type=int, default=None)
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--first-r", type=int, default=0)
    p.add_argument("--alpha-min", type=float, default=0.05)
    p.add_argument("--alpha-max", type=float, default=10.0)
    p.add_argument("--n-alphas", type=int, default=25)
    p.add_argument("--alpha-tol", type=float, default=1e-3)
    p.add_argument("--max-root-iter", type=int, default=40)
    p.add_argument("--onset-step", type=float, default=None)
    p.add_argument("--max-onset-refine", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-mc", type=int, default=512)
    p.add_argument("--sampling", choices=("sobol", "normal"), default="sobol")
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--opt-tol", type=float, default=1e-9)
    p.add_argument("--damping", type=float, default=0.8)
    p.add_argument("--init-scale", type=float, default=0.4)
    p.add_argument("--min-magnetization", type=float, default=1e-6)
    p.add_argument("--seed-retries", type=int, choices=(0, 1), default=1)
    p.add_argument("--branch-starts", type=int, choices=(1, 2, 3), default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--prox-max-iter", type=int, default=80)
    p.add_argument("--output-dir", type=str, default="results_gaussian_phase_boundary")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    if a.nus is not None:
        if a.delta is not None:
            p.error("Use --delta with --nu-total, not with --nus.")
        if a.F is not None and a.F != len(a.nus):
            p.error("--F must match the number of entries in --nus.")
        a.F = len(a.nus)
    else:
        a.F = 2 if a.F is None else a.F
        if a.F < 2 or a.delta is None or not 0 <= a.delta <= 1:
            p.error("--nu-total requires --F >= 2 and --delta in [0,1].")
        if not math.isfinite(a.nu_total) or a.nu_total <= 0:
            p.error("--nu-total must be finite and positive.")
        a.nus = (a.nu_total / a.F * (1 + a.delta * np.linspace(1, -1, a.F))).tolist()
    if any(not math.isfinite(v) or v < 0 for v in a.nus) or any(x < y for x, y in zip(a.nus, a.nus[1:])):
        p.error("--nus must be finite, nonnegative, and in nonincreasing order.")
    a.H = 2 * a.F if a.H is None else a.H
    if a.L < 2 or a.H < 2 or not math.isfinite(a.lam) or a.lam <= 0:
        p.error("Require L>=2, H>=2 and finite lambda>0.")
    available = min(a.F, a.H // 2)
    a.max_features = available if a.max_features is None else a.max_features
    if not 1 <= a.max_features <= available or not 0 <= a.first_r < a.max_features:
        p.error("Require 1 <= max-features <= min(F,H//2) and 0 <= first-r < max-features.")
    if not 0 < a.alpha_min < a.alpha_max or not math.isfinite(a.alpha_max) or a.alpha_tol <= 0:
        p.error("Require 0 < alpha-min < alpha-max and alpha-tol > 0.")
    if a.n_alphas < 2 or not 0 <= a.damping < 1:
        p.error("Require n-alphas>=2 and damping in [0,1).")
    for name in ("n_mc", "max_iter", "max_root_iter", "batch_size", "prox_max_iter", "max_onset_refine"):
        if getattr(a, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive.")
    for name in ("tol", "opt_tol", "init_scale", "min_magnetization", "alpha_tol"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive.")
    if a.onset_step is not None and (not math.isfinite(a.onset_step) or a.onset_step <= 0):
        p.error("--onset-step must be finite and positive.")
    if a.sampling == "sobol" and a.n_mc & (a.n_mc - 1):
        p.error("Sobol sampling requires n-mc=2**k (e.g. 256, 512, 1024); use --sampling normal otherwise.")
    return a


def main():
    args = parse_args()
    print(f"Gaussian RS boundaries: nus={args.nus}, lambda={args.lam}, H={args.H}, L={args.L}", flush=True)
    print(f"Summary transitions: r={args.first_r},...,{args.max_features - 1}; "
          "PS onset is computed internally to initialize the cascade.", flush=True)
    return BoundarySearch(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
