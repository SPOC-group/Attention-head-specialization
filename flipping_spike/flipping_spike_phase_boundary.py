#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import ndtri, softmax
from scipy.stats import qmc


U = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)


class NumericalFailure(RuntimeError):
    pass


def sym(a):
    return 0.5 * (a + np.swapaxes(a, -1, -2))


def loss_derivatives(y):
    n, L, _ = y.shape
    p = softmax(y @ U.T, axis=1)
    residual = p.mean(axis=2)
    residual[:, 0] -= 1.0
    b = p * (residual[:, :, None]
             - np.sum(p * residual[:, :, None], axis=1, keepdims=True))
    gradient = (b / 2.0) @ U
    ph = p.transpose(0, 2, 1)
    jac = ph[..., :, None] * np.eye(L) - ph[..., :, None] * ph[..., None, :]
    B = np.einsum("nhla,nkat->nlhtk", jac, jac).reshape(n, 2 * L, 2 * L) / 4.0
    for h in range(2):
        bh, sh = b[:, :, h], p[:, :, h]
        B[:, h::2, h::2] += (bh[:, :, None] * np.eye(L)
                              - bh[:, :, None] * sh[:, None, :]
                              - sh[:, :, None] * bh[:, None, :]) / 2.0
    rotation = np.kron(np.eye(L), U)
    return 0.5 * np.sum(residual**2, axis=1), gradient, sym(rotation.T @ B @ rotation)


def ps_derivatives(y, both=False):
    L = y.shape[1]
    p = softmax(y[:, :, 0] / np.sqrt(2.0), axis=1)
    residual = p.copy()
    residual[:, 0] -= 1.0
    b = p * (residual - np.sum(p * residual, axis=1, keepdims=True))
    S = p[:, :, None] * np.eye(L) - p[:, :, None] * p[:, None, :]
    Bminus = (b[:, :, None] * np.eye(L) - b[:, :, None] * p[:, None, :]
              - p[:, :, None] * b[:, None, :]) / 2.0
    Bplus = sym(S @ S / 2.0 + Bminus)
    output = (0.5 * np.sum(residual**2, axis=1), b[:, :, None] / np.sqrt(2.0), Bplus)
    return (*output, sym(Bminus)) if both else output


def proximal_batch(omega, precision, start, tol, max_iter, ps=False):
    derivatives = ps_derivatives if ps else loss_derivatives
    n, L, d = omega.shape
    diagonal = np.tile(precision, L)
    D = np.diag(diagonal)
    eye = np.eye(L * d)
    y = omega.copy() if start is None else start.copy()
    for iteration in range(max_iter + 1):
        loss, grad_loss, B = derivatives(y)
        delta = y - omega
        gradient = (grad_loss + delta * precision).reshape(n, -1)
        hessian = sym(B + D)
        errors = np.max(np.abs(gradient), axis=1)
        minimum = np.linalg.eigvalsh(hessian)[:, 0]
        if np.max(errors) <= tol:
            if np.any(minimum <= 0):
                raise NumericalFailure("A stationary proximal point has a nonpositive Hessian.")
            C = sym(np.linalg.inv(hessian))
            J = sym(-B + B @ C @ B)
            envelope = loss + 0.5 * np.sum(delta**2 * precision, axis=(1, 2))
            return y, -grad_loss, J, envelope, float(np.max(errors))
        if iteration == max_iter:
            break
        shifted = hessian + np.maximum(1e-7 - minimum, 0)[:, None, None] * eye
        step = np.linalg.solve(shifted, gradient[..., None])[..., 0]
        slope = np.sum(gradient * step, axis=1)
        base = loss + 0.5 * np.sum(delta**2 * precision, axis=(1, 2))
        scale = np.ones(n)
        accepted = errors <= tol
        candidate = y.copy()
        for _ in range(40):
            trial = y - scale[:, None, None] * step.reshape(n, L, d)
            logits = trial[:, :, 0] / np.sqrt(2.0) if ps else trial @ U.T
            p = softmax(logits, axis=1)
            residual = p if ps else p.mean(axis=2)
            residual[:, 0] -= 1.0
            objective = 0.5 * np.sum(residual**2, axis=1)
            objective += 0.5 * np.sum((trial - omega)**2 * precision, axis=(1, 2))
            good = objective <= base - 1e-4 * scale * slope + 5e-15
            take = good & ~accepted
            candidate[take] = trial[take]
            accepted |= good
            if np.all(accepted):
                break
            scale[~accepted] *= 0.5
        if not np.all(accepted):
            raise NumericalFailure("Proximal line search failed.")
        y = candidate
    raise NumericalFailure("Proximal solve did not converge; increase --prox-max-iter.")


@dataclass
class State:
    m_parallel: float
    r_parallel: float
    q_perp: float
    v_parallel: float
    v_perp: float

    @property
    def q_parallel(self):
        return self.m_parallel**2 + self.r_parallel


class FlippingSE:
    def __init__(self, args):
        self.args = args
        if args.sampling == "sobol":
            points = qmc.Sobol(2 * args.L, scramble=True, seed=args.seed).random_base2(
                int(math.log2(args.n_mc)))
            self.noise = ndtri(np.clip(points, 1e-15, 1 - 1e-15)).reshape(args.n_mc, args.L, 2)
        else:
            self.noise = np.random.default_rng(args.seed).normal(size=(args.n_mc, args.L, 2))

    def moments(self, state, ps, warm=None):
        a = self.args
        d = 1 if ps else 2
        fields = self.noise[:, :, :d] * np.sqrt([state.q_parallel, state.q_perp][:d])
        fields[:, 0, 0] += np.sqrt(a.nu1) * state.m_parallel
        precision = 1.0 / np.array([state.v_parallel, state.v_perp][:d])
        z_all = np.empty_like(fields)
        stats = {"g1": 0.0, "g2": np.zeros(d), "trace_J": np.zeros(d),
                 "J11_perp": 0.0, "J2_perp": 0.0, "envelope": 0.0, "prox_error": 0.0}
        spectra, signal_weights = [], []
        for lo in range(0, a.n_mc, a.batch_size):
            hi = min(lo + a.batch_size, a.n_mc)
            start = None if warm is None else warm[lo:hi]
            try:
                z, g, J, envelope, error = proximal_batch(
                    fields[lo:hi], precision, start, a.opt_tol, a.prox_max_iter, ps)
            except NumericalFailure:
                if start is None:
                    raise
                z, g, J, envelope, error = proximal_batch(
                    fields[lo:hi], precision, None, a.opt_tol, a.prox_max_iter, ps)
            z_all[lo:hi] = z
            stats["g1"] += float(g[:, 0, 0].sum()) / a.n_mc
            stats["g2"] += np.sum(g**2, axis=(0, 1)) / a.n_mc
            stats["trace_J"] += np.diagonal(J, axis1=1, axis2=2).reshape(-1, a.L, d).sum(axis=(0, 1)) / a.n_mc
            stats["envelope"] += float(envelope.sum()) / a.n_mc
            stats["prox_error"] = max(stats["prox_error"], error)
            if ps:
                values, vectors = np.linalg.eigh(ps_derivatives(z, both=True)[3])
                values[np.abs(values) < 1e-13 * np.maximum(1.0, np.max(np.abs(values), axis=1))[:, None]] = 0.0
                spectra.append(values)
                signal_weights.append(vectors[:, 0, :]**2)
            else:
                Jminus = J[:, 1::2, 1::2]
                stats["J11_perp"] += float(Jminus[:, 0, 0].sum()) / a.n_mc
                stats["J2_perp"] += float(np.sum(Jminus**2)) / a.n_mc
        stats["warm"] = z_all
        if ps:
            stats["spectrum"] = np.concatenate(spectra)
            stats["signal_weights"] = np.concatenate(signal_weights)
        return stats

    def initial_state(self, alpha, ps):
        scale = max(alpha, 1e-5)
        v = 1.0 / max(self.args.lam, 0.05)
        return State(max(np.sqrt(self.args.nu1) * scale * 0.1, 1e-10),
                     max(scale * 0.05, 1e-10), 0.0 if ps else self.args.init_q,
                     v, v)

    def solve(self, alpha, ps, initial=None):
        a = self.args
        initial = initial or self.initial_state(alpha, ps)
        names = (["m_parallel"] if a.nu1 > 0 else []) + ["r_parallel", "v_parallel"]
        if not ps:
            names += ["q_perp", "v_perp"]
        x0 = np.log([max(getattr(initial, name), 1e-12) for name in names])
        last = {"warm": None, "calls": 0}

        def decode(x):
            values = dict(zip(names, np.exp(x)))
            return State(values.get("m_parallel", 0.0), values["r_parallel"],
                         values.get("q_perp", 0.0), values["v_parallel"],
                         values.get("v_perp", initial.v_perp))

        def equations(x):
            state = decode(x)
            stats = self.moments(state, ps, last["warm"])
            last.update(warm=stats["warm"], stats=stats, state=state, calls=last["calls"] + 1)
            v = np.array([state.v_parallel, state.v_perp])
            target_m = alpha * v[0] * np.sqrt(a.nu1) * stats["g1"]
            target_q = alpha * v[:len(stats["g2"])]**2 * stats["g2"]
            precision_residual = 1.0 - v[:len(stats["g2"])] * (a.lam - alpha * stats["trace_J"])
            residual = ([1.0 - target_m / state.m_parallel] if a.nu1 > 0 else [])
            residual += [1.0 - target_q[0] / state.r_parallel, precision_residual[0]]
            if not ps:
                residual += [1.0 - target_q[1] / state.q_perp, precision_residual[1]]
            if a.verbose and last["calls"] % 50 == 0:
                print(f"    {'PS' if ps else 'MEM'} alpha={alpha:.8g} calls={last['calls']} "
                      f"residual={max(abs(t) for t in residual):.3e} q_perp={state.q_perp:.4g}", flush=True)
            return np.asarray(residual)

        try:
            fit = least_squares(equations, np.clip(x0, -27.0, 13.0), bounds=(-28.0, 14.0),
                                diff_step=1e-4, xtol=1e-10, ftol=1e-10, gtol=1e-10,
                                max_nfev=a.max_iter)
        except (ValueError, OverflowError) as exc:
            raise NumericalFailure(f"Saddle solve failed at alpha={alpha:.10g}: {exc}") from exc
        error = float(np.max(np.abs(equations(fit.x))))
        if error > a.tol:
            raise NumericalFailure(f"{'PS' if ps else 'MEM'} equations did not converge at alpha={alpha:.10g}; "
                                   f"relative residual={error:.3e} (required {a.tol:.3e}).")
        state, stats = last["state"], last["stats"]
        if not ps and state.q_perp < a.min_q_perp:
            raise NumericalFailure("MEM collapsed to PS; no nonzero-variance branch was resolved.")
        potential = alpha * stats["envelope"] + 0.5 * a.lam * (state.q_parallel + state.q_perp)
        potential -= 0.5 * state.r_parallel / state.v_parallel
        if not ps:
            potential -= 0.5 * state.q_perp / state.v_perp
        stats.pop("warm", None)
        return {"alpha": float(alpha), "branch": "PS" if ps else "MEM", "state": state,
                "stats": stats, "se_error": error, "iterations": fit.nfev,
                "evaluations": last["calls"], "prox_error": stats["prox_error"],
                "free_energy": float(potential)}


def ps_transverse(result, args):
    alpha = result["alpha"]
    b = result["stats"]["spectrum"]

    def gain(v):
        return alpha * float(np.mean(np.sum((v * b / (1.0 + v * b))**2, axis=1)))

    def regularizer(v):
        return 1.0 / v - alpha * float(np.mean(np.sum(b / (1.0 + v * b), axis=1)))

    bmin = float(b.min())
    if alpha == 0:
        edge = max(0.0, -bmin)
        result.update(g=edge - args.lam, lambda_edge=edge,
                      v_edge=1.0 / edge if edge else math.inf, growth=0.0,
                      v_perp=1.0 / args.lam, ps_susceptibility_exists=args.lam > edge,
                      specialization_gain=0.0, edge_specialization_gain=0.0)
        return result

    def specialization_gain(v):
        J11 = np.mean(np.sum(-b / (1.0 + v * b) * result["stats"]["signal_weights"], axis=1))
        return alpha * args.nu2 * v * float(J11)

    if bmin < 0:
        maximum = -1.0 / bmin
        hi = maximum * (1.0 - 1e-10)
    else:
        hi = 1.0 / args.lam
        for _ in range(80):
            if gain(hi) >= 1.0:
                break
            hi *= 2.0
        else:
            edge = 0.0
            v = brentq(lambda x: regularizer(x) - args.lam, 1e-14, hi)
            result.update(g=-args.lam, lambda_edge=edge, v_edge=math.inf,
                          growth=gain(v), v_perp=v, ps_susceptibility_exists=True,
                          specialization_gain=specialization_gain(v), edge_specialization_gain=math.nan)
            result["state"].v_perp = v
            return result
    if gain(hi) <= 1.0:
        raise NumericalFailure("Could not bracket the transverse spectral edge.")
    v_edge = brentq(lambda v: gain(v) - 1.0, 0.0, hi, xtol=1e-12, rtol=1e-12)
    edge = regularizer(v_edge)
    margin = edge - args.lam
    exists = margin <= 0.0
    v = brentq(lambda x: regularizer(x) - args.lam, 1e-14, v_edge,
               xtol=1e-12, rtol=1e-12) if exists else math.nan
    result.update(g=margin, lambda_edge=edge, v_edge=v_edge,
                  edge_growth=gain(v_edge),
                  growth=gain(v) if exists else math.nan,
                  v_perp=v, ps_susceptibility_exists=exists,
                  specialization_gain=specialization_gain(v) if exists else math.nan,
                  edge_specialization_gain=specialization_gain(v_edge))
    result["state"].v_perp = v
    return result


class BoundarySearch:
    def __init__(self, args):
        self.args = args
        self.se = FlippingSE(args)
        self.cache = {"PS": {}, "MEM": {}}
        self.rows, self.trace = [], []
        config = {k: v for k, v in vars(args).items() if k not in ("output_dir", "verbose")}
        config["schema_version"] = 2
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        self.stem = (f"alpha_critical_flipping_spike_L{args.L}_H2_nu1_{args.nu1:.6g}"
                     f"_nu2_{args.nu2:.6g}_lam_{args.lam:.6g}_seed_{args.seed}_{digest}")
        self.directory = Path(args.output_dir).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)

    def evaluate(self, alpha, branch):
        alpha = float(alpha)
        cache = self.cache[branch]
        if alpha in cache:
            return cache[alpha]
        a = self.args
        nearest = None
        if cache:
            nearest_alpha = min(cache, key=lambda x: abs(alpha - x))
            nearest = cache[nearest_alpha]["state"]
        if branch == "PS":
            if alpha == 0:
                state = State(0.0, 0.0, 0.0, 1.0 / a.lam, 1.0 / a.lam)
                stats = self.se.moments(state, True)
                stats.pop("warm", None)
                result = {"alpha": alpha, "branch": branch, "state": state, "stats": stats,
                          "se_error": 0.0, "prox_error": stats["prox_error"], "free_energy": 0.0,
                          "iterations": 0, "evaluations": 1}
            else:
                result = self.se.solve(alpha, True, nearest)
            result = ps_transverse(result, a)
        else:
            successes, errors = [], []
            if nearest is not None:
                try:
                    successes.append(self.se.solve(alpha, False, nearest))
                except NumericalFailure as exc:
                    errors.append(str(exc))
            if not successes:
                if nearest is None:
                    ps = self.evaluate(alpha, "PS")
                    base = ps["state"]
                    v_seed = ps["v_edge"] if math.isfinite(ps["v_edge"]) else 1.0 / a.lam
                    q_seed = a.init_q
                else:
                    base, v_seed, q_seed = nearest, nearest.v_perp, nearest.q_perp
                for factor in (1.0, 5.0, 0.2)[:a.branch_starts]:
                    initial = replace(base, q_perp=q_seed * factor, v_perp=v_seed)
                    try:
                        successes.append(self.se.solve(alpha, False, initial))
                    except NumericalFailure as exc:
                        errors.append(str(exc))
            if not successes:
                raise NumericalFailure("; ".join(dict.fromkeys(errors)))
            result = min(successes, key=lambda item: item["free_energy"])
            state = result["state"]
            growth = alpha * a.nu2 * state.v_perp * result["stats"]["J11_perp"]
            result.update(g=growth - 1.0, growth=growth, specialization_gain=growth)
        cache[alpha] = result
        self.record(result)
        return result

    def record(self, result):
        a = self.args
        row = {key: result.get(key, math.nan) for key in (
            "branch", "alpha", "g", "growth", "specialization_gain", "lambda_edge", "v_edge",
            "edge_specialization_gain", "edge_growth", "ps_susceptibility_exists", "free_energy",
            "se_error", "prox_error", "iterations", "evaluations")}
        row.update(asdict(result["state"]))
        row.update(valid=True, message="", **{"lambda": a.lam},
                   nu1=a.nu1, nu2=a.nu2, L=a.L, seed=a.seed, n_mc=a.n_mc)
        row["q_parallel"] = result["state"].q_parallel
        row["m_perp"] = 0.0
        row["criterion"] = "lambda_edge-lambda" if result["branch"] == "PS" else "MEM_SPEC_gain-1"
        self.trace.append(row)
        self.write_csv(self.directory / (self.stem + "_evaluations.csv"), self.trace)
        print(f"{result['branch']} alpha={result['alpha']:.10g}  g={result['g']:+.6e}  "
              f"gain={result['growth']:.8g}  q_perp={result['state'].q_perp:.6g}  "
              f"spec_gain={result.get('specialization_gain', math.nan):.8g}  SE={result['se_error']:.2e}", flush=True)

    @staticmethod
    def boundary(transition, status, low=math.nan, high=math.nan, result=None, message=""):
        found = status == "crossing_found"
        return {"transition": transition, "status": status,
                "alpha_c": 0.5 * (low + high) if found else (math.inf if status == "zero_signal" else math.nan),
                "bracket_low": low, "bracket_high": high,
                "alpha_error_bound": 0.5 * (high - low) if found else math.nan,
                "criterion": "lambda_edge-lambda" if transition == "PS_MEM" else transition + "_gain-1",
                "criterion_last": result["g"] if result else math.nan,
                "growth_last": result["growth"] if result else math.nan,
                "se_error": result["se_error"] if result else math.nan,
                "message": message}

    def evaluate_transition(self, alpha, transition):
        result = self.evaluate(alpha, "MEM" if transition == "MEM_SPEC" else "PS")
        if transition == "PS_SPEC":
            if not result["ps_susceptibility_exists"]:
                raise NumericalFailure("PS-to-SPEC requires a real, stable PS transverse susceptibility.")
            gain = result["specialization_gain"]
            return dict(result, growth=gain, g=gain - 1.0)
        return result

    def bisect(self, transition, low, high):
        a = self.args
        if not low["g"] < 0 <= high["g"]:
            raise NumericalFailure("Bisection requires valid stable and unstable endpoints.")
        scale = a.lam if transition == "PS_MEM" else 1.0
        converged = False
        for _ in range(a.max_root_iter):
            midpoint = 0.5 * (low["alpha"] + high["alpha"])
            center = self.evaluate_transition(midpoint, transition)
            width = high["alpha"] - low["alpha"]
            if width <= a.alpha_tol and abs(center["g"]) / scale <= a.stability_tol:
                converged = True
                break
            if width <= 1e-12 * max(1.0, abs(midpoint)):
                break
            if center["g"] < 0:
                low = center
            else:
                high = center
        lo, hi = low["alpha"], high["alpha"]
        if not converged:
            status = "root_tolerance_not_met" if hi - lo > a.alpha_tol else "marginality_not_resolved"
            row = self.boundary(transition, status, lo, hi, center,
                                "The bracket and marginality equation must both converge. Increase max-root-iter if needed. "
                                "A persistent sign jump can indicate a nonsmooth proximal branch or insufficient integration resolution; inspect the trace and increase n-mc.")
            row.update(criterion_low=low["g"], criterion_high=high["g"],
                       scaled_marginality_residual=abs(center["g"]) / scale)
            return row
        row = self.boundary(transition, "crossing_found", lo, hi, center)
        row.update(criterion_low=low["g"], criterion_high=high["g"],
                   growth_low=low["growth"], growth_high=high["growth"],
                   scaled_marginality_residual=abs(center["g"]) / scale,
                   growth_at_alpha_c=center["growth"], se_error=max(low["se_error"], high["se_error"], center["se_error"]))
        if transition == "PS_MEM":
            row["growth_at_edge"] = center["edge_growth"]
            row["edge_specialization_gain"] = center["edge_specialization_gain"]
            row["v_perp_at_edge"] = center["v_edge"]
            row["message"] = ("The PS criterion uses the susceptibility fold; its gain is exactly one at the edge. "
                              "A NaN PS gain beyond the edge means no stable transverse susceptibility exists.")
        return row

    def search_ps(self):
        a = self.args
        stable = self.evaluate(0.0, "PS")
        if stable["g"] >= 0:
            return self.boundary("PS_MEM", "no_stable_zero_limit", 0.0, 0.0, stable,
                                 "The assumed symmetric proximal branch is not stable as alpha tends to zero.")
        for alpha in np.linspace(a.alpha_min, a.alpha_max, a.n_alphas):
            result = self.evaluate(alpha, "PS")
            if result["g"] >= 0:
                return self.bisect("PS_MEM", stable, result)
            stable = result
        return self.boundary("PS_MEM", "stable_up_to_alpha_max", a.alpha_max, math.inf, stable,
                             "No sampled PS-to-MEM crossing; increase alpha-max or refine n-alphas. "
                             "This does not assert stability against specialization.")

    def search_ps_spec(self, onset):
        a = self.args
        if a.nu2 == 0:
            return self.boundary("PS_SPEC", "zero_signal", message="The flipping feature has zero strength.")
        stable = self.evaluate_transition(0.0, "PS_SPEC")
        upper = onset["bracket_low"] if onset["status"] == "crossing_found" else a.alpha_max
        scan_points = [float(x) for x in np.linspace(a.alpha_min, a.alpha_max, a.n_alphas) if x <= upper]
        cached_points = [x for x in self.cache["PS"] if 0 < x <= upper]
        grid = sorted(set([upper] + scan_points + cached_points))
        invalid = None
        for alpha in grid:
            if alpha <= stable["alpha"]:
                continue
            raw = self.evaluate(alpha, "PS")
            if not raw["ps_susceptibility_exists"]:
                invalid = raw
                break
            current = self.evaluate_transition(alpha, "PS_SPEC")
            if current["g"] >= 0:
                return self.bisect("PS_SPEC", stable, current)
            stable = current
        if onset["status"] == "crossing_found":
            invalid = self.cache["PS"][onset["bracket_high"]]
        if invalid is None:
            return self.boundary("PS_SPEC", "stable_up_to_alpha_max", a.alpha_max, math.inf, stable,
                                 "No sampled PS magnetization instability inside the search interval.")
        hi = invalid["alpha"]
        for _ in range(a.max_root_iter):
            if invalid["edge_specialization_gain"] < 1.0 - a.stability_tol:
                status = "preempted_by_PS_MEM" if onset["status"] == "crossing_found" else "ps_susceptibility_lost_unresolved"
                return self.boundary("PS_SPEC", status, stable["alpha"], hi, stable,
                                     "No mean-mode crossing before the real PS susceptibility branch ends. "
                                     "A resolved PS_MEM boundary is required to call this preemption.")
            mid = 0.5 * (stable["alpha"] + hi)
            if hi - stable["alpha"] <= 1e-12 * max(1.0, hi):
                break
            raw = self.evaluate(mid, "PS")
            if raw["ps_susceptibility_exists"]:
                current = self.evaluate_transition(mid, "PS_SPEC")
                if current["g"] >= 0:
                    return self.bisect("PS_SPEC", stable, current)
                stable = current
            else:
                hi, invalid = mid, raw
        return self.boundary("PS_SPEC", "coincident_or_unresolved", stable["alpha"], hi, stable,
                             "The mean-mode crossing cannot be separated from the PS covariance fold at this resolution.")

    def search_mem(self, onset):
        a = self.args
        if a.nu2 == 0.0:
            return self.boundary("MEM_SPEC", "zero_signal", message="The flipping feature has zero strength.")
        explicit_start = a.mem_start_alpha is not None
        if not explicit_start and onset["status"] != "crossing_found":
            return self.boundary("MEM_SPEC", "ps_onset_not_found",
                                 message="No PS-connected MEM onset was resolved. Use mem-start-alpha only for a separately known MEM branch.")
        if explicit_start:
            current = self.evaluate(a.mem_start_alpha, "MEM")
            if current["g"] >= 0:
                return self.boundary("MEM_SPEC", "unstable_at_mem_start", current["alpha"], current["alpha"], current,
                                     "MEM is already unstable at mem-start-alpha; choose an earlier stable starting point.")
        else:
            lo, hi = onset["bracket_low"], onset["bracket_high"]
            limiting = [self.cache["PS"][x]["edge_specialization_gain"] for x in (lo, hi)]
            if min(limiting) >= 1.0:
                return self.boundary("MEM_SPEC", "unstable_at_mem_onset", lo, hi,
                                     message=f"The specialization gain in the PS onset limit is already >=1 ({min(limiting):.6g} to {max(limiting):.6g}). "
                                     "There is no initially stable PS-connected MEM interval to bisect.")
            if hi >= a.alpha_max:
                return self.boundary("MEM_SPEC", "outside_search_interval", hi, math.inf,
                                     message="The MEM onset is at alpha-max; increase alpha-max.")
            spacing = (a.alpha_max - a.alpha_min) / (a.n_alphas - 1)
            step = a.onset_step or max(8.0 * a.alpha_tol, min(0.05, spacing / 4.0))
            current, errors = None, []
            for attempt in range(a.max_onset_tries):
                alpha = min(a.alpha_max, hi + step * 2**attempt)
                try:
                    current = self.evaluate(alpha, "MEM")
                    break
                except NumericalFailure as exc:
                    errors.append(f"alpha={alpha:.8g}: {exc}")
                    print(f"  MEM initialization unresolved at alpha={alpha:.8g}; trying farther above onset.", flush=True)
                if alpha == a.alpha_max:
                    break
            if current is None:
                raise NumericalFailure("Could not initialize MEM above its onset. " + " | ".join(errors))
            for _ in range(a.max_onset_tries):
                if current["g"] < 0 or current["alpha"] - lo <= a.alpha_tol:
                    break
                current = self.evaluate(0.5 * (hi + current["alpha"]), "MEM")
            if current["g"] >= 0:
                return self.boundary("MEM_SPEC", "no_stable_interval_resolved", lo, current["alpha"], current,
                                     "No stable nonzero-variance MEM starting point was resolved; reduce onset-step, increase n-mc, or increase max-onset-tries.")
        stable = current
        grid = np.linspace(a.alpha_min, a.alpha_max, a.n_alphas)
        for alpha in grid:
            if alpha <= stable["alpha"]:
                continue
            result = self.evaluate(alpha, "MEM")
            if result["g"] >= 0:
                return self.bisect("MEM_SPEC", stable, result)
            stable = result
        return self.boundary("MEM_SPEC", "stable_up_to_alpha_max", a.alpha_max, math.inf, stable,
                             "No sampled MEM-to-SPEC crossing; increase alpha-max or refine n-alphas.")

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
        a = self.args
        metadata = {"lambda": a.lam, "nu1": a.nu1, "nu2": a.nu2, "L": a.L, "H": 2, "F": 2,
                    "seed": a.seed, "n_mc": a.n_mc, "sampling": a.sampling,
                    "alpha_tol": a.alpha_tol, "stability_tol": a.stability_tol,
                    "se_tol": a.tol, "regularization_convention": "lambda"}
        self.write_csv(self.directory / (self.stem + ".csv"), [dict(metadata, **row) for row in self.rows])
        report = {"schema_version": 2, "arguments": vars(a), "boundaries": self.rows,
                  "diagnostic_scan_failures": getattr(self, "scan_failures", []),
                  "notes": ["Local RS stability thresholds, not first-order free-entropy crossings.",
                            "alpha_error_bound is half the finite-sample bisection bracket, not a Gaussian integration error bound.",
                            "Continuation detects the first sampled stable-to-unstable crossing. Refine the scan to resolve narrow or reentrant intervals.",
                            "At finite n-mc, the noisy-field MEM onset can differ from the analytically averaged PS gain. Increase n-mc if onset initialization cannot be resolved.",
                            "PS and MEM backgrounds do not depend on nu2. nu2 multiplies the specialization response only."]}

        def finite_json(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: finite_json(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [finite_json(item) for item in value]
            return value

        path = self.directory / (self.stem + ".json")
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
        arrays = {}
        for row in self.rows:
            if row["status"] != "crossing_found":
                continue
            for side in ("low", "high"):
                result = self.evaluate_transition(row[f"bracket_{side}"], row["transition"])
                prefix = f"{row['transition']}_{side}"
                for name, value in asdict(result["state"]).items():
                    arrays[f"{prefix}_{name}"] = np.array(value)
                for key in ("alpha", "g", "growth", "lambda_edge", "v_edge", "specialization_gain"):
                    if key in result:
                        arrays[f"{prefix}_{key}"] = np.array(result[key])
        if arrays:
            path = self.directory / (self.stem + "_states.npz")
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, path)

    def scan_diagnostics(self):
        a = self.args
        self.scan_failures = []
        for branch in ("PS", "MEM"):
            if branch == "MEM" and not self.cache["MEM"]:
                continue
            start = 0.0 if branch == "PS" else min(self.cache["MEM"])
            for alpha in np.linspace(a.alpha_min, a.alpha_max, a.n_alphas):
                if alpha < start:
                    continue
                try:
                    self.evaluate(alpha, branch)
                except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
                    failure = {"branch": branch, "alpha": float(alpha), "valid": False, "message": str(exc)}
                    self.scan_failures.append(failure)
                    self.trace.append(failure)
                    self.write_csv(self.directory / (self.stem + "_evaluations.csv"), self.trace)
                    print(f"Diagnostic point unresolved: {branch} alpha={alpha:.8g}: {exc}", flush=True)
        self.save()

    def run(self):
        for transition in ("PS_MEM", "PS_SPEC", "MEM_SPEC"):
            print(f"\nSearching {transition}", flush=True)
            try:
                if transition == "PS_MEM":
                    row = self.search_ps()
                elif transition == "PS_SPEC":
                    row = self.search_ps_spec(self.rows[0])
                else:
                    row = self.search_mem(self.rows[0])
            except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
                row = self.boundary(transition, "numerical_failure", message=str(exc))
            self.rows.append(row)
            self.save()
            print(f"{transition}: {row['status']}; alpha_c={row['alpha_c']}; "
                  f"bracket=[{row['bracket_low']}, {row['bracket_high']}]", flush=True)
            if row["message"]:
                print("  " + row["message"], flush=True)
        if self.args.scan_diagnostics:
            self.scan_diagnostics()
        print("\nSummary CSV:", self.directory / (self.stem + ".csv"), flush=True)
        failures = {"numerical_failure", "root_tolerance_not_met", "marginality_not_resolved",
                    "no_stable_interval_resolved", "no_stable_zero_limit", "coincident_or_unresolved",
                    "ps_susceptibility_lost_unresolved"}
        return int(any(row["status"] in failures for row in self.rows) or bool(getattr(self, "scan_failures", [])))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nu1", type=float, required=True)
    parser.add_argument("--nu2", type=float, required=True)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--H", type=int, choices=(2,), default=2)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--alpha-min", type=float, default=0.01)
    parser.add_argument("--alpha-max", type=float, default=7.0)
    parser.add_argument("--n-alphas", type=int, default=30)
    parser.add_argument("--alpha-tol", type=float, default=1e-3)
    parser.add_argument("--stability-tol", type=float, default=1e-4)
    parser.add_argument("--max-root-iter", type=int, default=40)
    parser.add_argument("--onset-step", type=float, default=None)
    parser.add_argument("--max-onset-tries", type=int, default=10)
    parser.add_argument("--mem-start-alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-mc", type=int, default=512)
    parser.add_argument("--sampling", choices=("sobol", "normal"), default="sobol")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--opt-tol", type=float, default=1e-10)
    parser.add_argument("--prox-max-iter", type=int, default=120)
    parser.add_argument("--init-q", type=float, default=0.1)
    parser.add_argument("--min-q-perp", type=float, default=1e-7)
    parser.add_argument("--branch-starts", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", default="results_flipping_spike_phase_boundary")
    parser.add_argument("--scan-diagnostics", action="store_true")
    parser.add_argument("--lambda-grid", nargs=3, type=float, metavar=("MIN", "MAX", "COUNT"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    for name in ("nu1", "nu2"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name} must be finite and nonnegative.")
    for name in ("lam", "alpha_min", "alpha_max", "alpha_tol", "stability_tol", "tol", "opt_tol", "init_q", "min_q_perp"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
    if args.L < 2 or args.alpha_max <= args.alpha_min or args.n_alphas < 2:
        parser.error("Require L>=2, alpha-max>alpha-min and n-alphas>=2.")
    for name in ("n_mc", "max_iter", "max_root_iter", "max_onset_tries", "prox_max_iter", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.seed < 0:
        parser.error("--seed must be nonnegative.")
    if args.lambda_grid is not None:
        low, high, count = args.lambda_grid
        if not all(math.isfinite(x) for x in args.lambda_grid) or not 0 < low <= high or count < 1 or count != int(count):
            parser.error("lambda-grid requires 0<MIN<=MAX and a positive integer COUNT.")
    if args.onset_step is not None and (not math.isfinite(args.onset_step) or args.onset_step <= 0):
        parser.error("--onset-step must be finite and positive.")
    if args.mem_start_alpha is not None and not 0 < args.mem_start_alpha <= args.alpha_max:
        parser.error("Require 0 < mem-start-alpha <= alpha-max.")
    if args.sampling == "sobol" and args.n_mc & (args.n_mc - 1):
        parser.error("Sobol integration requires n-mc=2**k; use --sampling normal for other counts.")
    if args.sampling == "sobol" and 2 * args.L > 21201:
        parser.error("The requested dimension exceeds the Sobol generator's limit; use --sampling normal.")
    return args


def main():
    args = parse_args()
    print(f"Flipping-spike RS stability: nu1={args.nu1}, nu2={args.nu2}, "
          f"lambda={args.lam}, L={args.L}, H=2, n_mc={args.n_mc}", flush=True)
    print("Computing local stability thresholds; a first-order free-entropy crossing is a different boundary.", flush=True)
    return BoundarySearch(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
