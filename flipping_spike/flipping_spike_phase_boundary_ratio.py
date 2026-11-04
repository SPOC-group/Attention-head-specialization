#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import ndtri, softmax
from scipy.stats import qmc


class NumericalFailure(RuntimeError):
    pass


def sym(a):
    return 0.5 * (a + np.swapaxes(a, -1, -2))


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


def proximal_batch(omega, precision, start, tol, max_iter):
    n, L, d = omega.shape
    D = np.diag(np.tile(precision, L))
    eye = np.eye(L * d)
    y = omega.copy() if start is None else start.copy()
    for iteration in range(max_iter + 1):
        loss, grad_loss, B = ps_derivatives(y)
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
            logits = trial[:, :, 0] / np.sqrt(2.0)
            p = softmax(logits, axis=1)
            residual = p
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


class PSSolver:
    def __init__(self, args):
        self.args = args
        if args.sampling == "sobol":
            points = qmc.Sobol(2 * args.L, scramble=True, seed=args.seed).random_base2(
                args.n_mc.bit_length() - 1)
            self.noise = ndtri(np.clip(points, 1e-15, 1 - 1e-15)).reshape(args.n_mc, args.L, 2)
        else:
            self.noise = np.random.default_rng(args.seed).normal(size=(args.n_mc, args.L, 2))

    def moments(self, state, warm=None):
        a = self.args
        fields = self.noise[:, :, :1] * np.sqrt(state.q_parallel)
        fields[:, 0, 0] += np.sqrt(a.nu1) * state.m_parallel
        precision = np.array([1.0 / state.v_parallel])
        z_all = np.empty_like(fields)
        stats = dict(g1=0.0, g2=0.0, trace_J=0.0, envelope=0.0, prox_error=0.0)
        spectra, weights = [], []
        for lo in range(0, a.n_mc, a.batch_size):
            hi = min(lo + a.batch_size, a.n_mc)
            start = None if warm is None else warm[lo:hi]
            try:
                z, g, J, envelope, error = proximal_batch(
                    fields[lo:hi], precision, start, a.opt_tol, a.prox_max_iter)
            except NumericalFailure:
                if start is None:
                    raise
                z, g, J, envelope, error = proximal_batch(
                    fields[lo:hi], precision, None, a.opt_tol, a.prox_max_iter)
            z_all[lo:hi] = z
            stats["g1"] += float(g[:, 0, 0].sum()) / a.n_mc
            stats["g2"] += float(np.sum(g**2)) / a.n_mc
            stats["trace_J"] += float(np.trace(J, axis1=1, axis2=2).sum()) / a.n_mc
            stats["envelope"] += float(envelope.sum()) / a.n_mc
            stats["prox_error"] = max(stats["prox_error"], error)
            values, vectors = np.linalg.eigh(ps_derivatives(z, both=True)[3])
            values[np.abs(values) < 1e-13 * np.maximum(
                1.0, np.max(np.abs(values), axis=1))[:, None]] = 0.0
            spectra.append(values)
            weights.append(vectors[:, 0, :]**2)
        stats.update(warm=z_all, spectrum=np.concatenate(spectra),
                     signal_weights=np.concatenate(weights))
        return stats

    def solve(self, alpha, initial=None):
        a = self.args
        if alpha == 0:
            state = State(0.0, 0.0, 0.0, 1.0 / a.lam, 1.0 / a.lam)
            stats = self.moments(state)
            stats.pop("warm")
            return dict(alpha=0.0, state=state, stats=stats, se_error=0.0,
                        iterations=0, evaluations=1, prox_error=stats["prox_error"], free_energy=0.0)
        if initial is None or initial.r_parallel == 0:
            scale = max(alpha, 1e-5)
            v = 1.0 / max(a.lam, 0.05)
            initial = State(max(np.sqrt(a.nu1) * scale * 0.1, 1e-10),
                            max(scale * 0.05, 1e-10), 0.0, v, v)
        names = (["m_parallel"] if a.nu1 > 0 else []) + ["r_parallel", "v_parallel"]
        x0 = np.log([max(getattr(initial, name), 1e-12) for name in names])
        last = dict(warm=None, calls=0)

        def equations(x):
            values = dict(zip(names, np.exp(x)))
            state = State(values.get("m_parallel", 0.0), values["r_parallel"],
                          0.0, values["v_parallel"], 1.0 / a.lam)
            stats = self.moments(state, last["warm"])
            last.update(warm=stats["warm"], stats=stats, state=state, calls=last["calls"] + 1)
            v = state.v_parallel
            target_m = alpha * v * np.sqrt(a.nu1) * stats["g1"]
            target_r = alpha * v**2 * stats["g2"]
            residual = ([1.0 - target_m / state.m_parallel] if a.nu1 > 0 else [])
            residual += [1.0 - target_r / state.r_parallel,
                         1.0 - v * (a.lam - alpha * stats["trace_J"])]
            if a.verbose and last["calls"] % 50 == 0:
                print(f"    PS alpha={alpha:.8g} calls={last['calls']} "
                      f"residual={max(abs(t) for t in residual):.3e}", flush=True)
            return np.asarray(residual)

        try:
            fit = least_squares(equations, np.clip(x0, -27.0, 13.0), bounds=(-28.0, 14.0),
                                diff_step=1e-4, xtol=1e-10, ftol=1e-10, gtol=1e-10,
                                max_nfev=a.max_iter)
        except (ValueError, OverflowError) as exc:
            raise NumericalFailure(f"PS solve failed at alpha={alpha:.10g}: {exc}") from exc
        error = float(np.max(np.abs(equations(fit.x))))
        if not math.isfinite(error) or error > a.tol:
            raise NumericalFailure(f"PS equations did not converge at alpha={alpha:.10g}; "
                                   f"relative residual={error:.3e} (required {a.tol:.3e}).")
        state, stats = last["state"], last["stats"]
        stats.pop("warm")
        potential = (alpha * stats["envelope"] + 0.5 * a.lam * state.q_parallel
                     - 0.5 * state.r_parallel / state.v_parallel)
        return dict(alpha=float(alpha), state=state, stats=stats, se_error=error,
                    iterations=fit.nfev, evaluations=last["calls"],
                    prox_error=stats["prox_error"], free_energy=float(potential))


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
        self.se = PSSolver(args)
        self.cache, self.trace, self.scan_failures = {}, [], []
        config = {k: v for k, v in vars(args).items() if k not in ("output_dir", "verbose")}
        config["schema_version"] = 1
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        self.stem = (f"alpha_critical_flipping_spike_ratio_L{args.L}_H2"
                     f"_nutot_{args.nu_total:.8g}_lam_{args.lam:.8g}"
                     f"_delta_{args.delta:.12g}_seed_{args.seed}_{digest}")
        self.directory = Path(args.output_dir).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = {"delta": args.delta, "nu_total": args.nu_total,
                         "nu1": args.nu1, "nu2": args.nu2, "lambda": args.lam,
                         "L": args.L, "H": 2, "F": 2, "seed": args.seed,
                         "n_mc": args.n_mc, "sampling": args.sampling,
                         "alpha_tol": args.alpha_tol, "stability_tol": args.stability_tol,
                         "se_tol": args.tol, "regularization_convention": "lambda"}
        if args.delta_grid is not None:
            lo, hi, count = args.delta_grid
            self.metadata.update(delta_min=lo, delta_max=hi, n_deltas=int(count))

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

    def evaluate(self, alpha):
        alpha = float(alpha)
        if alpha in self.cache:
            return self.cache[alpha]
        nearest = self.cache[min(self.cache, key=lambda x: abs(alpha - x))]["state"] if self.cache else None
        result = ps_transverse(self.se.solve(alpha, nearest), self.args)
        result.update(covariance_margin=result["g"], covariance_gain=result["growth"],
                      g=result["specialization_gain"] - 1.0,
                      growth=result["specialization_gain"])
        if result["ps_susceptibility_exists"] and not all(math.isfinite(result[key])
                for key in ("g", "v_perp", "covariance_gain", "se_error")):
            raise NumericalFailure(f"Nonfinite PS response at alpha={alpha:.10g}.")
        self.cache[alpha] = result
        row = dict(self.metadata, branch="PS", valid=True, message="",
                   criterion="PS_SPEC_gain-1", m_perp=0.0, q_parallel=result["state"].q_parallel)
        row.update({key: result.get(key, math.nan) for key in (
            "alpha", "g", "growth", "covariance_gain", "covariance_margin",
            "specialization_gain", "lambda_edge", "v_edge", "edge_specialization_gain",
            "ps_susceptibility_exists", "free_energy", "se_error", "prox_error", "iterations", "evaluations")})
        row.update(asdict(result["state"]))
        self.trace.append(row)
        self.write_csv(self.directory / (self.stem + "_evaluations.csv"), self.trace)
        print(f"PS alpha={alpha:.10g} spec_gain={result['growth']:.8g} "
              f"cov_gain={result['covariance_gain']:.8g} SE={result['se_error']:.2e}", flush=True)
        return result

    @staticmethod
    def boundary(status, low=math.nan, high=math.nan, result=None, message=""):
        found = status == "crossing_found"
        return {"transition": "PS_SPEC", "status": status,
                "alpha_c": 0.5 * (low + high) if found else (math.inf if status == "zero_signal" else math.nan),
                "bracket_low": low, "bracket_high": high,
                "alpha_error_bound": 0.5 * (high - low) if found else math.nan,
                "criterion": "PS_SPEC_gain-1",
                "criterion_last": result["g"] if result else math.nan,
                "growth_last": result["growth"] if result else math.nan,
                "se_error": result["se_error"] if result else math.nan,
                "message": message}

    def bisect(self, low, high):
        a = self.args
        if not low["g"] < 0 <= high["g"]:
            raise NumericalFailure("Bisection requires valid stable and unstable endpoints.")
        for _ in range(a.max_root_iter):
            center = self.evaluate(0.5 * (low["alpha"] + high["alpha"]))
            if not center["ps_susceptibility_exists"]:
                raise NumericalFailure("PS susceptibility lost inside the specialization bracket.")
            width = high["alpha"] - low["alpha"]
            if width <= a.alpha_tol and abs(center["g"]) <= a.stability_tol:
                row = self.boundary("crossing_found", low["alpha"], high["alpha"], center)
                row.update(criterion_low=low["g"], criterion_high=high["g"],
                           growth_low=low["growth"], growth_high=high["growth"],
                           growth_at_alpha_c=center["growth"],
                           scaled_marginality_residual=abs(center["g"]),
                           covariance_gain_at_alpha_c=center["covariance_gain"],
                           v_perp_at_alpha_c=center["v_perp"],
                           se_error=max(r["se_error"] for r in (low, center, high)))
                return row
            if width <= 1e-12 * max(1.0, abs(center["alpha"])):
                break
            if center["g"] < 0:
                low = center
            else:
                high = center
        status = ("root_tolerance_not_met" if high["alpha"] - low["alpha"] > a.alpha_tol
                  else "marginality_not_resolved")
        row = self.boundary(status, low["alpha"], high["alpha"], center,
                            "Both the alpha bracket and |specialization_gain-1| must converge. "
                            "Increase max-root-iter; inspect the trace for branch switches or integration error.")
        row.update(criterion_low=low["g"], criterion_high=high["g"],
                   scaled_marginality_residual=abs(center["g"]))
        return row

    def approach_susceptibility_edge(self, stable, invalid):
        a = self.args
        for _ in range(a.max_root_iter):
            width = invalid["alpha"] - stable["alpha"]
            edge_gain = invalid["edge_specialization_gain"]
            if width <= a.alpha_tol and edge_gain < 1.0 - a.stability_tol:
                row = self.boundary("ps_susceptibility_lost_before_specialization",
                                    stable["alpha"], invalid["alpha"], stable,
                                    "The PS susceptibility ends without a resolved specialization crossing. "
                                    "The assumed direct PS->SPEC regime was not found at this resolution; "
                                    "increase regularization or use the full boundary solver.")
                row["edge_specialization_gain"] = edge_gain
                return row
            if width <= 1e-12 * max(1.0, invalid["alpha"]):
                break
            current = self.evaluate(0.5 * (stable["alpha"] + invalid["alpha"]))
            if current["ps_susceptibility_exists"]:
                if current["g"] >= 0:
                    return self.bisect(stable, current)
                stable = current
            else:
                invalid = current
        return self.boundary("coincident_or_unresolved", stable["alpha"], invalid["alpha"], stable,
                             "Could not separate specialization from the end of the PS susceptibility branch.")

    def search(self):
        a = self.args
        if a.nu2 == 0:
            return self.boundary("zero_signal", message="No flipping signal: alpha_c is infinite.")
        stable = self.evaluate(0.0)
        if not stable["ps_susceptibility_exists"]:
            return self.boundary("no_stable_zero_limit", 0.0, 0.0, stable,
                                 "The assumed PS proximal branch is not stable as alpha tends to zero.")
        for alpha in np.linspace(a.alpha_min, a.alpha_max, a.n_alphas):
            current = self.evaluate(alpha)
            if not current["ps_susceptibility_exists"]:
                return self.approach_susceptibility_edge(stable, current)
            if current["g"] >= 0:
                return self.bisect(stable, current)
            stable = current
        return self.boundary("stable_up_to_alpha_max", a.alpha_max, math.inf, stable,
                             "No sampled PS->SPEC crossing in the search interval. "
                             "Increase alpha-max or refine n-alphas; alpha_c has not been determined.")

    def scan_diagnostics(self):
        for alpha in np.linspace(self.args.alpha_min, self.args.alpha_max, self.args.n_alphas):
            try:
                self.evaluate(alpha)
            except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
                failure = dict(self.metadata, alpha=float(alpha), branch="PS", valid=False, message=str(exc))
                self.scan_failures.append(failure)
                self.trace.append(failure)
                print(f"Unresolved diagnostic point alpha={alpha:.8g}: {exc}", flush=True)
        self.write_csv(self.directory / (self.stem + "_evaluations.csv"), self.trace)

    def save(self, row):
        self.write_csv(self.directory / (self.stem + ".csv"), [dict(self.metadata, **row)])
        report = {"schema_version": 1, "model": "flipping_spike", "arguments": vars(self.args),
                  "delta_definition": "nu2/(nu1+nu2)", "boundaries": [row],
                  "diagnostic_scan_failures": self.scan_failures,
                  "notes": ["Only the PS->SPEC local stability threshold is computed.",
                            "Interpreted as a phase boundary in the direct continuous-transition regime.",
                            "alpha_error_bound is half the root bracket, not a Gaussian integration error bound.",
                            "The first sampled stable-to-unstable crossing is refined. Narrow/reentrant intervals may require a finer scan.",
                            "Nonfinite values are null in JSON; inspect status. zero_signal means alpha_c=inf."]}

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
        arrays = {key: np.asarray(value) for key, value in dict(self.metadata, **row).items()}
        if row["status"] == "crossing_found":
            for side, alpha in (("low", row["bracket_low"]), ("high", row["bracket_high"]),
                                ("critical", row["alpha_c"])):
                result = self.cache[alpha]
                prefix = f"PS_SPEC_{side}"
                for key, value in asdict(result["state"]).items():
                    arrays[f"{prefix}_{key}"] = np.asarray(value)
                for key in ("alpha", "g", "growth", "covariance_gain", "lambda_edge", "v_edge"):
                    arrays[f"{prefix}_{key}"] = np.asarray(result[key])
        path = self.directory / (self.stem + "_states.npz")
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)

    def run(self):
        try:
            row = self.search()
        except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
            row = self.boundary("numerical_failure", message=str(exc))
        self.save(row)
        print(f"PS_SPEC: {row['status']}; alpha_c={row['alpha_c']}; "
              f"bracket=[{row['bracket_low']}, {row['bracket_high']}]", flush=True)
        if row["message"]:
            print(row["message"], flush=True)
        if self.args.scan_diagnostics and self.args.nu2 > 0:
            self.scan_diagnostics()
            self.save(row)
        print("Summary CSV:", self.directory / (self.stem + ".csv"), flush=True)
        return int(row["status"] not in {"crossing_found", "zero_signal", "stable_up_to_alpha_max"}
                   or bool(self.scan_failures))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nu-total", type=float, required=True)
    p.add_argument("--delta", type=float, required=True)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--H", type=int, choices=(2,), default=2)
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--alpha-min", type=float, default=0.01)
    p.add_argument("--alpha-max", type=float, default=20.0)
    p.add_argument("--n-alphas", type=int, default=50)
    p.add_argument("--alpha-tol", type=float, default=1e-3)
    p.add_argument("--stability-tol", type=float, default=1e-4)
    p.add_argument("--max-root-iter", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-mc", type=int, default=1024)
    p.add_argument("--sampling", choices=("sobol", "normal"), default="sobol")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--opt-tol", type=float, default=1e-10)
    p.add_argument("--prox-max-iter", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output-dir", default="results_flipping_spike_phase_boundary_ratio")
    p.add_argument("--delta-grid", nargs=3, type=float, metavar=("MIN", "MAX", "COUNT"))
    p.add_argument("--scan-diagnostics", action="store_true")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    for name in ("nu_total", "lam", "alpha_min", "alpha_max", "alpha_tol", "stability_tol", "tol", "opt_tol"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive.")
    if not math.isfinite(a.delta) or not 0 <= a.delta <= 1:
        p.error("--delta must be finite and in [0,1].")
    if a.L < 2 or a.alpha_max <= a.alpha_min or a.n_alphas < 2:
        p.error("Require L>=2, alpha-max>alpha-min and n-alphas>=2.")
    for name in ("n_mc", "max_iter", "max_root_iter", "prox_max_iter", "batch_size"):
        if getattr(a, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive.")
    if a.seed < 0:
        p.error("--seed must be nonnegative.")
    if a.sampling == "sobol" and a.n_mc & (a.n_mc - 1):
        p.error("Sobol integration requires n-mc=2**k; use --sampling normal for other counts.")
    if a.sampling == "sobol" and 2 * a.L > 21201:
        p.error("Requested dimension exceeds the Sobol generator limit; use --sampling normal.")
    if a.delta_grid is not None:
        lo, hi, count = a.delta_grid
        if (not all(math.isfinite(x) for x in a.delta_grid)
                or not 0 <= lo <= hi <= 1 or count < 1 or count != int(count)):
            p.error("delta-grid requires 0<=MIN<=MAX<=1 and a positive integer COUNT.")
        if not lo - 1e-14 <= a.delta <= hi + 1e-14:
            p.error("--delta must lie inside --delta-grid.")
    a.nu1, a.nu2 = a.nu_total * (1.0 - a.delta), a.nu_total * a.delta
    return a


def main():
    args = parse_args()
    print(f"Flipping-spike PS->SPEC: delta={args.delta:.12g}, nu_total={args.nu_total}, "
          f"nu1={args.nu1}, nu2={args.nu2}, lambda={args.lam}, L={args.L}, n_mc={args.n_mc}", flush=True)
    return BoundarySearch(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
