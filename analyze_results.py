"""Statistical analysis for compute_opt_test.py, plus a GPU-free self-test.

Kept separate from the benchmark harness so the statistics can be validated without
CUDA, NVML, or model weights:

    python analyze_results.py --self-test

Depends on numpy only. Welch's t-test is implemented here via the regularized
incomplete beta function so that scipy is not required, and `--self-test` cross-checks
it against scipy whenever scipy happens to be importable.
"""

import argparse
import math

import numpy as np

WELCH_ALPHA = 0.05


# --------------------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------------------
def _betacf(a, b, x, max_iter=200, eps=3e-16, fpmin=1e-300):
    """Continued fraction for the incomplete beta function (Lentz's algorithm)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(ln_beta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t, dof):
    """Upper-tail survival function of Student's t with `dof` degrees of freedom."""
    if dof <= 0:
        return 1.0
    x = dof / (dof + t * t)
    tail = 0.5 * betainc(dof / 2.0, 0.5, x)
    return tail if t > 0 else (1.0 - tail if t < 0 else 0.5)


def welch_ttest(x, y):
    """Welch's unequal-variance t-test. Returns t, dof, two-sided p, n_x, n_y."""
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return {"t": 0.0, "dof": 0.0, "p": 1.0, "n_a": nx, "n_b": ny, "note": "insufficient_data"}
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    se2 = vx / nx + vy / ny
    if se2 <= 0.0:
        return {"t": 0.0, "dof": 0.0, "p": 1.0, "n_a": nx, "n_b": ny, "note": "zero_variance"}
    t = (mx - my) / math.sqrt(se2)
    dof = se2 ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return {"t": t, "dof": dof, "p": 2.0 * student_t_sf(abs(t), dof), "n_a": nx, "n_b": ny}


def cohens_d_pooled(x, y):
    """Cohen's d with the pooled SD (assumes comparable variance)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    sp2 = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    if sp2 <= 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / math.sqrt(sp2))


def glass_delta(x, y):
    """Glass's delta: difference standardized by the control arm's SD (y = reference)."""
    y = np.asarray(y, float)
    if len(y) < 2:
        return 0.0
    sd = float(np.std(y, ddof=1))
    if sd <= 0:
        return 0.0
    return float((np.mean(np.asarray(x, float)) - np.mean(y)) / sd)


def hedges_g(x, y):
    """Hedges' g: small-sample bias-corrected Cohen's d."""
    d = cohens_d_pooled(x, y)
    nx, ny = len(x), len(y)
    if nx + ny < 4:
        return d
    correction = 1.0 - 3.0 / (4.0 * (nx + ny) - 9.0)
    return float(d * correction)


# --------------------------------------------------------------------------------------
# Correlation helpers
# --------------------------------------------------------------------------------------
def _pearson_with_p(r, n):
    """Two-sided p-value for a Pearson correlation with n observations."""
    if n < 3 or abs(r) >= 1.0:
        return 0.0 if abs(r) >= 1.0 and n >= 3 else 1.0
    t = r * math.sqrt((n - 2) / (1.0 - r * r))
    return 2.0 * student_t_sf(abs(t), n - 2)


def pearson(x, y):
    if len(x) != len(y):
        return {"r": 0.0, "p": 1.0, "n": 0, "note": f"length_mismatch({len(x)} vs {len(y)})"}
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"r": 0.0, "p": 1.0, "n": n, "note": "degenerate"}
    r = float(np.corrcoef(x, y)[0, 1])
    return {"r": r, "p": _pearson_with_p(r, n), "n": n}


def partial_corr(x, y, z):
    """Pearson correlation of x and y after linearly removing z from both.

    Returns {"r", "p", "n", "note"} — never a bare tuple, and flags the degenerate
    case instead of silently reporting a zero correlation.
    """
    if not (len(x) == len(y) == len(z)):
        return {"r": 0.0, "p": 1.0, "n": 0,
                "note": f"length_mismatch({len(x)}, {len(y)}, {len(z)})"}
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    n = len(x)
    if n < 4:
        return {"r": 0.0, "p": 1.0, "n": n, "note": "insufficient_data"}

    def residual(a):
        design = np.column_stack([z, np.ones(n)])
        coef, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    rx, ry = residual(x), residual(y)
    # Compare residual dispersion to the raw dispersion: after removing z, a residual
    # that is constant to within float noise means z explained the variable entirely.
    def _flat(v, source):
        scale = float(np.max(np.abs(source))) or 1.0
        return float(np.ptp(v)) <= 1e-9 * scale

    if _flat(rx, x) or _flat(ry, y):
        return {"r": 0.0, "p": 1.0, "n": n, "note": "covariate_explains_all_variance"}
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - 3
    p = 1.0
    if dof > 0 and abs(r) < 1.0:
        t = r * math.sqrt(dof / (1.0 - r * r))
        p = 2.0 * student_t_sf(abs(t), dof)
    return {"r": r, "p": p, "n": n, "dof": dof}


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------
def format_p_value(p):
    """Never print an impossible exact zero, and never round 0.04 up to 0.05."""
    if p is None:
        return "n/a"
    if p < 1e-4:
        return "< 0.0001"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _p_phrase(p):
    """Render a p-value as valid markdown, keeping a strict inequality outside math mode."""
    formatted = format_p_value(p)
    if formatted.startswith("<"):
        return f"$p$ {formatted}"
    return f"$p = {formatted}$"


def _mean_sd(values):
    values = [float(v) for v in values]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def _ci95(values):
    values = [float(v) for v in values]
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1)) / math.sqrt(n)
    # Normal approximation; flagged as approximate in the rendered report.
    return (mean - 1.96 * se, mean + 1.96 * se)


# --------------------------------------------------------------------------------------
# Arm-level analysis
# --------------------------------------------------------------------------------------
def analyze_arms(results, labels):
    """Compute every statistic the report quotes, from raw per-trial lists."""
    arms = {}
    for label in labels:
        r = results[label]
        lengths = {k: len(r[k]) for k in
                   ("energies", "jpt_list", "tokens_gen", "surprises", "powers", "durations")}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"Arm '{label}' has inconsistent per-trial list lengths {lengths}; "
                "refusing to compute statistics that would silently misalign variables."
            )
        energies = list(r["energies"])
        jpt = list(r["jpt_list"])
        toks = list(r["tokens_gen"])
        surprises = list(r["surprises"])
        p_mean, p_sd = _mean_sd(r["powers"])
        d_mean, d_sd = _mean_sd(r["durations"])
        e_mean, e_sd = _mean_sd(energies)
        j_mean, j_sd = _mean_sd(jpt)
        t_mean = float(np.mean(toks)) if toks else float("nan")
        temp_mean = float(np.mean(r["temperatures"])) if r["temperatures"] else float("nan")
        arms[label] = {
            "n_valid": len(jpt),
            "power_w_mean": p_mean,
            "power_w_sd": p_sd,
            "duration_s_mean": d_mean,
            "duration_s_sd": d_sd,
            "energy_j_mean": e_mean,
            "energy_j_sd": e_sd,
            "jpt_mean": j_mean,
            "jpt_sd": j_sd,
            "jpt_cv_pct": (100.0 * j_sd / j_mean) if j_mean else float("nan"),
            "jpt_ci95": _ci95(jpt),
            "tokens_mean": t_mean,
            "tokens_min": min(toks) if toks else 0,
            "tokens_max": max(toks) if toks else 0,
            "tokens_sd": float(np.std(toks, ddof=1)) if len(toks) > 1 else 0.0,
            "surprisal_mean": float(np.mean(surprises)) if surprises else float("nan"),
            "temp_end_c_mean": temp_mean,
            "tokens_per_second_mean": (
                float(np.mean([t / d for t, d in zip(toks, r["durations"]) if d > 0]))
                if toks and any(d > 0 for d in r["durations"]) else float("nan")
            ),
            "prompt_tokens": r["prompt_tokens"][0] if r["prompt_tokens"] else None,
            "raw_r_surprisal_jpt": pearson(surprises, jpt),
            "partial_r_surprisal_jpt_given_tokens": partial_corr(surprises, jpt, toks),
        }

    a_label, b_label = labels[0], labels[1]
    a, b = arms[a_label], arms[b_label]
    welch_jpt = welch_ttest(results[a_label]["jpt_list"], results[b_label]["jpt_list"])
    welch_energy = welch_ttest(results[a_label]["energies"], results[b_label]["energies"])
    welch_power = welch_ttest(results[a_label]["powers"], results[b_label]["powers"])

    payload = {
        "arms": arms,
        "comparison": {
            "a": a_label,
            "b": b_label,
            "jpt_diff_mean": a["jpt_mean"] - b["jpt_mean"],
            "jpt_pct_diff": 100.0 * (a["jpt_mean"] - b["jpt_mean"]) / b["jpt_mean"] if b["jpt_mean"] else float("nan"),
            "duration_pct_diff": 100.0 * (a["duration_s_mean"] - b["duration_s_mean"]) / b["duration_s_mean"] if b["duration_s_mean"] else float("nan"),
            "tokens_diff_mean": a["tokens_mean"] - b["tokens_mean"],
        },
        "welch_t_jpt": welch_jpt,
        "welch_t_energy": welch_energy,
        "welch_t_power": welch_power,
        "effect_size": {
            "cohens_d_pooled_jpt": cohens_d_pooled(results[a_label]["jpt_list"], results[b_label]["jpt_list"]),
            "hedges_g_jpt": hedges_g(results[a_label]["jpt_list"], results[b_label]["jpt_list"]),
            "glass_delta_jpt": glass_delta(results[a_label]["jpt_list"], results[b_label]["jpt_list"]),
        },
        "significant_jpt": bool(welch_jpt["p"] < WELCH_ALPHA),
        "significant_energy": bool(welch_energy["p"] < WELCH_ALPHA),
    }

    # Guard rails against the errors found in the original report.
    warnings = []
    if not math.isnan(a["jpt_cv_pct"]) and not math.isnan(b["jpt_cv_pct"]) \
            and abs(a["jpt_cv_pct"] - b["jpt_cv_pct"]) > 25.0:
        warnings.append("Arm variances differ substantially; pooled Cohen's d is unreliable — prefer Glass's delta.")
    if a["n_valid"] != b["n_valid"]:
        warnings.append(f"Unbalanced arms ({a['n_valid']} vs {b['n_valid']}) — arm order may be confounded with drift.")
    if results[a_label]["prompt_tokens"] and results[b_label]["prompt_tokens"]:
        delta = abs(results[a_label]["prompt_tokens"][0] - results[b_label]["prompt_tokens"][0])
        if delta > 0:
            warnings.append(
                f"Attended prompt lengths differ by {delta} tokens; masked padding does not "
                "equalize prefill work. Prefill cost is not controlled in this run."
            )
    if payload["significant_jpt"] and not payload["significant_energy"]:
        warnings.append(
            "J/token differs significantly while total energy does not: the arms trade power "
            "against runtime, so the effect is on efficiency per token only, not on total energy."
        )
    if not payload["significant_energy"] and payload["welch_t_power"]["p"] < WELCH_ALPHA:
        warnings.append(
            "Active power differs but total energy does not — consistent with the arms producing "
            "different numbers of tokens over different runtimes."
        )
    payload["warnings"] = warnings
    return payload


# --------------------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------------------
def render_report(payload, labels):
    arms = payload["arms"]
    run = payload.get("run", {})
    a_label, b_label = labels[0], labels[1]
    a, b = arms[a_label], arms[b_label]
    welch = payload["welch_t_jpt"]
    d = payload["effect_size"]["cohens_d_pooled_jpt"]

    def fmt(value, spec="{:.2f}"):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        return spec.format(value)

    lines = []
    lines.append("### ⚡ Local LLM Compute Analysis Report")
    lines.append("")
    lines.append(f"* **Hardware:** {run.get('gpu', 'unknown')}")
    lines.append(f"* **Model:** `{run.get('model', 'unknown')}` "
                 f"({'4-bit quantized' if run.get('quantized') else 'FP16 unquantized'})")
    lines.append(f"* **Trials:** {a['n_valid']} valid per arm ({a['n_valid'] + b['n_valid']} total), "
                 f"seed {run.get('seed')}, {run.get('attempts', '?')} attempts")
    lines.append(f"* **Decoding:** `max_new_tokens={run.get('max_new_tokens')}`, `temperature=0.7`, "
                 "sampling enabled")
    lines.append(f"* **Controls:** attention-masked left padding; per-trial local idle baseline "
                 f"({run.get('idle_seconds')} s) subtracted from the decode-phase power average; "
                 f"{run.get('cooldown_s')} s pre-trial cooldown; degenerate-output guard "
                 f"($N \\ge {MIN_VALID_TOKENS_FOR_REPORT}$ tokens) with redraw to keep arms balanced")
    lines.append(f"* **Surprise:** mean negative log-likelihood of sampled tokens, in nats "
                 "(per-token surprisal, *not* output entropy)")
    lines.append("")
    lines.append("| Prompt Arm | Active Power (W) [Mean ± SD] | Decode Duration (s) [Mean ± SD] | "
                 "Tokens Gen [Mean] | Total Energy (J) [Mean ± SD] | Efficiency [Mean J/t] | "
                 "Surprise [Mean Nats] | End Temp (°C) [Mean] |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for label in labels:
        s = arms[label]
        lines.append(
            f"| **{label}** | {fmt(s['power_w_mean'])} ± {fmt(s['power_w_sd'])} | "
            f"{fmt(s['duration_s_mean'])} ± {fmt(s['duration_s_sd'])} | "
            f"{fmt(s['tokens_mean'], '{:.1f}')} | "
            f"{fmt(s['energy_j_mean'])} ± {fmt(s['energy_j_sd'])} | "
            f"`{fmt(s['jpt_mean'], '{:.4f}')} J/t` | "
            f"{fmt(s['surprisal_mean'], '{:.3f}')} | {fmt(s['temp_end_c_mean'], '{:.1f}')} |"
        )
    lines.append("")

    lines.append("#### Per-Arm Detail")
    lines.append("")
    lines.append("| Prompt Arm | Valid Trials | Prompt Tokens | J/t CV | J/t 95% CI (approx.) | "
                 "Tokens/s | Raw $r$ (Surprise vs J/t) | Partial $r$ (given tokens) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for label in labels:
        s = arms[label]
        raw, part = s["raw_r_surprisal_jpt"], s["partial_r_surprisal_jpt_given_tokens"]
        lines.append(
            f"| **{label}** | {s['n_valid']} | {s['prompt_tokens']} | "
            f"{fmt(s['jpt_cv_pct'], '{:.1f}')}% | "
            f"[{fmt(s['jpt_ci95'][0], '{:.4f}')}, {fmt(s['jpt_ci95'][1], '{:.4f}')}] | "
            f"{fmt(s['tokens_per_second_mean'], '{:.1f}')} | "
            f"`{raw['r']:.4f}` ({_p_phrase(raw['p'])}) | "
            f"`{part['r']:.4f}` ({_p_phrase(part['p'])}) |"
        )
    lines.append("")

    lines.append("#### Hypothesis Testing")
    lines.append("")
    lines.append(f"* **Welch's t-test (J/token):** $t = {welch['t']:.4f}$, "
                 f"$\\mathrm{{dof}} = {welch['dof']:.2f}$, {_p_phrase(welch['p'])}")
    lines.append(f"* **Effect size (Cohen's $d$, pooled):** $d = {d:.4f}$; "
                 f"Hedges' $g = {payload['effect_size']['hedges_g_jpt']:.4f}$; "
                 f"Glass's $\\Delta = {payload['effect_size']['glass_delta_jpt']:.4f}$")
    lines.append(f"* **Welch's t-test (total energy, J):** $t = {payload['welch_t_energy']['t']:.4f}$, "
                 f"{_p_phrase(payload['welch_t_energy']['p'])}")
    lines.append(f"* **Welch's t-test (active power, W):** $t = {payload['welch_t_power']['t']:.4f}$, "
                 f"{_p_phrase(payload['welch_t_power']['p'])}")
    comp = payload["comparison"]
    tok_delta = comp["tokens_diff_mean"]
    tok_clause = ("essentially identical token counts" if abs(tok_delta) < 0.5
                  else f"{abs(tok_delta):.1f} {'fewer' if tok_delta > 0 else 'more'} tokens generated on average")
    lines.append(f"* **Direction:** {b_label} vs {a_label} — J/token "
                 f"{abs(comp['jpt_pct_diff']):.1f}% {'higher' if comp['jpt_pct_diff'] < 0 else 'lower'}, "
                 f"decode duration {abs(comp['duration_pct_diff']):.1f}% "
                 f"{'shorter' if comp['duration_pct_diff'] > 0 else 'longer'}, {tok_clause}.")
    if payload["significant_jpt"]:
        higher = b_label if comp["jpt_pct_diff"] < 0 else a_label
        lines.append(f"* **Result:** significant difference in J/token at "
                     f"$\\alpha = {WELCH_ALPHA}$; **{higher}** is less energy-efficient per token. "
                     "This is a difference between two specific prompt strings on this one device, "
                     "not a general law about prompt complexity.")
    else:
        lines.append(f"* **Result:** no significant difference in J/token at $\\alpha = {WELCH_ALPHA}$. "
                     "The observed difference is consistent with noise.")

    if payload.get("warnings"):
        lines.append("")
        lines.append("#### Warnings")
        lines.append("")
        for w in payload["warnings"]:
            lines.append(f"* ⚠️ {w}")

    lines.append("")
    lines.append("*Note: 95% CIs use a normal approximation; with n ≈ 20 per arm, prefer the "
                 "t-interval if exact coverage matters. Length control is limited to the token-count "
                 "range actually observed, reported above.*")
    return "\n".join(lines)


MIN_VALID_TOKENS_FOR_REPORT = 5


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------
def _self_test():
    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    print("analyze_results self-test")

    # Known t-distribution tail values (upper tail, one-sided)
    check("t_sf(2.086, 20) ~= 0.025", abs(student_t_sf(2.086, 20) - 0.025) < 1e-3,
          f"got {student_t_sf(2.086, 20):.5f}")
    check("t_sf(1.725, 20) ~= 0.05", abs(student_t_sf(1.725, 20) - 0.05) < 1e-3,
          f"got {student_t_sf(1.725, 20):.5f}")
    check("2*t_sf(2.086, 20) ~= 0.05 two-sided", abs(2 * student_t_sf(2.086, 20) - 0.05) < 2e-3)
    check("t_sf(1.96, 1e6) ~= 0.025", abs(student_t_sf(1.96, 1_000_000) - 0.025) < 1e-3)
    check("t_sf(0, 10) == 0.5", abs(student_t_sf(0.0, 10) - 0.5) < 1e-12)

    # Symmetry
    check("t_sf symmetric", abs(student_t_sf(-2.5, 9) - (1 - student_t_sf(2.5, 9))) < 1e-12)

    # Welch against a hand-checkable case
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 3.0, 4.0, 5.0, 6.0]
    r = welch_ttest(x, y)
    check("welch t == -1.0 for shifted samples", abs(r["t"] + 1.0) < 1e-9, f"got {r['t']:.6f}")
    check("welch dof ~= 8", abs(r["dof"] - 8.0) < 1e-6, f"got {r['dof']:.4f}")
    check("welch p ~= 0.3466", abs(r["p"] - 0.34659) < 1e-4, f"got {r['p']:.5f}")

    # Degenerate inputs must not raise
    check("welch single value -> p=1", welch_ttest([1.0], [2.0, 3.0])["p"] == 1.0)
    check("welch zero variance -> no crash", welch_ttest([1, 1, 1], [2, 2, 2])["note"] == "zero_variance")

    # p-value formatting must never print 0.0000
    check("p=0.0 formats as bound", format_p_value(0.0) == "< 0.0001")
    check("p=4e-20 formats as bound", format_p_value(4e-20) == "< 0.0001")
    check("p=0.000862 keeps magnitude", "e-04" in format_p_value(0.000862))
    check("p=0.0499 never rounds to 0.0500", format_p_value(0.0499) == "0.0499")
    check("p=0.5 literal", format_p_value(0.5) == "0.5000")

    # Effect sizes
    check("cohens_d sign", cohens_d_pooled([1, 2, 3], [4, 5, 6]) < 0)
    check("hedges_g shrinks d", abs(hedges_g([1, 2, 3], [4, 5, 6])) < abs(cohens_d_pooled([1, 2, 3], [4, 5, 6])))
    check("glass_delta zero-variance reference", glass_delta([1, 2], [3, 3]) == 0.0)

    # Correlations
    pr = pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    check("pearson perfect r == 1", abs(pr["r"] - 1.0) < 1e-12)
    check("pearson degenerate constant -> p=1", pearson([1, 1, 1, 1], [1, 2, 3, 4])["p"] == 1.0)
    check("partial_corr returns dict, not tuple", isinstance(partial_corr([1, 2, 3, 4, 5], [2, 3, 5, 4, 6], [1, 2, 3, 4, 5]), dict))
    check("partial_corr of covariate with itself -> flagged",
          partial_corr([1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4])["note"] == "covariate_explains_all_variance")

    # The published-report inconsistency must be caught by this module too.
    pub = welch_ttest(
        _synth(1.5760, 12.96 / 97.8, 20, seed=1),
        _synth(1.7095, 12.96 / 97.8, 20, seed=2),
    )
    check("published-style inputs give |t| < 9.13", abs(pub["t"]) < 9.13,
          f"t = {pub['t']:.4f} (2x SD cap reasoning)")

    # Cross-check against scipy when available
    try:
        from scipy import stats as scipy_stats  # noqa
        rng = np.random.default_rng(7)
        a = rng.normal(10, 1.0, 25)
        b = rng.normal(12, 2.5, 18)
        ours = welch_ttest(a, b)
        theirs = scipy_stats.ttest_ind(a, b, equal_var=False)
        check("welch t matches scipy", abs(ours["t"] - float(theirs.statistic)) < 1e-9,
              f"{ours['t']:.10f} vs {float(theirs.statistic):.10f}")
        check("welch p matches scipy", abs(ours["p"] - float(theirs.pvalue)) < 1e-12)
        check("welch dof matches scipy", abs(ours["dof"] - float(theirs.df)) < 1e-9)
        rng2 = np.random.default_rng(11)
        u, v = rng2.normal(0, 1, 30), rng2.normal(0, 1, 30)
        pr_ours = pearson(u, v)
        pr_scipy = scipy_stats.pearsonr(u, v)
        check("pearson r matches scipy", abs(pr_ours["r"] - float(pr_scipy[0])) < 1e-12)
        check("pearson p matches scipy", abs(pr_ours["p"] - float(pr_scipy[1])) < 1e-10,
              f"{pr_ours['p']:.3e} vs {float(pr_scipy[1]):.3e}")
    except ImportError:
        print("  [SKIP] scipy not importable; distribution values checked against tables only")

    print()
    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("SELF-TEST PASSED")
    return 0


def _synth(mean, sd, n, seed):
    rng = np.random.default_rng(seed)
    return list(rng.normal(mean, sd, n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Statistics for the LLM compute benchmark.")
    ap.add_argument("--self-test", action="store_true", help="run GPU-free validation checks")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(_self_test())
    ap.print_help()
