"""End-to-end smoke test of the analysis + report pipeline without a GPU.

Feeds synthetic per-trial data through analyze_results.analyze_arms / render_report and
asserts that the generated markdown is structurally valid and numerically consistent.
This is the regression guard for the formatting bugs that broke the original report.

Run:  python test_pipeline.py
"""

import re
import sys

import numpy as np

from analyze_results import analyze_arms, format_p_value, render_report, welch_ttest

LABELS = ["Prompt A (Direct)", "Prompt B (Complex/Stress)"]


def synthetic_results(n=20, seed=42):
    """Two arms with a real J/token difference and unequal variances."""
    rng = np.random.default_rng(seed)
    out = {}
    for label, jpt_mean, jpt_sd, tok_mean, power, dur in [
        (LABELS[0], 1.5760, 0.06, 98.5, 23.12, 6.72),
        (LABELS[1], 1.7095, 0.13, 97.8, 28.09, 5.95),
    ]:
        toks = list(rng.normal(tok_mean, 4.0, n).round().astype(int))
        jpt = list(rng.normal(jpt_mean, jpt_sd, n))
        out[label] = {
            "energies": [j * t for j, t in zip(jpt, toks)],
            "powers": list(rng.normal(power, 0.85, n)),
            "durations": list(rng.normal(dur, 0.30, n)),
            "tokens_gen": toks,
            "surprises": list(rng.normal(0.35, 0.08, n)),
            "temperatures": list(rng.normal(61.5, 1.5, n)),
            "jpt_list": jpt,
            "prompt_tokens": [100] * n,
            "idle_powers": [9.0] * n,
        }
    return out


def main():
    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    print("test_pipeline: analysis + markdown rendering")

    results = synthetic_results()
    payload = analyze_arms(results, LABELS)
    payload["run"] = {
        "seed": 42, "gpu": "Test GPU", "model": "test/model", "quantized": True,
        "max_new_tokens": 100, "target_trials_per_arm": 20, "attempts": 40,
        "equalize_prefill_tokens": False, "cooldown_s": 3.0, "idle_seconds": 1.5,
    }
    report = render_report(payload, LABELS)

    # 1. The results table must actually contain the arm rows.
    lines = report.splitlines()
    header_idx = next((i for i, l in enumerate(lines) if l.startswith("| Prompt Arm | Active Power")), None)
    check("results table header present", header_idx is not None)
    if header_idx is not None:
        separator = lines[header_idx + 1]
        check("table separator follows header", set(separator.replace("|", "").replace(":", "").strip()) <= {"-", " "})
        body = [l for l in lines[header_idx + 2:header_idx + 2 + len(LABELS)]]
        check("both arms are table rows", all(l.startswith("| **") for l in body), f"{len(body)} rows")
        check("no heading interrupts the table",
              not any(l.startswith("#") for l in lines[header_idx:header_idx + 2 + len(LABELS)]))
        n_cols = lines[header_idx].count("|")
        check("all rows have matching column counts",
              all(l.count("|") == n_cols for l in body), f"expected {n_cols} pipes")

    # 2. p-values must never render as an impossible exact zero
    quoted_p = re.findall(r"\$p = ([^$]+)\$", report)
    check("no p-value renders as exact zero",
          all(v.strip() != "0.0000" for v in quoted_p), str(quoted_p)[:120])
    check("format_p_value(0.0) is a bound", format_p_value(0.0) == "< 0.0001")

    # 3. Headline statistics must match an independent recomputation
    direct = welch_ttest(results[LABELS[0]]["jpt_list"], results[LABELS[1]]["jpt_list"])
    check("reported t matches recomputation", abs(payload["welch_t_jpt"]["t"] - direct["t"]) < 1e-12)
    check("t in markdown matches payload",
          f"$t = {direct['t']:.4f}$" in report, f"t={direct['t']:.4f}")

    # 4. Sample size must be described per arm, never as the total
    check("report states trials per arm", "valid per arm" in report)
    check("report does not claim 'N=40' per arm", "N=40" not in report and "N = 40" not in report)

    # 5. Unbalanced arms must be flagged (all per-trial lists truncated together)
    unbalanced = synthetic_results()
    for key in ("energies", "jpt_list", "tokens_gen", "surprises", "powers", "durations",
                "temperatures", "prompt_tokens", "idle_powers"):
        unbalanced[LABELS[0]][key] = unbalanced[LABELS[0]][key][:10]
    p2 = analyze_arms(unbalanced, LABELS)
    check("unbalanced arms raise a warning",
          any("Unbalanced" in w for w in p2["warnings"]), str(p2["warnings"])[:100])

    # 5b. Misaligned per-trial lists must be refused, not silently misaligned
    misaligned = synthetic_results()
    misaligned[LABELS[0]]["jpt_list"] = misaligned[LABELS[0]]["jpt_list"][:10]
    try:
        analyze_arms(misaligned, LABELS)
        check("misaligned lists raise ValueError", False, "no exception raised")
    except ValueError:
        check("misaligned lists raise ValueError", True)

    # 6. Unequal prompt lengths must be disclosed as an uncontrolled prefill cost
    unequal = synthetic_results()
    unequal[LABELS[1]]["prompt_tokens"] = [112] * 20
    p3 = analyze_arms(unequal, LABELS)
    check("prefill-length asymmetry is disclosed",
          any("prefill" in w for w in p3["warnings"]), str(p3["warnings"])[:140])

    # 7. Effect-size spread must be reported, not a lone pooled d
    check("multiple effect sizes reported",
          all(k in report for k in ["Cohen's $d$", "Hedges'", "Glass's"]))

    # 8. Degenerate input must not crash rendering
    empty = {label: {k: [] for k in v} for label, v in synthetic_results(n=0).items()}
    try:
        p4 = analyze_arms(empty, LABELS)
        render_report(p4, LABELS)
        check("empty arms render without crashing", True)
    except Exception as exc:  # noqa: BLE001
        check("empty arms render without crashing", False, f"{type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"PIPELINE TEST FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("PIPELINE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
