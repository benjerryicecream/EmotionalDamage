"""Independent audit of the published summary statistics in benchmark_report.md.

Read-only: this script recomputes the headline statistics from the arm summaries
that appear in the report and checks them for internal consistency. It requires
only numpy + scipy and never touches the GPU, the model, or the report itself.

Run:  python verify_stats.py
"""

import numpy as np
from scipy import stats

N = 20  # trials per arm (report's "N = 40" is the total across both arms)

# Mean/SD pairs exactly as published in benchmark_report.md
PUBLISHED = {
    "A": {"power": (23.12, 0.85), "duration": (6.72, 0.26), "tokens": 98.5,
          "energy": (155.28, 6.02), "jpt": 1.5760},
    "B": {"power": (28.09, 0.90), "duration": (5.95, 0.39), "tokens": 97.8,
          "energy": (167.26, 12.96), "jpt": 1.7095},
}
PUBLISHED_T = -9.1305
PUBLISHED_D = -2.8873


def welch(m1, s1, m2, s2, n1=N, n2=N):
    """Welch's t-test from summary statistics; returns t, dof, two-sided p."""
    se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    if se == 0:
        return 0.0, 0.0, 1.0
    t = (m1 - m2) / se
    dof = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2 / (
        (s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1)
    )
    return t, dof, 2 * stats.t.sf(abs(t), dof)


def pooled_d(m1, s1, m2, s2):
    sp = np.sqrt(((N - 1) * s1 ** 2 + (N - 1) * s2 ** 2) / (2 * N - 2))
    return (m1 - m2) / sp


def main():
    a, b = PUBLISHED["A"], PUBLISHED["B"]
    print(f"Published headline: t = {PUBLISHED_T}, Cohen's d = {PUBLISHED_D}\n")

    checks = [
        ("Active power (W)", *a["power"], *b["power"]),
        ("Total energy (J)", *a["energy"], *b["energy"]),
    ]
    for name, m1, s1, m2, s2 in checks:
        t, dof, p = welch(m1, s1, m2, s2)
        print(f"[{name}] Welch t = {t:.4f} (dof {dof:.2f}), p = {p:.3g}, "
              f"pooled d = {pooled_d(m1, s1, m2, s2):.4f}")

    # J/token SDs implied by the published per-arm energy SDs
    sa = a["energy"][1] / a["tokens"]
    sb = b["energy"][1] / b["tokens"]
    t, dof, p = welch(a["jpt"], sa, b["jpt"], sb)
    print(f"[J/token, SD implied by energy SD] sd_A = {sa:.4f}, sd_B = {sb:.4f}")
    print(f"    Welch t = {t:.4f} (dof {dof:.2f}), p = {p:.3g}, "
          f"pooled d = {pooled_d(a['jpt'], sa, b['jpt'], sb):.4f}")

    # Reverse-engineer the standard errors the published t requires
    diff = a["jpt"] - b["jpt"]
    se_req = diff / PUBLISHED_T
    bound = se_req * np.sqrt(N)
    print(f"\nPublished t = {PUBLISHED_T} requires SE = {se_req:.5f}")
    print(f"  => sd_A^2 + sd_B^2 = {bound ** 2:.6f}, so BOTH J/token SDs are "
          f"capped at {bound:.4f}")
    print(f"  => but the published energy SD (12.96 J / {b['tokens']} tokens) "
          f"implies sd_B = {sb:.4f}, which is {sb / bound:.1f}x that cap.")

    implied_t = diff / np.sqrt(sa ** 2 / N + sb ** 2 / N)
    print(f"\nCONCLUSION: the published t = {PUBLISHED_T} is inconsistent with the "
          f"published means/SDs.")
    print(f"  Total-energy SDs and J/token SDs imply t ~ {implied_t:.2f}, not {PUBLISHED_T}.")
    print("  The p-value cannot be recomputed from the printed table; it must be "
          "regenerated from raw per-trial data.")


if __name__ == "__main__":
    main()
