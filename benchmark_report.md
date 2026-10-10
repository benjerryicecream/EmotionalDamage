### ⚡ Local LLM Compute Analysis Report

* **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU
* **Model:** `Qwen/Qwen2.5-3B-Instruct` (4-bit quantized)
* **Trials:** 20 valid per arm (40 total), seed 42, 40 attempts
* **Decoding:** `max_new_tokens=100`, `temperature=0.7`, sampling enabled
* **Controls:** attention-masked left padding; per-trial local idle baseline (1.5 s) subtracted from the decode-phase power average; 3.0 s pre-trial cooldown; degenerate-output guard ($N \ge 5$ tokens) with redraw to keep arms balanced
* **Surprise:** mean negative log-likelihood of sampled tokens, in nats (per-token surprisal, *not* output entropy)

| Prompt Arm | Active Power (W) [Mean ± SD] | Decode Duration (s) [Mean ± SD] | Tokens Gen [Mean] | Total Energy (J) [Mean ± SD] | Efficiency [Mean J/t] | Surprise [Mean Nats] | End Temp (°C) [Mean] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prompt A (Direct)** | 22.78 ± 1.57 | 6.67 ± 0.41 | 98.5 | 151.51 ± 7.23 | `1.5381 J/t` | 0.305 | 59.6 |
| **Prompt B (Complex/Stress)** | 25.76 ± 1.19 | 6.00 ± 0.49 | 97.8 | 154.53 ± 13.73 | `1.5782 J/t` | 0.420 | 61.0 |

#### Per-Arm Detail

| Prompt Arm | Valid Trials | Prompt Tokens | J/t CV | J/t 95% CI (approx.) | Tokens/s | Raw $r$ (Surprise vs J/t) | Partial $r$ (given tokens) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prompt A (Direct)** | 20 | 47 | 4.6% | [1.5072, 1.5689] | 14.8 | `-0.1029` ($p = 0.6658$) | `-0.1667` ($p = 0.4952$) |
| **Prompt B (Complex/Stress)** | 20 | 58 | 3.7% | [1.5524, 1.6040] | 16.3 | `0.2169` ($p = 0.3584$) | `-0.1053` ($p = 0.6679$) |

#### Hypothesis Testing

* **Welch's t-test (J/token):** $t = -1.9544$, $\mathrm{dof} = 36.85$, $p = 0.0583$
* **Effect size (Cohen's $d$, pooled):** $d = -0.6180$; Hedges' $g = -0.6058$; Glass's $\Delta = -0.6811$
* **Welch's t-test (total energy, J):** $t = -0.8705$, $p = 0.3912$
* **Welch's t-test (active power, W):** $t = -6.7548$, $p$ < 0.0001
* **Direction:** Prompt B (Complex/Stress) vs Prompt A (Direct) — J/token 2.5% higher, decode duration 11.1% shorter, 0.8 fewer tokens generated on average.
* **Result:** no significant difference in J/token at $\alpha = 0.05$. The observed difference is consistent with noise.

#### Warnings

* ⚠️ Attended prompt lengths differ by 11 tokens; masked padding does not equalize prefill work. Prefill cost is not controlled in this run.
* ⚠️ Active power differs but total energy does not — consistent with the arms producing different numbers of tokens over different runtimes.

*Note: 95% CIs use a normal approximation; with n ≈ 20 per arm, prefer the t-interval if exact coverage matters. Length control is limited to the token-count range actually observed, reported above.*