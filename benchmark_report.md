# ⚡ Rigorous Local LLM Compute & Entropy Analysis Report

* **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (strictly on-chip)
* **Model:** `Qwen/Qwen2.5-3B-Instruct` (4-bit NF4 quantized, `bnb_4bit_compute_dtype=torch.float16`)
* **Trials:** 40 total — **20 valid trials per arm** (2 arms), randomized order, seed 42
* **Decoding:** stochastic sampling, `temperature=0.7`, `max_new_tokens=100`
* **Controls:** attention-masked left padding to a common input width, per-trial local idle baseline subtraction, degenerate-output guard ($N \ge 5$ tokens)
* **Surprise:** mean negative log-likelihood of the sampled tokens, in nats

> **Scope.** This is a single-SKU, single-machine comparison of two specific prompt strings. It is not a
> general test of "prompt complexity," and it does not measure entropy. See
> [Corrections & Limitations](#-corrections--limitations) before citing any number below.

---

## 📊 Results (as published in run 42)

| Prompt Arm | Active Power (W) [Mean ± SD] | Decode Duration (s) [Mean ± SD] | Tokens Gen [Mean] | Total Energy (J) [Mean ± SD] | Efficiency [Mean J/t] | Surprise [Mean Nats] | End Temp (°C) [Mean] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prompt A (Direct)** | 23.12 ± 0.85 | 6.72 ± 0.26 | 98.5 | 155.28 ± 6.02 | 1.5760 | 0.305 | 60.7 |
| **Prompt B (Complex/Stress)** | 28.09 ± 0.90 | 5.95 ± 0.39 | 97.8 | 167.26 ± 12.96 | 1.7095 | 0.420 | 62.4 |

### Within-arm correlations (Surprise vs. J/token)

| Prompt Arm | Valid trials | Raw $r$ | $p$ | Partial $r$ (controlling for token count) | $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Prompt A (Direct)** | 20 | −0.1013 | 0.671 | −0.1507 | 0.526 |
| **Prompt B (Complex/Stress)** | 20 | 0.1318 | 0.580 | −0.0240 | 0.920 |

Neither arm shows a significant association between token surprise and per-token energy, and the partial
correlations are uninformative: token count varies by less than one token on average within each arm, so
"controlling for token count" removes almost no variance.

---

## 🔧 Corrections & Limitations

### 1. The published $t$ statistic is not consistent with the published means and SDs

The published `t = -9.1305, p = 0.0000` **cannot be reproduced from the table above**. With $n = 20$ per arm:

| Quantity recomputed from the published means/SDs | Welch $t$ | dof | $p$ |
| :--- | :---: | :---: | :---: |
| Active power (23.12 ± 0.85 vs 28.09 ± 0.90 W) | −17.95 | 37.9 | 4.1e−20 |
| Total energy (155.28 ± 6.02 vs 167.26 ± 12.96 J) | −3.75 | 26.8 | 8.6e−4 |
| J/token, SDs implied by the energy SDs (0.0611 vs 0.1325) | −4.09 | 26.7 | 3.5e−4 |

Reaching $|t| = 9.1305$ on a 0.1335 J/t difference requires a standard error of 0.01462, i.e.
$\mathrm{sd}_A^2 + \mathrm{sd}_B^2 = 0.004276$. The published per-arm energy SD of 12.96 J over 97.8 tokens
implies $\mathrm{sd}_B = 0.1325$ on its own — **twice the cap** that $t = 9.1305$ permits. So the published
$t$ (and therefore the published $p$ and $d$) is internally inconsistent with the published effect sizes at
roughly the 4σ level.

*Reproduce with* `python verify_stats.py` *(read-only; requires numpy + scipy only).*

**Status: unresolved.** Anchoring the headline claim requires regenerating the statistics from raw per-trial
data, which was not preserved in this repository. Treat `t = -9.1305`, `p = 0.0000`, and `d = -2.8873` as
**unverified** until then. The defensible reading of the table is a total-energy difference of
$t \approx -3.75$, $p \approx 8.6\times10^{-4}$, $d \approx -1.19$ — still significant, but half the
advertised effect.

### 2. "N = 40" was reported as 40 trials per arm

The run executed 40 trials **in total**: 20 per arm. The previous version of this report headlined
"N = 40 Trials" and labeled each arm's row "20", which reads as 40 per arm. All $n = 20$ figures above are
per arm.

### 3. The markdown table was structurally broken

The results header row was immediately followed by a `####` heading, and each arm's metrics were emitted as
free-standing lines outside any table body — the arms were not table rows at all. The table rendered empty
in most markdown viewers. Fixed above by emitting the arm rows into the table body and moving the
correlation statistics into their own table.

### 4. $p = 0.0000$ is not a reportable value

Floating-point formatting produced `0.0000`, implying an impossible exact zero. $p$-values must be reported
as `p < 0.0001`. The generator now formats this correctly.

### 5. The conclusion overreached and contradicted its own framing

The published conclusion was *"complex semantic prompt framing drives a statistically significant increase
in active board power draw,"* and the README added *"and Joules-per-token efficiency costs."* Three problems:

* **The effect is on J/token only.** Prompt B generated **fewer** tokens and finished **faster**, so it
  consumed **less time at higher power**: A/B total-power and duration move in opposite directions. A
  result that is "higher power, shorter runtime, fewer tokens" is a tradeoff, not a monotone increase.
* **"Active board power draw" is not a validated measurement.** The metric is whole-board NVML power with a
  single local idle subtraction. It includes host/PCIe overhead, and it is sampled from Python at 15 ms
  intervals against NVML's own update rate.
* **The comparison cannot isolate "semantic complexity."** The two arms differ in *content*, not just in
  framing: Prompt B is a policy-violating request, so the natural response is a safety refusal. Any observed
  difference is confounded by refusal behavior, output style, and prompt token count — all of which changed
  at the same time as the variable of interest.

The supportable claim is narrower: *for these two exact prompt strings, on this one GPU, the
adversarial/refusal-eliciting prompt averaged ~8.5% more energy per generated token, over a ~11% shorter
generation.*

### 6. "Prefill equalization" did not equalize prefill work

The two arms' chat-templated prompts differ in token length. Attention-masked padding pads the *tensor* to a
common width, but masked pad positions are excluded from attention, so the real prefill and decode work still
tracks the true sequence length. Padding therefore did not make prefill cost equal across arms; it only made
the input shapes equal. The generator now measures true per-arm prompt lengths, logs the delta, and offers
`--equalize-prefill-tokens` to pad the shorter arm with real, unmasked neutral tokens so the attended
sequence length is genuinely identical.

### 7. "Entropy analysis" is mean surprisal

Mean negative log-likelihood in nats is a per-token surprisal measure under the model's own distribution. It
is not the model's output entropy, not predictive entropy, and not semantic entropy. No entropy or
semantic-entropy quantity is reported anywhere in this suite.

### 8. No causal energy conclusion is supported

Total-energy and power differences here also carry:

* **Device scope:** one 6 GB laptop GPU, one quantization recipe, one model, two prompts. No repetition
  across seeds, models, or hardware.
* **Thermal ordering:** trials are randomized and a 3 s cooldown precedes each trial's baseline, but the
  loop applies no post-trial cooldown, so the final trial runs on the hottest device state with no settling.
* **Single idle baseline per trial:** one 1.5 s local average, subtracted arithmetically from a decode-phase
  average, with no uncertainty propagated through the subtraction.
* **n = 20 per arm:** adequate for a large effect, underpowered for anything small, and not corrected for
  multiple comparisons across the power/energy/J-per-token/temperature family.

---

## 📂 Artifacts

* `compute_opt_test.py` — benchmark harness. Regenerates this report from raw per-trial data and writes
  `benchmark_trials_<seed>.csv` (per-trial rows) plus `benchmark_stats_<seed>.json` (all test statistics).
* `verify_stats.py` — read-only audit of the summary statistics in this report.
* `generate_card.py`, `generate_card_mobile.py` — render the social cards. Both quote the *published*
  snapshot above and must be re-run after any regeneration.
* `benchmark_trials_<seed>.csv` — **required** for any claim in this report to be independently checkable.
  It was not preserved for run 42; the inconsistency in §1 cannot be resolved without it.
