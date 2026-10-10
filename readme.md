# ⚡ Local LLM Compute Analysis Suite

An empirical benchmarking harness that measures whole-board GPU power during local LLM inference and
compares **energy per generated token** between two prompt arms: a direct request and an
adversarial / refusal-eliciting one.

The interesting question is narrow and testable: *if you hold the model, the device, and the token budget
fixed, does swapping the prompt string change how much energy each generated token costs?* This suite
answers that for two specific prompts on one machine — and is explicit about everything it does not answer.

---

## 🔬 Methodology & Controls

Power benchmarking on consumer laptop hardware is dominated by noise. The harness neutralizes what it can
and **discloses what it cannot**:

### Controls that are implemented

* **Fixed device & model:** `Qwen/Qwen2.5-3B-Instruct`, loaded 4-bit NF4 (`bnb_4bit_compute_dtype=float16`)
  when free VRAM is below 7 GB, pinned to GPU 0 with `device_map={"": 0}` so all compute stays on-chip.
* **Randomized arm order:** trial labels are shuffled with a fixed seed, so arm order is not systematically
  confounded with warm-up or drift.
* **Per-trial local idle baseline:** a fresh idle power average is sampled immediately before *every* trial
  and subtracted from that trial's decode-phase power, so slow thermal drift is not read as compute cost.
* **Pre-trial thermal settling:** a configurable cooldown runs *before* each idle baseline, so the baseline
  and the trial that follows it describe the same thermal state.
* **Unmeasured warm-up:** several full generations run before the trial loop to settle clocks, autotune
  kernels, and grow the allocator, so trial 1 is not structurally different from trial 40.
* **Decode-window power sampling:** NVML board power is sampled on a background thread that starts *after*
  `torch.cuda.synchronize()`, so queued warm-up or prefill work cannot leak into the average.
* **Attention-masked left padding:** padded positions are masked out of attention, so pad tokens do not
  become attended content.
* **Balanced arms under discarding:** trials with degenerate short outputs ($N < 5$ tokens) are discarded and
  the arm is redrawn, so a guard cannot silently unbalance the two arms.
* **Equalized prefill (`--equalize-prefill-tokens`, opt-in):** prepends *real, unmasked* tokens to the shorter
  arm so both arms have identical attended sequence length.
* **Raw data retention:** every per-trial row is written to `benchmark_trials_<seed>.csv` and every statistic
  to `benchmark_stats_<seed>.json`, so published numbers can be recomputed from raw data.
* **Statistics:** Welch's unequal-variance t-test (scipy-free implementation cross-checked against scipy),
  Cohen's $d$ reported **alongside** Hedges' $g$ and Glass's $\Delta$ because arm variances differ, and
  Pearson/partial correlations with correct $p$-values. $p$-values are never printed as `0.0000`.

### What these controls do *not* do

* **Masked padding does not equalize prefill work.** It equalizes tensor *shape* only; masked positions are
  excluded from attention, so attended length — and therefore real prefill cost — still tracks the true
  prompt length. Use `--equalize-prefill-tokens` for genuine length matching, and read the token-length
  delta the harness prints.
* **No token-count control across arms.** `max_new_tokens` caps output length but EOS can end a generation
  earlier, and generation is sampled, so arms can generate different token counts. The report prints each
  arm's token range; a length confound is disclosed rather than hidden.
* **No entropy measurement.** "Surprise" here is mean per-token negative log-likelihood in nats under the
  model's own distribution. It is not output entropy, predictive entropy, or semantic entropy.
* **One device, one model, two prompts.** No repetition across seeds, models, or hardware, and no correction
  for multiple comparisons across the power / energy / J-per-token / temperature family.
* **Board power is not per-SM power.** NVML reports whole-board draw including host and PCIe overhead, and
  it is polled from Python against NVML's own update rate.

---

## 📊 Published Results (run 42)

40 trials total, **20 valid trials per arm**, seed 42.

| Prompt Arm | Active Power (W) [Mean ± SD] | Decode Duration (s) [Mean ± SD] | Tokens Gen [Mean] | Total Energy (J) [Mean ± SD] | Efficiency [Mean J/t] | Surprise [Mean Nats] | End Temp (°C) [Mean] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prompt A (Direct)** | 23.12 ± 0.85 | 6.72 ± 0.26 | 98.5 | 155.28 ± 6.02 | 1.5760 | 0.305 | 60.7 |
| **Prompt B (Complex/Stress)** | 28.09 ± 0.90 | 5.95 ± 0.39 | 97.8 | 167.26 ± 12.96 | 1.7095 | 0.420 | 62.4 |

**Defensible reading:** for these two exact prompt strings on this one GPU, the adversarial prompt averaged
~8.5% more energy per generated token over a ~11% shorter generation — **higher power, shorter runtime,
fewer tokens**. That is a tradeoff in efficiency per token, not a monotone increase in energy use.

⚠️ **The published `t = -9.1305` is not reproducible from these means and SDs.** The Welch t-test recomputed
from the table gives ≈ −4.09 on J/token (≈ −3.75 on total energy), and reaching $|t| = 9.13$ would require
per-arm standard errors roughly half of what the published energy SDs imply. Run `python verify_stats.py`
for the audit; the headline statistics are therefore listed as **unverified** in
[`benchmark_report.md`](benchmark_report.md#1-the-published-t-statistic-is-not-consistent-with-the-published-means-and-sds)
until regenerated from raw per-trial data.

Also note the arms differ in **content**, not just framing: Prompt B is a policy-violating request, so its
natural response is a safety refusal. Any difference is confounded with refusal behavior, output style, and
prompt length — all of which changed at the same time as the variable of interest.

---

## 📂 Repository Structure

* `compute_opt_test.py` — the benchmark harness (NVML power sampling, trial loop, report generation).
* `analyze_results.py` — statistics and markdown report rendering; `--self-test` validates the whole
  statistical layer with no GPU required.
* `test_pipeline.py` — regression test for the analysis and report pipeline (synthetic data, no GPU).
* `verify_stats.py` — read-only audit of the summary statistics published in the report.
* `benchmark_report.md` — the published run-42 report, including a corrections and limitations section.
* `card_data.py` — single source of truth for the numbers and caveats printed on the share cards.
* `generate_card.py` / `generate_card_mobile.py` — render the share cards from the published snapshot.
* `benchmark_card.jpg` / `benchmark_card_mobile.jpg` — the rendered cards (frozen to the published snapshot).

---

## 🚀 Usage

```bash
pip install torch transformers pynvml numpy scipy matplotlib

# Full benchmark: 20 valid trials per arm, writes the report plus raw CSV/JSON
python compute_opt_test.py

# Smaller/faster run with a stricter prefill control
python compute_opt_test.py --trials 5 --cooldown 1.0 --equalize-prefill-tokens

# Validate the statistics layer without a GPU
python analyze_results.py --self-test
python test_pipeline.py

# Audit the published summary statistics (numpy + scipy only)
python verify_stats.py

# Re-render the share cards from the published snapshot
python generate_card.py
python generate_card_mobile.py
```

Outputs: `benchmark_report.md`, `benchmark_trials_<seed>.csv` (audit trail), `benchmark_stats_<seed>.json`
(machine-readable statistics).

> Card images are generated assets. If you re-run the benchmark, update `card_data.py` from the new
> report and re-run both card scripts — they quote the frozen run-42 snapshot by design.
