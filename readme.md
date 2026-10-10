# ⚡ Local LLM Compute & Entropy Analysis Suite

An empirical benchmarking tool designed to measure whether adversarial or complex prompt framing increases GPU power consumption and energy expenditure during local LLM inference. 

To eliminate common measurement artifacts—such as prefill amortization, thermal drift, and token-count variance—this suite implements rigorous controls including dynamic per-trial idle baselining, left-padded prefill equalization, and thread-polled power sampling.

---

## 🔬 Methodology & Controls

Running local power benchmarks on consumer hardware introduces severe noise. This script neutralizes those variables via:
* **Hardware & Quantization:** Targets `Qwen/Qwen2.5-3B-Instruct` running under **4-bit quantization** to fit strictly on-chip within a 4GB VRAM budget (tested on an NVIDIA GeForce RTX 3050 Laptop GPU)[cite: 1].
* **Prefill Equalization:** Left-pads input tokens across all arms to equalize prefill computation costs.
* **Dynamic Idle Baselines:** Captures a fresh local idle power sample immediately before every single trial to isolate active compute overhead from thermal drift.
* **Degenerate Output Guards:** Enforces minimum generation constraints ($N \ge 5$ tokens) to prevent short-circuit refusals from polluting the dataset.
* **Statistical Rigor:** Evaluates results using **Welch’s t-test**, **Cohen’s d** effect sizes, and **partial correlations** controlling for token count.

---

## 📊 Empirical Findings (Summary of Results)

Across **40 randomized trials** ($N = 20$ per arm, Seed 42), the benchmark revealed a statistically significant energy efficiency delta between direct prompts and complex/stress prompts[cite: 1]:

| Prompt Arm | Active Power (W) [Mean ± SD] | Duration (s) [Mean ± SD] | Tokens Gen [Mean] | Energy (J) [Mean ± SD] | Efficiency [Mean J/t] | End Temp (°C) [Mean] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prompt A (Direct)** | 23.12 ± 0.85 W | 6.72 ± 0.26 s | 98.5 | 155.28 ± 6.02 J | `1.5760 J/t` | 60.7°C |
| **Prompt B (Complex/Stress)** | 28.09 ± 0.90 W | 5.95 ± 0.39 s | 97.8 | 167.26 ± 12.96 J | `1.7095 J/t` | 62.4°C |

* **Welch's t-test (J/token):** $t = -9.1305, p = 0.0000$[cite: 1]
* **Effect Size:** $d = -2.8873$[cite: 1]
* **Conclusion:** Despite equalized input lengths, complex semantic prompt framing drives a statistically significant increase in active board power draw and Joules-per-token efficiency costs[cite: 1].

---

## 📂 Repository Structure

* `compute_opt_test.py`: The complete multi-trial benchmark script featuring thread-polled NVML power sampling and statistical analysis.
* `benchmark_report.md`: The full markdown report generated from the 40-trial execution[cite: 1].
* `generate_card_mobile.py`: Python script utilizing `matplotlib` to render a mobile-optimized portrait summary graphic for social sharing.
* `benchmark_card_mobile.jpg`: The pre-rendered visual asset displaying key metrics and statistical summaries.

---

## 🚀 Usage

1. Install dependencies:
   ```bash
   pip install torch transformers pynvml matplotlib scipy numpy