### ⚡ Rigorous Local LLM Compute & Entropy Analysis Report
* **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (Strictly On-Chip)
* **Model:** `Qwen/Qwen2.5-3B-Instruct` (4-bit Quantized)
* **Methodology:** Randomized trials (Seed 42), left-padded equalized prefill inputs, local dynamic idle baseline subtraction per trial, empirical surprise tracking, degenerate output guards ($N \ge 5$ tokens).

| Prompt Arm | Active Power (W) [Mean ± SD] | Duration (s) [Mean ± SD] | Tokens Gen [Mean] | Energy (J) [Mean ± SD] | Efficiency [Mean J/t] | Surprise [Mean Nats] | End Temp (°C) [Mean] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

#### Per-Arm Analysis & Partial Correlations (Surprise vs. J/token, controlling for token count):
| **Prompt A (Direct)** | 23.12 ± 0.85 W | 6.72 ± 0.26 s | 98.5 | 155.28 ± 6.02 J | `1.5760 J/t` | 0.305 | 60.7°C |
  * *Valid trials:* 20 | *Raw $r$ (Surprise vs J/t):* `-0.1013` ($p=0.671$) | *Partial $r$ (controlling for length):* `-0.1507` ($p=0.526$)
| **Prompt B (Complex/Stress)** | 28.09 ± 0.90 W | 5.95 ± 0.39 s | 97.8 | 167.26 ± 12.96 J | `1.7095 J/t` | 0.420 | 62.4°C |
  * *Valid trials:* 20 | *Raw $r$ (Surprise vs J/t):* `0.1318` ($p=0.580$) | *Partial $r$ (controlling for length):* `-0.0240` ($p=0.920$)

#### Hypothesis Testing & Statistical Summary:
* **Welch's t-test (J/token across arms):** $t = -9.1305$, $p = 0.0000$
* **Effect Size (Cohen's $d$):** $d = -2.8873$
* **Result:** Statistically significant difference detected ($p < 0.05$, $d=-2.89$), with **Prompt B (Complex/Stress)** exhibiting higher energy cost per token.