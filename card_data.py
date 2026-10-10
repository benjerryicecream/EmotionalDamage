"""Shared data for the share-card renderers.

Single source of truth for the numbers and caveats printed on benchmark_card.jpg and
benchmark_card_mobile.jpg, so the two cards cannot drift apart.

These values are the FROZEN run-42 snapshot published in benchmark_report.md. If the
benchmark is re-run, update this file from the new report and re-render both cards.
"""

HARDWARE = "NVIDIA GeForce RTX 3050 Laptop GPU (4-bit Q)"
MODEL = "Qwen/Qwen2.5-3B-Instruct"

HEADERS = ["Prompt Arm", "Active Power", "Tokens Gen", "Energy (J)", "Efficiency", "Surprise"]

# Compact rows for the wide card
ROWS_WIDE = [
    ["Prompt A (Direct)", "23.12 W", "98.5", "155.28 J", "1.5760 J/t", "0.305 nats"],
    ["Prompt B (Stress)", "28.09 W", "97.8", "167.26 J", "1.7095 J/t", "0.420 nats"],
]

# Mobile rows: pre-wrapped so matplotlib never clips an over-long arm label.
ROWS_MOBILE = [
    ["Prompt A\n(Direct)", "23.12 W", "98.5", "155.28 J", "1.5760 J/t", "0.305"],
    ["Prompt B\n(Complex/\nStress)", "28.09 W", "97.8", "167.26 J", "1.7095 J/t", "0.420"],
]

# "20 trials/arm" — the published "N=40" is 40 trials TOTAL, not 40 per arm.
SAMPLE_NOTE = "20 trials per arm (40 total)"
PUBLISHED_T = -9.1305
PUBLISHED_D = -2.8873

STATS_TEXT_WIDE = (
    f"STATISTICAL SUMMARY ({SAMPLE_NOTE})\n"
    f"• Welch's t-test (J/token): t = {PUBLISHED_T:.2f}, p < 0.0001\n"
    f"• Effect size: Cohen's d = {PUBLISHED_D:.2f} (pooled; arm variances differ)\n"
    "• Finding: adversarial arm costs ~8.5% more energy per token,\n"
    "  over a ~11% shorter generation (higher power, less runtime).\n"
    "• Caveat: t is NOT reproducible from the published means/SDs —\n"
    "  see benchmark_report.md §1; single GPU, single model, 2 prompts."
)

STATS_TEXT_MOBILE = (
    f"STATISTICAL SUMMARY ({SAMPLE_NOTE})\n"
    f"• Welch's t-test (J/token):\n"
    f"    t = {PUBLISHED_T:.2f}, p < 0.0001\n"
    f"• Effect size: Cohen's d = {PUBLISHED_D:.2f}\n"
    "• Finding: adversarial arm costs ~8.5%\n"
    "  more energy per token, over a ~11%\n"
    "  shorter generation.\n"
    "• Caveat: t is NOT reproducible from the\n"
    "  published means/SDs (report §1).\n"
    "  Single GPU / model / 2 prompts."
)

# Titles deliberately avoid the ⚡ emoji: matplotlib's bundled DejaVu Sans reports cmap
# coverage for U+26A1 but renders it as a thin, clipped stroke, so glyph-coverage checks
# give false positives and the bolt looks broken at card resolution.
TITLE_WIDE = "Local LLM Compute & Energy Benchmark"
TITLE_MOBILE = "Local LLM Compute &\n    Energy Analysis"
