import matplotlib.pyplot as plt

# --- Data from benchmark_report.md ---
hardware = "NVIDIA GeForce RTX 3050 Laptop GPU (4-bit Q)"
model = "Qwen/Qwen2.5-3B-Instruct"

headers = ["Prompt Arm", "Active Power", "Tokens", "Energy (J)", "Efficiency", "Surprise"]
rows = [
    ["Prompt A (Direct)", "23.12 W", "98.5", "155.28 J", "1.5760 J/t", "0.305 nats"],
    ["Prompt B (Stress)", "28.09 W", "97.8", "167.26 J", "1.7095 J/t", "0.420 nats"]
]

stats_text = (
    "STATISTICAL SUMMARY (N=40 Trials)\n"
    "• Welch's t-test: t = -9.13 (p = 0.0000)\n"
    "• Effect Size: Cohen's d = -2.89\n"
    "• Finding: Complex/adversarial prompts exhibit statistically\n"
    "  significant higher energy consumption per token."
)

# --- Set up X (Twitter) optimized figure size ---
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
fig.patch.set_facecolor('#0b0f19')  # Deep dark X-style background
ax.set_facecolor('#0b0f19')
ax.axis('off')

# Title & Metadata
ax.text(0.05, 0.92, "⚡ Local LLM Compute & Entropy Benchmark", fontsize=18, fontweight='bold', color='#ffffff', fontfamily='sans-serif')
ax.text(0.05, 0.86, f"Hardware: {hardware} | Model: {model}", fontsize=11, color='#8a99ad', fontfamily='sans-serif')

# Table Rendering
table = ax.table(
    cellText=rows,
    colLabels=headers,
    loc='center',
    cellLoc='center',
    bbox=[0.05, 0.38, 0.90, 0.35]
)

table.auto_set_font_size(False)
table.set_fontsize(10)

# Style Table Cells
for (row_idx, col_idx), cell in table.get_celld().items():
    cell.set_edgecolor('#1f293d')
    if row_idx == 0:
        cell.set_facecolor('#1e293b')
        cell.set_text_props(weight='bold', color='#38bdf8')
    else:
      cell.set_facecolor('#0f172a')
      cell.set_text_props(color='#e2e8f0')

# Stats Box at the Bottom
props = dict(boxstyle='round,pad=0.8', facecolor='#111827', edgecolor='#374151', alpha=0.9)
ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', bbox=props, color='#34d399', fontfamily='monospace', fontweight='bold')

plt.tight_layout()
plt.savefig("benchmark_card.jpg", format='jpeg', bbox_inches='tight', facecolor=fig.get_facecolor())
print("✨ Gorgeous benchmark card saved successfully as 'benchmark_card.jpg'!")