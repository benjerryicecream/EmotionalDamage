"""Render the wide (X/Twitter-optimized) share card from the frozen run-42 snapshot.

Data lives in card_data.py. Re-run this after any benchmark re-run.

    python generate_card.py
"""

import matplotlib.pyplot as plt

import card_data as data

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
fig.patch.set_facecolor("#0b0f19")
ax.set_facecolor("#0b0f19")
ax.axis("off")

title = data.TITLE_WIDE
ax.text(0.05, 0.92, title, fontsize=18, fontweight="bold",
        color="#ffffff", fontfamily="sans-serif")
ax.text(0.05, 0.86, f"Hardware: {data.HARDWARE} | Model: {data.MODEL}", fontsize=11,
        color="#8a99ad", fontfamily="sans-serif")

table = ax.table(
    cellText=data.ROWS_WIDE,
    colLabels=data.HEADERS,
    loc="center",
    cellLoc="center",
    bbox=[0.05, 0.38, 0.90, 0.35],
)
table.auto_set_font_size(False)
table.set_fontsize(10)

for (row_idx, _col_idx), cell in table.get_celld().items():
    cell.set_edgecolor("#1f293d")
    if row_idx == 0:
        cell.set_facecolor("#1e293b")
        cell.set_text_props(weight="bold", color="#38bdf8")
    else:
        cell.set_facecolor("#0f172a")
        cell.set_text_props(color="#e2e8f0")

props = dict(boxstyle="round,pad=0.8", facecolor="#111827", edgecolor="#374151", alpha=0.9)
ax.text(0.05, 0.05, data.STATS_TEXT_WIDE, transform=ax.transAxes, fontsize=10,
        verticalalignment="bottom", bbox=props, color="#34d399",
        fontfamily="monospace", fontweight="bold")

plt.tight_layout()
plt.savefig("benchmark_card.jpg", format="jpeg", bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print("✨ Benchmark card saved as 'benchmark_card.jpg'")
