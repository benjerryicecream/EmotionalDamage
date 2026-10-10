"""Render the mobile (portrait 9:16) share card from the frozen run-42 snapshot.

Data lives in card_data.py. Row labels are pre-wrapped in card_data.ROWS_MOBILE because
matplotlib tables do not wrap text and the arm labels previously overflowed their cells.

    python generate_card_mobile.py
"""

import matplotlib.pyplot as plt

import card_data as data

fig, ax = plt.subplots(figsize=(8, 10), dpi=300)
fig.patch.set_facecolor("#0b0f19")
ax.set_facecolor("#0b0f19")
ax.axis("off")

title = data.TITLE_MOBILE
ax.text(0.08, 0.94, title, fontsize=20, fontweight="bold", color="#ffffff",
        fontfamily="sans-serif", va="top")
ax.text(0.08, 0.83, f"Hardware: {data.HARDWARE}\nModel: {data.MODEL}", fontsize=11,
        color="#8a99ad", fontfamily="sans-serif", va="top")

# bbox top lowered to 0.78 so a 3-line arm label cannot collide with the header
table = ax.table(
    cellText=data.ROWS_MOBILE,
    colLabels=data.HEADERS,
    loc="center",
    cellLoc="center",
    bbox=[0.05, 0.36, 0.90, 0.42],
)
table.auto_set_font_size(False)
table.set_fontsize(9)

for (row_idx, _col_idx), cell in table.get_celld().items():
    cell.set_edgecolor("#1f293d")
    if row_idx == 0:
        cell.set_facecolor("#1e293b")
        cell.set_text_props(weight="bold", color="#38bdf8")
    else:
        cell.set_facecolor("#0f172a")
        cell.set_text_props(color="#e2e8f0")

props = dict(boxstyle="round,pad=1.0", facecolor="#111827", edgecolor="#374151", alpha=0.9)
ax.text(0.08, 0.30, data.STATS_TEXT_MOBILE, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=props, color="#34d399",
        fontfamily="monospace", fontweight="bold")

plt.tight_layout()
plt.savefig("benchmark_card_mobile.jpg", format="jpeg", bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print("✨ Mobile-optimized benchmark card saved as 'benchmark_card_mobile.jpg'")
