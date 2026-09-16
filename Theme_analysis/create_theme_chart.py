from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
out = ROOT / "output" / "theme_analysis_summary.png"

labels = [
    "AI infrastructure", "Stock/investor commentary", "AMD/competitors & deals",
    "China/export controls", "Market-wide coverage", "Jensen Huang/CEO",
    "Micron/HBM memory", "NVDA commentary/media"
]
up = np.array([25.834, 24.862, 12.035, 10.476, 5.333, 6.359, 7.197, 9.306])
down = np.array([25.070, 24.101, 11.174, 11.861, 5.091, 7.624, 6.102, 8.977])
keywords = [
    "AI, infrastructure, data center, cloud",
    "stock, buy, investors, Wall Street",
    "AMD, OpenAI, Meta, deal, Broadcom",
    "China, H200, chips, Trump, export",
    "Dow, stock market, futures, index",
    "Jensen Huang, CEO, Nvidia CEO",
    "Micron, memory, HBM, bandwidth, supply",
    "Jim Cramer, Nasdaq, stocks, NVDA",
]

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(14, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1])

ax = fig.add_subplot(gs[0])
y = np.arange(len(labels))
h = 0.36
ax.barh(y + h/2, up, h, label="Up events", color="#2563eb")
ax.barh(y - h/2, down, h, label="Down events", color="#dc2626")
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel("Mean share of matched articles (%)")
ax.set_title("NVDA News Themes Around Price Events", loc="left", fontsize=17, weight="bold")
ax.legend(frameon=False, ncols=2, loc="lower right")
for i, (u, d) in enumerate(zip(up, down)):
    ax.text(max(u, d) + 0.35, i, f"{u:.1f} / {d:.1f}", va="center", fontsize=9, color="#374151")
ax.text(0, -1.05, "Labels show Up / Down percentages", fontsize=10, color="#6b7280")
ax.set_xlim(0, 31)

ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
ax2.set_title("Keywords defining each discovered topic", loc="left", fontsize=14, weight="bold", pad=10)
table = ax2.table(cellText=[[labels[i], keywords[i]] for i in range(len(labels))],
                  colLabels=["Interpretable label", "Top words / phrases"],
                  cellLoc="left", colLoc="left", loc="center", colWidths=[0.27, 0.67])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.65)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#d1d5db")
    if r == 0:
        cell.set_facecolor("#e5e7eb")
        cell.set_text_props(weight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f9fafb")

fig.savefig(out, dpi=200, bbox_inches="tight")
print(out)
