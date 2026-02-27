"""
Business-insight chart generator for supertoys.az product data.
Run: python scripts/generate_charts.py
Requires: pip install matplotlib pandas
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
DATA_FILE   = Path(__file__).parent.parent / "data" / "products.csv"
CHARTS_DIR  = Path(__file__).parent.parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ── style ───────────────────────────────────────────────────────────────────
BRAND_BLUE    = "#1A3C6E"
ACCENT_ORANGE = "#F57C00"
ACCENT_RED    = "#C62828"
ACCENT_GREEN  = "#2E7D32"
ACCENT_AMBER  = "#F9A825"
LIGHT_GREY    = "#F5F5F5"
TEXT_GREY     = "#444444"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   LIGHT_GREY,
    "axes.edgecolor":   "white",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
    "font.family":      "sans-serif",
    "text.color":       TEXT_GREY,
    "axes.labelcolor":  TEXT_GREY,
    "xtick.color":      TEXT_GREY,
    "ytick.color":      TEXT_GREY,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.titlepad":    14,
})


# ── load data ────────────────────────────────────────────────────────────────
def load() -> list[dict]:
    with open(DATA_FILE, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

rows = load()

for r in rows:
    r["sale_price"] = float(r["sale_price"]) if r["sale_price"] else 0.0
    r["quantity"]   = int(r["quantity"])   if r["quantity"]   else 0


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1 — Top 15 categories by product count
# ═══════════════════════════════════════════════════════════════════════════
def chart_category_count():
    counts = Counter(r["category"] for r in rows)
    top = counts.most_common(15)
    labels = [t[0] for t in reversed(top)]
    values = [t[1] for t in reversed(top)]
    colors = [BRAND_BLUE if v == max(values) else "#4A90D9" for v in values]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(labels, values, color=colors, height=0.6)
    ax.set_xlabel("Number of Products", labelpad=10)
    ax.set_title("Top 15 Product Categories by Catalogue Size")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for bar, val in zip(bars, values):
        ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9, color=TEXT_GREY)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "01_category_product_count.png", dpi=150)
    plt.close(fig)
    print("  [ok] 01_category_product_count.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 2 — Price distribution (bracket histogram)
# ═══════════════════════════════════════════════════════════════════════════
def chart_price_distribution():
    brackets = ["< 10 AZN", "10–20 AZN", "20–50 AZN",
                "50–100 AZN", "100–200 AZN", "200+ AZN"]
    edges    = [0, 10, 20, 50, 100, 200, float("inf")]
    counts   = [0] * len(brackets)
    for r in rows:
        p = r["sale_price"]
        for i in range(len(edges) - 1):
            if edges[i] <= p < edges[i + 1]:
                counts[i] += 1
                break

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(brackets, counts, color=BRAND_BLUE, width=0.55)
    ax.set_ylabel("Number of Products")
    ax.set_title("Price Distribution Across the Catalogue")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                str(val), ha="center", fontsize=9, color=TEXT_GREY)
    ax.set_ylim(0, max(counts) * 1.15)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "02_price_distribution.png", dpi=150)
    plt.close(fig)
    print("  [ok] 02_price_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 3 — Stock health overview (stacked bar, overall)
# ═══════════════════════════════════════════════════════════════════════════
def chart_stock_overview():
    def stock_band(q):
        if q == 0:    return "Out of Stock"
        if q <= 5:    return "Low (1–5)"
        if q <= 30:   return "Medium (6–30)"
        return "Healthy (30+)"

    bands   = ["Out of Stock", "Low (1–5)", "Medium (6–30)", "Healthy (30+)"]
    colors  = [ACCENT_RED, ACCENT_AMBER, ACCENT_ORANGE, ACCENT_GREEN]
    counts  = Counter(stock_band(r["quantity"]) for r in rows)
    values  = [counts[b] for b in bands]
    total   = sum(values)

    fig, ax = plt.subplots(figsize=(9, 4))
    left = 0
    for band, val, color in zip(bands, values, colors):
        pct = val / total * 100
        ax.barh(["All Products"], val, left=left, color=color,
                label=f"{band}  ({val:,} / {pct:.0f}%)", height=0.4)
        if pct > 4:
            ax.text(left + val / 2, 0,
                    f"{pct:.0f}%", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
        left += val

    ax.set_xlim(0, total)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Overall Stock Health — 4,866 Products")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45),
              ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "03_stock_health_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] 03_stock_health_overview.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 4 — Stock health by top 12 categories (stacked bar)
# ═══════════════════════════════════════════════════════════════════════════
def chart_stock_by_category():
    def stock_band(q):
        if q == 0:    return "Out of Stock"
        if q <= 5:    return "Low (1–5)"
        if q <= 30:   return "Medium (6–30)"
        return "Healthy (30+)"

    bands  = ["Out of Stock", "Low (1–5)", "Medium (6–30)", "Healthy (30+)"]
    colors = [ACCENT_RED, ACCENT_AMBER, ACCENT_ORANGE, ACCENT_GREEN]

    top_cats = [c for c, _ in Counter(r["category"] for r in rows).most_common(12)]
    data = {b: [] for b in bands}
    for cat in top_cats:
        sub = [r for r in rows if r["category"] == cat]
        bc  = Counter(stock_band(r["quantity"]) for r in sub)
        for b in bands:
            data[b].append(bc.get(b, 0))

    fig, ax = plt.subplots(figsize=(13, 6))
    lefts = np.zeros(len(top_cats))
    for band, color in zip(bands, colors):
        vals = np.array(data[band])
        ax.barh(top_cats, vals, left=lefts, color=color, label=band, height=0.55)
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v > 8:
                ax.text(l + v / 2, i, str(v), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        lefts += vals

    ax.set_xlabel("Number of Products")
    ax.set_title("Stock Health by Top 12 Categories")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "04_stock_by_category.png", dpi=150)
    plt.close(fig)
    print("  [ok] 04_stock_by_category.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 5 — Average price by top 15 categories
# ═══════════════════════════════════════════════════════════════════════════
def chart_avg_price_by_category():
    cat_prices = defaultdict(list)
    for r in rows:
        if r["sale_price"] > 0:
            cat_prices[r["category"]].append(r["sale_price"])

    top_cats = [c for c, _ in Counter(r["category"] for r in rows).most_common(15)]
    avgs = [(cat, np.mean(cat_prices[cat])) for cat in top_cats if cat_prices[cat]]
    avgs.sort(key=lambda x: x[1])

    labels = [a[0] for a in avgs]
    values = [a[1] for a in avgs]
    colors = [ACCENT_ORANGE if v == max(values) else "#4A90D9" for v in values]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(labels, values, color=colors, height=0.6)
    ax.set_xlabel("Average Sale Price (AZN)")
    ax.set_title("Average Product Price by Category")
    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} AZN", va="center", fontsize=9, color=TEXT_GREY)
    ax.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "05_avg_price_by_category.png", dpi=150)
    plt.close(fig)
    print("  [ok] 05_avg_price_by_category.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 6 — Product count by top-level subcategory (level-3 path segment)
# ═══════════════════════════════════════════════════════════════════════════
def chart_subcategory_breakdown():
    def level3(path):
        parts = [p.strip() for p in path.split(">")]
        return parts[3] if len(parts) > 3 else "Other"

    counts = Counter(level3(r["category_path"]) for r in rows)
    counts.pop("", None)
    top = counts.most_common(12)
    labels = [t[0] for t in reversed(top)]
    values = [t[1] for t in reversed(top)]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, values, color=BRAND_BLUE, height=0.6)
    ax.set_xlabel("Number of Products")
    ax.set_title("Product Volume by Subcategory Group")
    for bar, val in zip(bars, values):
        ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9, color=TEXT_GREY)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "06_subcategory_breakdown.png", dpi=150)
    plt.close(fig)
    print("  [ok] 06_subcategory_breakdown.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 7 — Out-of-stock rate by price bracket
# ═══════════════════════════════════════════════════════════════════════════
def chart_stockout_by_price():
    brackets = ["< 10 AZN", "10–20 AZN", "20–50 AZN",
                "50–100 AZN", "100–200 AZN", "200+ AZN"]
    edges    = [0, 10, 20, 50, 100, 200, float("inf")]

    total_by_bracket  = [0] * len(brackets)
    oos_by_bracket    = [0] * len(brackets)

    for r in rows:
        p = r["sale_price"]
        for i in range(len(edges) - 1):
            if edges[i] <= p < edges[i + 1]:
                total_by_bracket[i] += 1
                if r["quantity"] == 0:
                    oos_by_bracket[i] += 1
                break

    rates = [
        oos / tot * 100 if tot else 0
        for oos, tot in zip(oos_by_bracket, total_by_bracket)
    ]
    colors = [ACCENT_RED if r == max(rates) else "#E57373" for r in rates]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(brackets, rates, color=colors, width=0.55)
    ax.set_ylabel("Out-of-Stock Rate (%)")
    ax.set_title("Out-of-Stock Rate by Price Bracket")
    ax.set_ylim(0, max(rates) * 1.2)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.0f}%", ha="center", fontsize=9, color=TEXT_GREY)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "07_stockout_rate_by_price.png", dpi=150)
    plt.close(fig)
    print("  [ok] 07_stockout_rate_by_price.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 8 — Price vs. catalogue depth (scatter-like: avg price + count bubble)
#           rendered as grouped bar to stay business-friendly
# ═══════════════════════════════════════════════════════════════════════════
def chart_price_vs_depth():
    top_cats = [c for c, _ in Counter(r["category"] for r in rows).most_common(10)]
    counts = Counter(r["category"] for r in rows)
    cat_prices = defaultdict(list)
    for r in rows:
        if r["sale_price"] > 0:
            cat_prices[r["category"]].append(r["sale_price"])

    avgs = [np.mean(cat_prices[c]) if cat_prices[c] else 0 for c in top_cats]
    cnts = [counts[c] for c in top_cats]

    x     = np.arange(len(top_cats))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - width / 2, cnts, width, color=BRAND_BLUE,  label="Products in Catalogue")
    b2 = ax2.bar(x + width / 2, avgs, width, color=ACCENT_ORANGE, label="Avg Price (AZN)")

    ax1.set_ylabel("Number of Products", color=BRAND_BLUE)
    ax2.set_ylabel("Average Price (AZN)", color=ACCENT_ORANGE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_cats, rotation=30, ha="right", fontsize=9)
    ax1.set_title("Catalogue Depth vs. Average Price — Top 10 Categories")

    lines = [b1, b2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "08_depth_vs_avg_price.png", dpi=150)
    plt.close(fig)
    print("  [ok] 08_depth_vs_avg_price.png")


# ── run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating charts -> {CHARTS_DIR}\n")
    chart_category_count()
    chart_price_distribution()
    chart_stock_overview()
    chart_stock_by_category()
    chart_avg_price_by_category()
    chart_subcategory_breakdown()
    chart_stockout_by_price()
    chart_price_vs_depth()
    print("\nAll charts saved.")
