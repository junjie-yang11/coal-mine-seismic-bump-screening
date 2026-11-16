"""Recreate Figure 8 from the project's saved classification and Top-K results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fontTools.ttLib import TTFont


MODELS = ("LR", "RF", "RBF-SVM", "XGBoost")
COLORS = ("#D55E00", "#0072B2", "#007D61", "#8E6C8A")
MARKERS = ("o", "s", "^", "D")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_a = args.project / "实验/results/model_summary.csv"
    source_b = args.project / "补充分析/等复核量比较/results/pooled_results.csv"
    frozen = {r["model"]: r for r in read_rows(source_a) if r["model"] in MODELS}
    matched = {r["model"]: r for r in read_rows(source_b) if r["setting"] == "fixed_20pct"}
    assert tuple(frozen) != () and set(frozen) == set(matched) == set(MODELS)

    data = []
    for model in MODELS:
        for panel, rows, workload_key in (("a", frozen, "alerts"), ("b", matched, "review_count")):
            row = rows[model]
            count, tp, fp, fn = (int(row[k]) for k in (workload_key, "tp", "fp", "fn"))
            assert tp + fp == count and tp + fn == 170
            data.append(dict(panel=panel, model=model, review_count=count, tp=tp, fp=fp, fn=fn))
    a_rows, b_rows = [r for r in data if r["panel"] == "a"], [r for r in data if r["panel"] == "b"]
    workload, = {r["review_count"] for r in b_rows}

    sys.path.insert(0, str(args.project.resolve() / "代码/图件检查工具"))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "图8_检出与复核名额"
    with TemporaryDirectory(prefix="seismic-fig8-", ignore_cleanup_errors=True) as directory:
        font_path = Path(directory) / "YaHei.ttf"
        with TTFont("C:/Windows/Fonts/msyh.ttc", fontNumber=0) as chinese:
            chinese.save(font_path)
        font_manager.fontManager.addfont(font_path)
        mpl.rcParams.update({
            "font.family": ["Arial", font_manager.FontProperties(fname=font_path).get_name()],
            "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
            "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
            "axes.labelcolor": "#20262D", "axes.edgecolor": "#505A63", "text.color": "#20262D",
            "xtick.color": "#505A63", "ytick.color": "#505A63", "axes.unicode_minus": False,
            "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False,
        })
        fig, (a, b) = plt.subplots(1, 2, figsize=(6.55, 3.05))
        fig.subplots_adjust(left=.103, right=.972, bottom=.22, top=.845, wspace=.50)
        for ax, letter, label in (
            (a, "a", "各折冻结阈值"),
            (b, "b", f"每折前 20%（各复核 {workload} 条）"),
        ):
            ax.text(0, 1.045, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
            ax.annotate(label, (0, 1.045), xycoords="axes fraction",
                        xytext=(12, 0), textcoords="offset points", fontsize=8.5, va="bottom")
        for row, color, marker, offset in zip(a_rows, COLORS, MARKERS, ((6, 4), (6, 4), (6, 5), (6, -12))):
            x, y = row["review_count"], row["tp"]
            a.plot(x, y, marker=marker, markersize=5, color=color)
            a.annotate(row["model"], (x, y), xytext=offset, textcoords="offset points",
                       fontsize=7.5, color="#20262D")
        lr, rf = a_rows[:2]
        a.text(.035, .955, f"RF 较 LR 多检出 {rf['tp']-lr['tp']} 条\n多复核 {rf['review_count']-lr['review_count']} 条",
               transform=a.transAxes, fontsize=7, va="top", linespacing=1.3)
        a.set(xlim=(440, 805), ylim=(84, 135), xticks=(500, 600, 700, 800),
              yticks=(90, 100, 110, 120, 130), xlabel="待复核记录数", ylabel="检出危险记录数")
        for index, (row, color) in enumerate(zip(b_rows, COLORS)):
            tp, fn = row["tp"], row["fn"]
            b.barh(index, tp, color=color, height=.54)
            b.barh(index, fn, left=tp, color="#E5EAEE", height=.54)
            b.text(tp / 2, index, str(tp), va="center", ha="center", color="white", fontsize=8)
            b.text(tp + fn / 2, index, str(fn), va="center", ha="center", fontsize=8)
        b.set(yticks=range(len(MODELS)), yticklabels=MODELS, ylim=(3.55, -.55),
              xlim=(0, 170), xticks=(0, 50, 100, 150, 170), xlabel="危险记录数（检出 / 漏报）")
        b.tick_params(axis="y", length=0)
        require_matplotlib_panel_alignment(
            fig, axes=[a, b], panel_ids=["a", "b"], row_groups=[["a", "b"]],
            json_out=stem.with_suffix(".alignment.json"), require_panel_labels=True,
            strict=True, tolerance_pt=1.5,
        )
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".svg"))
        fig.savefig(stem.with_suffix(".png"), dpi=600)
        plt.close(fig)

    with stem.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    provenance = {
        "a": {"path": source_a.relative_to(args.project).as_posix()},
        "b": {"path": source_b.relative_to(args.project).as_posix(), "filter": "fixed_20pct"},
    }
    for panel, file in (("a", source_a), ("b", source_b)):
        provenance[panel]["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    stem.with_suffix(".sources.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
