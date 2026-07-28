"""
Eval-Frame-290: patch exclusion stats vs expert ground truth (290 agreement frames).

Writes out_ablation_290/eval_frame_290_per_image.csv and patch_exclusion_histogram.png.
CSV columns: frame_classifier_prediction (argmax), initial_frame_prediction (after Urchin #1–#2),
recommendation (full rules), and patch exclusion counts.
Prints patch-exclusion summary, frame-classifier metrics (partial expert GT excluded), and histogram.

Agreement CSV uses legacy names (Reef-Grazed, Reef-Partial-Grazed, Reef-Vegetated); all
printed labels use Habibot names (Reef-BrLfa, Reef-Partial-BrLfa, Reef-FnEc).

Run from repository root:
  python python/ablation_290.py
"""

from __future__ import annotations

import gc
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
os.chdir(_REPO_ROOT)

OUT_DIR = _REPO_ROOT / "out_ablation_290"
IMAGE_DIR = _REPO_ROOT / "dataset" / "Eval-Frame-295"
AGREEMENT_CSV = _REPO_ROOT / "dataset" / "dataset_csv_files" / "User_Agreement_3.csv"
PREDICTIONS_CSV = OUT_DIR / "eval_frame_290_per_image.csv"
EXCLUSION_HIST_PNG = OUT_DIR / "patch_exclusion_histogram.png"
PARTIAL_EXPERT_GT = frozenset({"Reef-Partial-BrLfa", "Reef-Partial-Urchin-Barren"})
FRAME_CLASSIFIER_LABELS = (
    "Reef-Urchin-Barren",
    "Reef-BrLfa",
    "Reef-Kelp",
    "Reef-FnEc",
    "Unconsolidated",
)
SLIM_COLUMNS = (
    "image_name",
    "frame_classifier_prediction",
    "initial_frame_prediction",
    "recommendation",
    "n_patches_excluded",
    "pct_patches_excluded",
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_N = 290
PATCHES_PER_IMAGE = 88
EXTREME_PCT_THRESHOLD = 50.0  # strictly greater than this fraction of all patches


def normalize_label(label) -> str:
    """Map agreement / legacy strings to Habibot recommendation class names."""
    if not isinstance(label, str):
        return ""
    label = label.strip()
    if label == "Reef-Grazed":
        return "Reef-BrLfa"
    if label == "Reef-Partial-Grazed":
        return "Reef-Partial-BrLfa"
    if label == "Reef-Vegetated":
        return "Reef-FnEc"
    return label


def is_no_agreement(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "no agreement"
    return False


def load_eval290_agreement() -> pd.DataFrame:
    agr = pd.read_csv(AGREEMENT_CSV)
    agr = agr[~agr["Correct Recommendation"].map(is_no_agreement)].copy()
    agr["Image Name"] = agr["Image Name"].astype(str).str.strip()
    agr["expert_ground_truth"] = agr["Expert ground truth"].map(normalize_label)
    return agr


def list_eval290_images(folder: Path) -> list[Path]:
    allowed = set(load_eval290_agreement()["Image Name"])
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.name in allowed
        ],
        key=lambda p: p.name,
    )


def load_rgb(path: Path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (3648, 2736), interpolation=cv2.INTER_LINEAR)


def run_inference() -> pd.DataFrame:
    from demo import Habibot

    bot = Habibot(
        frame_classifier_path=str(
            _REPO_ROOT / "trained_classifiers" / "frame_classifier" / "frame_classifer.h5"
        ),
        patch_classifier_path=str(
            _REPO_ROOT / "trained_classifiers" / "patch_classifier" / "patch_classifier.h5"
        ),
        crop_ratio=0.2,
    )

    rows = []
    for path in tqdm(list_eval290_images(IMAGE_DIR), desc="Eval-Frame-290"):
        (
            _img,
            _pb,
            _pc,
            ratios,
            _ppc,
            patches_probs,
            frame_probs,
            urchin_boxes,
        ) = bot.get_all_models_predictions(load_rgb(path))
        probs = np.asarray(patches_probs)
        n_excl = int(np.sum(np.max(probs, axis=1) < bot.patch_conf_threshold))
        initial_frame, recommendation, _rules = bot.apply_recommendation_rules(
            frame_probs, ratios, urchin_boxes
        )
        frame_probs_arr = np.asarray(frame_probs)
        frame_argmax = bot.i_to_c_frame[int(np.argmax(frame_probs_arr))]
        rows.append(
            {
                "image_name": path.name,
                "frame_classifier_prediction": normalize_label(frame_argmax),
                "initial_frame_prediction": normalize_label(initial_frame),
                "recommendation": normalize_label(recommendation),
                "n_patches_excluded": n_excl,
                "pct_patches_excluded": float(ratios["Unscorable"]),
            }
        )
        gc.collect()

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Wrote {len(df)} rows to {PREDICTIONS_CSV.resolve()}")
    return df


def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_CSV)
    if "n_patches_excluded" not in df.columns:
        if "n_unscorable_excluded" in df.columns:
            df["n_patches_excluded"] = df["n_unscorable_excluded"]
        elif "n_patches_excluded_low_confidence" in df.columns:
            df["n_patches_excluded"] = df["n_patches_excluded_low_confidence"]
    if "pct_patches_excluded" not in df.columns and "pct_unscorable" in df.columns:
        df["pct_patches_excluded"] = df["pct_unscorable"]
    df["recommendation"] = df["recommendation"].map(normalize_label)
    for col in ("initial_frame_prediction", "frame_classifier_prediction"):
        if col in df.columns:
            df[col] = df[col].map(normalize_label)
    if not set(SLIM_COLUMNS).issubset(df.columns):
        return df
    slim = df[list(SLIM_COLUMNS)].copy()
    if list(df.columns) != list(SLIM_COLUMNS):
        slim.to_csv(PREDICTIONS_CSV, index=False)
    return slim


def print_patch_exclusion_summary(pred: pd.DataFrame) -> None:
    agr = load_eval290_agreement()
    merged = agr.merge(
        pred,
        left_on="Image Name",
        right_on="image_name",
        how="inner",
    )

    mean_excluded = float(merged["n_patches_excluded"].mean())
    extreme = merged[merged["pct_patches_excluded"] > EXTREME_PCT_THRESHOLD]

    print()
    print(f"1) Mean low-confidence patches excluded per image (n={len(merged)}, "
          f"{PATCHES_PER_IMAGE} patches per image): {mean_excluded:.2f}")
    print()
    print(
        f"2) Images with exclusion > {EXTREME_PCT_THRESHOLD:.0f}% of patches: "
        f"{len(extreme)}"
    )
    print()
    print(f"3) Expert ground-truth distribution on those images (normalized labels):")
    if len(extreme) == 0:
        print("   (none — no image exceeds 50% excluded; "
              f"{int((merged['pct_patches_excluded'] == EXTREME_PCT_THRESHOLD).sum())} "
              f"at exactly {EXTREME_PCT_THRESHOLD:.0f}%)")
    else:
        counts = Counter(extreme["expert_ground_truth"].tolist())
        for label, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"   {label}: {count}")


EXCLUSION_BIN_CAP = 55  # last x bin is 55+ (integer % >= 55)


def _exclusion_bin(pct: float) -> int:
    i = min(100, int(pct))
    return EXCLUSION_BIN_CAP if i >= EXCLUSION_BIN_CAP else i


def plot_exclusion_histogram(pred: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    n_bins = EXCLUSION_BIN_CAP + 1  # 0..54 plus 55+

    agr = load_eval290_agreement()
    merged = agr.merge(
        pred,
        left_on="Image Name",
        right_on="image_name",
        how="inner",
    )
    merged["excl_bin"] = merged["pct_patches_excluded"].map(_exclusion_bin)
    merged["rule_wrong"] = merged["recommendation"] != merged["expert_ground_truth"]

    bin_totals = (
        merged.groupby("excl_bin")
        .size()
        .reindex(range(n_bins), fill_value=0)
        .astype(int)
    )
    bin_wrong = (
        merged.groupby("excl_bin")["rule_wrong"]
        .sum()
        .reindex(range(n_bins), fill_value=0)
        .astype(int)
    )
    heights = bin_totals.to_numpy()
    wrong_heights = bin_wrong.to_numpy()
    correct_heights = heights - wrong_heights

    x = np.arange(n_bins)
    n_images = len(merged)
    class_counts = merged["expert_ground_truth"].value_counts()
    class_order = class_counts.index.tolist()
    class_pct_all = (class_counts / n_images * 100.0).to_dict()

    def legend_label(cls: str) -> str:
        return f"{cls} ({class_pct_all[cls]:.1f}%)"

    by_bin = (
        merged.groupby(["excl_bin", "expert_ground_truth"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(n_bins), fill_value=0)
    )
    for cls in class_order:
        if cls not in by_bin.columns:
            by_bin[cls] = 0
    by_bin = by_bin[class_order]

    palette = plt.get_cmap("tab10")
    class_colors = {cls: palette(i % 10) for i, cls in enumerate(class_order)}

    fs_axis = 18
    fs_bar_top = 17
    fs_bar_stack = 16
    fs_legend = 16

    xtick_pos = list(range(0, EXCLUSION_BIN_CAP, 5)) + [EXCLUSION_BIN_CAP]
    xtick_lbl = [str(i) for i in range(0, EXCLUSION_BIN_CAP, 5)] + [f"{EXCLUSION_BIN_CAP}+"]
    x_label = "Patch exclusion (% of patches per image, low-confidence excluded)"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [1, 1.15]}
    )

    bar_blue = "#4C72B0"
    bar_edge = "#2f4f7f"
    hatch_wrong = "////"
    plt.rcParams["hatch.linewidth"] = 2.2

    ax_top.bar(
        x,
        correct_heights,
        width=0.85,
        color=bar_blue,
        edgecolor="white",
        linewidth=0.3,
    )
    ax_top.bar(
        x,
        wrong_heights,
        bottom=correct_heights,
        width=0.85,
        color=bar_blue,
        hatch=hatch_wrong,
        edgecolor=bar_edge,
        linewidth=0.5,
    )
    for xi, h, w in zip(x, heights, wrong_heights):
        if h > 0:
            ax_top.text(xi, h, str(h), ha="center", va="bottom", fontsize=fs_bar_top)
        if w > 0:
            ax_top.text(
                xi,
                correct_heights[xi] + w / 2,
                str(w),
                ha="center",
                va="center",
                fontsize=fs_bar_top,
                color="white",
            )
    ax_top.set_xlim(-0.5, EXCLUSION_BIN_CAP + 0.5)
    ax_top.set_ylabel("Number of images", fontsize=fs_axis)
    ax_top.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_top.tick_params(axis="y", labelsize=fs_axis)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.legend(
        handles=[
            Patch(facecolor=bar_blue, edgecolor="white", label="Number of images"),
            Patch(
                facecolor=bar_blue,
                hatch=hatch_wrong,
                edgecolor=bar_edge,
                label="Wrong rule-based recommendation vs expert GT",
            ),
        ],
        loc="upper right",
        fontsize=fs_legend,
    )

    bottom = np.zeros(n_bins)
    for cls in class_order:
        vals = by_bin[cls].to_numpy(dtype=float)
        ax_bot.bar(
            x,
            vals,
            bottom=bottom,
            width=0.85,
            color=class_colors[cls],
            edgecolor="white",
            linewidth=0.2,
        )
        for xi, v in enumerate(vals):
            if v > 0:
                ax_bot.text(
                    xi,
                    bottom[xi] + v / 2,
                    str(int(v)),
                    ha="center",
                    va="center",
                    fontsize=fs_bar_stack,
                    color="white" if v >= 2 else "black",
                )
        bottom += vals

    ax_bot.set_xlim(-0.5, EXCLUSION_BIN_CAP + 0.5)
    ax_bot.set_ylabel("Number of images", fontsize=fs_axis)
    ax_bot.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_bot.set_xticks(xtick_pos)
    ax_bot.set_xticklabels(xtick_lbl, fontsize=fs_axis)
    ax_bot.tick_params(axis="y", labelsize=fs_axis)
    ax_bot.legend(
        handles=[
            Patch(facecolor=class_colors[cls], edgecolor="white", label=legend_label(cls))
            for cls in class_order
        ],
        loc="upper right",
        fontsize=fs_legend,
        ncol=1,
    )

    fig.supxlabel(x_label, fontsize=fs_axis, y=0.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    fig.savefig(EXCLUSION_HIST_PNG, dpi=150)
    plt.close(fig)
    print(f"\nWrote {EXCLUSION_HIST_PNG.resolve()}")


def print_frame_classifier_metrics(pred: pd.DataFrame) -> None:
    from sklearn.metrics import accuracy_score, classification_report

    agr = load_eval290_agreement()
    merged = agr.merge(
        pred,
        left_on="Image Name",
        right_on="image_name",
        how="inner",
    )

    if "frame_classifier_prediction" in merged.columns:
        pred_col = "frame_classifier_prediction"
        pred_desc = "frame classifier argmax"
    elif "initial_frame_prediction" in merged.columns:
        pred_col = "initial_frame_prediction"
        pred_desc = "initial_frame_prediction (Urchin rules #1–#2 only; no frame_classifier column)"
    else:
        print("\nSkipping frame classifier metrics: no frame prediction column in CSV.")
        return

    n_all = len(merged)
    merged = merged[~merged["expert_ground_truth"].isin(PARTIAL_EXPERT_GT)].copy()
    n_eval = len(merged)
    n_dropped_partial = n_all - n_eval

    y_true = merged["expert_ground_truth"].tolist()
    y_pred = merged[pred_col].tolist()
    labels = list(FRAME_CLASSIFIER_LABELS)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, digits=3
    )

    print()
    print("=" * 72)
    print("Frame classifier vs expert ground truth (Eval-Frame-290 subset)")
    print("=" * 72)
    print(f"Excluded {n_dropped_partial} images with partial expert GT "
          f"({', '.join(sorted(PARTIAL_EXPERT_GT))}).")
    print(f"Evaluated {n_eval} images; prediction: {pred_desc}.")
    print(f"Label set (5 frame classes): {', '.join(labels)}")
    print()
    print(f"Accuracy: {acc * 100:.2f}% ({int((np.array(y_true) == np.array(y_pred)).sum())}/{n_eval})")
    print()
    print(report)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if PREDICTIONS_CSV.is_file():
        pred = load_predictions()
        if "initial_frame_prediction" not in pred.columns:
            print(
                "Predictions CSV missing initial_frame_prediction; re-running inference..."
            )
            pred = run_inference()
    else:
        pred = run_inference()

    names = set(load_eval290_agreement()["Image Name"])
    pred = pred[pred["image_name"].astype(str).str.strip().isin(names)]
    if len(pred) != EXPECTED_N:
        print(f"Note: {len(pred)} prediction rows (expected {EXPECTED_N}).")

    print_patch_exclusion_summary(pred)
    plot_exclusion_histogram(pred)
    print_frame_classifier_metrics(pred)


if __name__ == "__main__":
    main()
