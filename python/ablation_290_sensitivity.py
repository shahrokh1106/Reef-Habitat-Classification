"""
Eval-Frame-290 rule-threshold sensitivity: cache minimal CSV, then sweep and plot.

Writes to out_ablation_290_sensitivity/:
  - eval_frame_290_sensitivity.csv  (columns needed to replay rules at patch_conf=0.7)
  - sensitivity_*.csv / sensitivity_*.png

If eval_frame_290_sensitivity.csv exists, skips inference and only runs sweeps/plots.
Delete the CSV to rebuild from scratch.

Run from repository root:
  python python/ablation_290_sensitivity.py
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
os.chdir(_REPO_ROOT)

OUT_DIR = _REPO_ROOT / "out_ablation_290_sensitivity"
IMAGE_DIR = _REPO_ROOT / "dataset" / "Eval-Frame-295"
AGREEMENT_CSV = _REPO_ROOT / "dataset" / "dataset_csv_files" / "User_Agreement_3.csv"
CACHE_CSV = OUT_DIR / "eval_frame_290_sensitivity.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_N = 290
PATCH_CONF_DEFAULT = 0.7
FRAME_LOW_CONF_DEFAULT = 0.6
BRLFA_MIX_DEFAULT = 25.0
MACRO_MIX_DEFAULT = 25.0

I_TO_C_FRAME = {
    0: "Reef-Urchin-Barren",
    1: "Reef-BrLfa",
    2: "Reef-Kelp",
    3: "Reef-FnEc",
    4: "Unconsolidated",
}
FRAME_PROB_COLS = [f"frame_prob_{i}_{I_TO_C_FRAME[i]}" for i in range(5)]
MACRO_CANOPY_KEYS = ("Carpophyllum", "Ecklonia", "Other canopy", "Foliose algae")

CACHE_COLUMNS = (
    ["expert_ground_truth", "urchin_box_count", "ratio_BrLfa", "macro_canopy_sum_pct"]
    + FRAME_PROB_COLS
)


def normalize_label(label) -> str:
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


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (3648, 2736), interpolation=cv2.INTER_LINEAR)


def apply_recommendation_rules_cached(
    frame_probs: np.ndarray,
    ratio_brlfa: float,
    macro_canopy_sum: float,
    urchin_box_count: int,
    frame_low_conf: float,
    brlfa_mix_pct: float,
    macro_mix_pct: float,
) -> str:
    """Mirror demo.apply_recommendation_rules using cached scalars (ratios at patch 0.7)."""
    frame_probs = np.asarray(frame_probs, dtype=float).reshape(-1)
    has_urchin = urchin_box_count > 0
    macro_sum = macro_canopy_sum

    initial = I_TO_C_FRAME[int(np.argmax(frame_probs))]
    recommendation = "None"

    if initial == "Reef-BrLfa" and has_urchin:
        initial = "Reef-Urchin-Barren"
    elif initial == "Reef-Urchin-Barren" and not has_urchin:
        initial = "Reef-BrLfa"

    if np.max(frame_probs) <= frame_low_conf:
        recommendation = (
            "Reef-Partial-Urchin-Barren" if has_urchin else "Reef-Partial-BrLfa"
        )
    elif initial == "Reef-Kelp":
        if has_urchin:
            recommendation = "Reef-Partial-Urchin-Barren"
        elif ratio_brlfa > brlfa_mix_pct and not has_urchin:
            recommendation = "Reef-Partial-BrLfa"
    elif initial == "Reef-FnEc":
        if has_urchin:
            recommendation = "Reef-Partial-Urchin-Barren"
        elif ratio_brlfa > brlfa_mix_pct and not has_urchin:
            recommendation = "Reef-Partial-BrLfa"
    elif initial == "Reef-Urchin-Barren":
        if macro_sum > macro_mix_pct:
            recommendation = "Reef-Partial-Urchin-Barren"
    elif initial == "Reef-BrLfa":
        if macro_sum > macro_mix_pct:
            recommendation = "Reef-Partial-BrLfa"
    elif initial == "Unconsolidated" and has_urchin:
        recommendation = "Reef-Partial-Urchin-Barren"

    if recommendation == "None":
        recommendation = initial
    return recommendation


def run_cache_inference() -> pd.DataFrame:
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

    agr = load_eval290_agreement()
    expert_by_image = dict(zip(agr["Image Name"], agr["expert_ground_truth"]))

    rows: list[dict] = []
    n_mismatch = 0

    for path in tqdm(list_eval290_images(IMAGE_DIR), desc="Sensitivity cache"):
        (
            _img,
            _pb,
            _pc,
            ratios,
            _ppc,
            _patches_probs,
            frame_probs,
            urchin_boxes,
        ) = bot.get_all_models_predictions(load_rgb(path))

        _initial_frame, recommendation, _rule_numbers = bot.apply_recommendation_rules(
            frame_probs, ratios, urchin_boxes
        )

        frame_probs_arr = np.asarray(frame_probs).reshape(-1)
        ratio_brlfa = float(ratios.get("BrLfa", 0.0))
        macro_sum = sum(float(ratios.get(k, 0.0)) for k in MACRO_CANOPY_KEYS)

        replay = apply_recommendation_rules_cached(
            frame_probs_arr,
            ratio_brlfa,
            macro_sum,
            len(urchin_boxes),
            FRAME_LOW_CONF_DEFAULT,
            BRLFA_MIX_DEFAULT,
            MACRO_MIX_DEFAULT,
        )
        demo_rec = normalize_label(recommendation)
        if replay != demo_rec:
            n_mismatch += 1

        row = {
            "expert_ground_truth": expert_by_image.get(path.name, ""),
            "urchin_box_count": len(urchin_boxes),
            "ratio_BrLfa": ratio_brlfa,
            "macro_canopy_sum_pct": macro_sum,
        }
        for i in range(5):
            row[FRAME_PROB_COLS[i]] = float(frame_probs_arr[i])
        rows.append(row)
        gc.collect()

    df = pd.DataFrame(rows)[CACHE_COLUMNS]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_CSV, index=False)
    print(f"Wrote {len(df)} rows to {CACHE_CSV.resolve()}")
    if n_mismatch:
        print(f"Warning: replay vs demo recommendation mismatch on {n_mismatch} images.")
    else:
        print("Replay at default thresholds matches demo on all images.")
    return df


def predict_column(
    df: pd.DataFrame, frame_low_conf: float, brlfa_mix_pct: float, macro_mix_pct: float
) -> pd.Series:
    preds = []
    for _, row in df.iterrows():
        frame_probs = row[FRAME_PROB_COLS].to_numpy(dtype=float)
        preds.append(
            apply_recommendation_rules_cached(
                frame_probs,
                float(row["ratio_BrLfa"]),
                float(row["macro_canopy_sum_pct"]),
                int(row["urchin_box_count"]),
                frame_low_conf,
                brlfa_mix_pct,
                macro_mix_pct,
            )
        )
    return pd.Series(preds, index=df.index)


def evaluate_preds(y_true: pd.Series, y_pred: pd.Series) -> dict:
    yt = y_true.astype(str).tolist()
    yp = y_pred.astype(str).tolist()
    labels = sorted(set(yt) | set(yp))
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
        "n": len(yt),
    }


def sweep_1d(
    df: pd.DataFrame,
    values: np.ndarray,
    param_name: str,
    baseline_pred: pd.Series,
    fixed_frame: float,
    fixed_brlfa: float,
    fixed_macro: float,
    vary: str,
) -> pd.DataFrame:
    rows = []
    y_true = df["expert_ground_truth"]
    baseline_metrics = evaluate_preds(y_true, baseline_pred)
    baseline_acc = baseline_metrics["accuracy"]
    baseline_f1 = baseline_metrics["macro_f1"]
    for v in values:
        if vary == "frame_low_conf":
            pred = predict_column(df, float(v), fixed_brlfa, fixed_macro)
        elif vary == "brlfa_mix":
            pred = predict_column(df, fixed_frame, float(v), fixed_macro)
        elif vary == "macro_mix":
            pred = predict_column(df, fixed_frame, fixed_brlfa, float(v))
        else:
            raise ValueError(vary)
        m = evaluate_preds(y_true, pred)
        n_flip = int((pred != baseline_pred).sum())
        rows.append(
            {
                param_name: float(v),
                "delta_accuracy_pct": (m["accuracy"] - baseline_acc) * 100.0,
                "delta_macro_f1_pct": (m["macro_f1"] - baseline_f1) * 100.0,
                "n_images": m["n"],
                "n_changed_vs_default": n_flip,
                "pct_changed_vs_default": 100.0 * n_flip / m["n"],
            }
        )
    return pd.DataFrame(rows)


def plot_sweep(
    sweep_df: pd.DataFrame, param_name: str, default_val: float, out_png: Path
) -> None:
    flip_color = "#E67E22"
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = sweep_df[param_name]
    ax1.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax1.plot(
        x,
        sweep_df["delta_accuracy_pct"],
        "o-",
        color="#4C72B0",
        label="Accuracy change (%)",
        zorder=3,
    )
    ax1.plot(
        x,
        sweep_df["delta_macro_f1_pct"],
        "s--",
        color="#C44E52",
        label="Macro F1 change (%)",
        zorder=3,
    )
    ax1.axvline(default_val, color="gray", linestyle=":", linewidth=1.5, label=f"Default ({default_val})")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Gain or loss vs default (% accuracy or macro F1)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    n_changed = sweep_df["n_changed_vs_default"]
    ax2.plot(
        x,
        n_changed,
        "^-",
        color=flip_color,
        linewidth=2,
        markersize=6,
        label="# recommendations",
        clip_on=True,
        zorder=3,
    )
    n_max = int(n_changed.max())
    ax2.set_ylim(0, max(n_max + 1, 1))
    ax2.margins(y=0)
    ax2.set_ylabel("# recommendations", color=flip_color)
    ax2.tick_params(axis="y", labelcolor=flip_color)
    ax2.spines["right"].set_color(flip_color)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.22)
    fig.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        bbox_transform=fig.transFigure,
        ncol=4,
        frameon=True,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def run_sweeps_and_plots(df: pd.DataFrame) -> None:
    missing = [c for c in CACHE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Cache CSV is missing columns {missing}. "
            f"Delete {CACHE_CSV} and re-run to rebuild."
        )

    if len(df) != EXPECTED_N:
        print(f"Note: {len(df)} rows (expected {EXPECTED_N}).")

    baseline_pred = predict_column(
        df, FRAME_LOW_CONF_DEFAULT, BRLFA_MIX_DEFAULT, MACRO_MIX_DEFAULT
    )
    print()
    print("=" * 72)
    print("Rule threshold sensitivity (all expert classes, n=%d)" % len(df))
    print("=" * 72)
    print("Fixed: patch_conf=0.7 (ratios baked into cache), urchin_box_count unchanged.")
    print(
        f"Default thresholds: frame_low_conf={FRAME_LOW_CONF_DEFAULT}, "
        f"brlfa_mix={BRLFA_MIX_DEFAULT}%, macro_mix={MACRO_MIX_DEFAULT}%"
    )
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_vals = np.round(np.arange(0.50, 0.751, 0.025), 3)
    mix_vals = np.arange(10.0, 40.1, 2.5)

    sweeps = [
        ("frame_low_conf", frame_vals, FRAME_LOW_CONF_DEFAULT, "sensitivity_frame_low_conf"),
        ("brlfa_mix_pct", mix_vals, BRLFA_MIX_DEFAULT, "sensitivity_brlfa_mix_pct"),
        ("macro_mix_pct", mix_vals, MACRO_MIX_DEFAULT, "sensitivity_macro_mix_pct"),
    ]

    for vary, values, default, stem in sweeps:
        if vary == "frame_low_conf":
            sweep_df = sweep_1d(
                df,
                values,
                vary,
                baseline_pred,
                FRAME_LOW_CONF_DEFAULT,
                BRLFA_MIX_DEFAULT,
                MACRO_MIX_DEFAULT,
                "frame_low_conf",
            )
        elif vary == "brlfa_mix_pct":
            sweep_df = sweep_1d(
                df,
                values,
                vary,
                baseline_pred,
                FRAME_LOW_CONF_DEFAULT,
                BRLFA_MIX_DEFAULT,
                MACRO_MIX_DEFAULT,
                "brlfa_mix",
            )
        else:
            sweep_df = sweep_1d(
                df,
                values,
                vary,
                baseline_pred,
                FRAME_LOW_CONF_DEFAULT,
                BRLFA_MIX_DEFAULT,
                MACRO_MIX_DEFAULT,
                "macro_mix",
            )

        csv_path = OUT_DIR / f"{stem}.csv"
        png_path = OUT_DIR / f"{stem}.png"
        sweep_df.to_csv(csv_path, index=False)
        plot_sweep(sweep_df, vary, default, png_path)
        at_default = sweep_df.loc[(sweep_df[vary] - default).abs().idxmin()]
        acc_delta = sweep_df["delta_accuracy_pct"]
        f1_delta = sweep_df["delta_macro_f1_pct"]
        max_acc_inc = float(acc_delta.max())
        max_acc_dec = float(-acc_delta.min())
        max_f1_inc = float(f1_delta.max())
        max_f1_dec = float(-f1_delta.min())
        print(f"{vary}:")
        print(f"  Wrote {csv_path.name}, {png_path.name}")
        print(
            f"  At default: accuracy change={at_default['delta_accuracy_pct']:.3f}%, "
            f"macro F1 change={at_default['delta_macro_f1_pct']:.3f}%, "
            f"changed={int(at_default['n_changed_vs_default'])} recommendations"
        )
        print(f"  Max accuracy increase over sweep: {max_acc_inc:.2f}%")
        print(f"  Max accuracy decrease over sweep: {max_acc_dec:.2f}%")
        print(f"  Max macro F1 increase over sweep: {max_f1_inc:.2f}%")
        print(f"  Max macro F1 decrease over sweep: {max_f1_dec:.2f}%")
        print()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_CSV.is_file():
        print(f"Using existing cache: {CACHE_CSV.resolve()}")
        df = pd.read_csv(CACHE_CSV)
    else:
        df = run_cache_inference()

    run_sweeps_and_plots(df)


if __name__ == "__main__":
    main()
