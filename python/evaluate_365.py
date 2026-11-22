"""
Evaluate Habibot models on dataset/Test-Frame-365 expert annotations.

Evaluation set (important):
  Frame, patch, and urchin metrics use every annotated image in Test-Frame-365.json.
  Rule agreement metrics still use only images where the three experts agree
  (Check == True in Habitat_7_Class.xlsx). Each expert folder uses all images.

  Ground truth:
    --frame / --patch / --urchin  -> Test-Frame-365.json
      (frame_label / points / urchin boxes)
    --rule                        -> Habitat_7_Class.xlsx agreed 7-class label
      (Arie/Kelsey/Celia when Check==True; e.g. Reef-nEckFo remapped to Reef-FnEc)

Usage (from repository root):
  python python/evaluate_365.py --frame
  python python/evaluate_365.py --patch
  python python/evaluate_365.py --urchin
  python python/evaluate_365.py --rule
  python python/evaluate_365.py --time
  python python/evaluate_365.py --all
  python python/evaluate_365.py --frame --debug
  python python/evaluate_365.py --patch --debug
  python python/evaluate_365.py --urchin --debug
  python python/evaluate_365.py --rule --debug

Results:
  results-365/{frame,patch,urchin,rule}/   (repository root)
    classification_report.json / .png, confusion_matrix.png
  Patch has two subfolders:
    patch/argmax/       — every defined GT point scored by argmax (default metrics)
    patch/conf-cutoff/  — Habibot-style: only points with max(prob) >= 0.7;
                          low-confidence predictions abstain (excluded from metrics)
    Each subfolder also gets per_image_accuracy.png / .json: one accuracy per
    image, with the mean, the standard deviation, and an accuracy box plot.
  --time reads time_keeping.json and writes results-365/stats_time.json
    (totals and per-image minutes, no image names).
  Rule writes rule/ (metrics + predictions.csv).
    The rule is demo.Habibot.apply_recommendation_rules applied to the SAME predictions
    as --frame/--patch/--urchin: original image, annotator patch centers, patch %
    from conf>=0.7 predictions (Urchin excluded).
  With --debug, also side-by-side plots under .../<task>/debug/ (patch: under each subfolder):
    left=GT, right=prediction on the SAME Test-Frame-365 image & annotator grid
    --frame --debug  : frame label only
    --patch --debug  : patch points only (same centers both sides; conf-cutoff marks Unscorable)
    --urchin --debug : urchin boxes only
    --rule --debug   : one figure with habitat, frame, patches (conf-cutoff, the rule's
                       input), urchins, patch %, and fired rules
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow.keras import layers
from tensorflow.keras.applications.convnext import preprocess_input as preprocess_input_convnext
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input as preprocess_input_inception_v3,
)
from tqdm import tqdm

# Allow `python python/evaluate_365.py` from repo root
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from get_yolo_model import GetYoloModel


def configure_tf_memory() -> None:
    """Avoid TF pre-allocating all GPU memory (helps host RAM pressure)."""
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = REPO_ROOT / "dataset" / "Test-Frame-365"
DEFAULT_JSON = TEST_DIR / "Test-Frame-365.json"
DEFAULT_IMAGES = TEST_DIR / "images"
DEFAULT_AGREEMENT_XLSX = TEST_DIR / "Habitat_7_Class.xlsx"
DEFAULT_TIME_JSON = TEST_DIR / "time_keeping.json"
RESULTS_ROOT = REPO_ROOT / "results-365"

FRAME_MODEL_PATH = REPO_ROOT / "trained_classifiers" / "frame_classifier" / "frame_classifer.h5"
PATCH_MODEL_PATH = REPO_ROOT / "trained_classifiers" / "patch_classifier" / "patch_classifier.h5"

# Model label maps (same order as python/demo.py Habibot)
I_TO_C_FRAME = {
    0: "Reef-Urchin-Barren",
    1: "Reef-BrLfa",
    2: "Reef-Kelp",
    3: "Reef-FnEc",
    4: "Unconsolidated",
}
C_TO_I_FRAME = {v: k for k, v in I_TO_C_FRAME.items()}
FRAME_LABELS = list(I_TO_C_FRAME.values())

I_TO_C_PATCH = {
    0: "Carpophyllum",
    1: "Ecklonia",
    2: "Foliose algae",
    3: "Other canopy",
    4: "Unconsolidated",
    5: "Urchin",
    6: "BrLfa",
}
C_TO_I_PATCH = {v: k for k, v in I_TO_C_PATCH.items()}
PATCH_LABELS = list(I_TO_C_PATCH.values())

URCHIN_LABELS = ["no_urchin", "urchin"]
UNDEFINED_POINT = "undefined_point"
UNDEFINED_URCHIN = "undefined_urchin"
UNSCORABLE = "Unscorable"  # Habibot low-confidence abstain label (debug / reporting)
PATCH_CONF_THRESHOLD = 0.7  # same as demo.Habibot.patch_conf_threshold

# Habibot rule-based 7-class label space (matches Habitat_7_Class.xlsx after remap)
RULE_LABELS = [
    "Reef-Urchin-Barren",
    "Reef-BrLfa",
    "Reef-Kelp",
    "Reef-FnEc",
    "Unconsolidated",
    "Reef-Partial-BrLfa",
    "Reef-Partial-Urchin-Barren",
]

# Excel / legacy names -> Habibot rule recommendation names
RULE_LABEL_REMAP = {
    "Reef-Urchin-Barren": "Reef-Urchin-Barren",
    "Reef-BrLfa": "Reef-BrLfa",
    "Reef-Kelp": "Reef-Kelp",
    "Reef-FnEc": "Reef-FnEc",
    "Reef-nEckFo": "Reef-FnEc",
    "Reef-nEckfo": "Reef-FnEc",
    "Unconsolidated": "Unconsolidated",
    "Reef-Partial-BrLfa": "Reef-Partial-BrLfa",
    "Reef-Partial-Urchin-Barren": "Reef-Partial-Urchin-Barren",
    "Reef-Grazed": "Reef-BrLfa",
    "Reef-Vegetated": "Reef-FnEc",
    "Reef-Barren": "Reef-Urchin-Barren",
    "Reef-Partial-Grazed": "Reef-Partial-BrLfa",
    "Reef-Partial-Barren": "Reef-Partial-Urchin-Barren",
}

EXPERTS = ("Arie", "Kelsey", "Celia")

# Expert annotation label -> model label (identity + known aliases)
FRAME_LABEL_REMAP = {
    "Reef-Urchin-Barren": "Reef-Urchin-Barren",
    "Reef-BrLfa": "Reef-BrLfa",
    "Reef-Kelp": "Reef-Kelp",
    "Reef-FnEc": "Reef-FnEc",
    "Unconsolidated": "Unconsolidated",
    # aliases seen elsewhere in the project
    "Reef-Grazed": "Reef-BrLfa",
    "Reef-Vegetated": "Reef-FnEc",
    "Reef-Barren": "Reef-Urchin-Barren",
}

PATCH_LABEL_REMAP = {
    "Carpophyllum": "Carpophyllum",
    "Carpophyllum spp": "Carpophyllum",
    "Carpophyllum spp.": "Carpophyllum",
    "Ecklonia": "Ecklonia",
    "Ecklonia radiata": "Ecklonia",
    "Foliose algae": "Foliose algae",
    "Foliose Algae": "Foliose algae",
    "Other canopy": "Other canopy",
    "Other canopy forming macroalgae": "Other canopy",
    "Unconsolidated": "Unconsolidated",
    "Urchin": "Urchin",
    "BrLfa": "BrLfa",
    "grazed rock": "BrLfa",
}


class LayerScale(layers.Layer):
    """Required when loading ConvNeXt patch_classifier.h5."""

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(self.init_values * tf.ones((self.projection_dim,)))

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {"init_values": self.init_values, "projection_dim": self.projection_dim}
        )
        return cfg


def load_annotations(json_path: Path) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    frames = data.get("frames", {})
    if not frames:
        raise SystemExit(f"No frames found in {json_path}")
    return frames


def _agreement_mask(df: pd.DataFrame) -> pd.Series:
    if "Check" not in df.columns:
        raise SystemExit("Habitat_7_Class.xlsx must contain a Check column")
    check = df["Check"]
    if check.dtype == object:
        return check.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    return check.astype(bool)


def load_agreement_image_names(xlsx_path: Path) -> set[str]:
    """Image filenames where Arie/Kelsey/Celia agree (Check == True)."""
    if not xlsx_path.is_file():
        raise SystemExit(
            f"Missing agreement spreadsheet: {xlsx_path}\n"
            "Evaluation requires Habitat_7_Class.xlsx "
            "with a Check column (True = all three experts agree)."
        )
    df = pd.read_excel(xlsx_path)
    if "Image" not in df.columns:
        raise SystemExit(f"{xlsx_path} must contain an Image column")
    agree_mask = _agreement_mask(df)
    names = {
        str(name).strip()
        for name in df.loc[agree_mask, "Image"].tolist()
        if pd.notna(name)
    }
    if not names:
        raise SystemExit(f"No Check==True rows found in {xlsx_path}")
    return names


def remap_rule_label(label: str) -> str | None:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    key = str(label).strip()
    if not key or key.lower() == "nan":
        return None
    return RULE_LABEL_REMAP.get(key, key)


def load_agreement_rule_ground_truth(xlsx_path: Path) -> dict[str, str]:
    """
    Map image filename -> agreed 7-class label from Habitat_7_Class.xlsx.
    Uses Arie when Check==True (Arie/Kelsey/Celia are identical on agreement rows).
    """
    if not xlsx_path.is_file():
        raise SystemExit(f"Missing agreement spreadsheet: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    for col in ("Image", "Arie", "Check"):
        if col not in df.columns:
            raise SystemExit(f"{xlsx_path} must contain column: {col}")
    agree = df.loc[_agreement_mask(df)]
    out: dict[str, str] = {}
    skipped = 0
    for _, row in agree.iterrows():
        name = str(row["Image"]).strip()
        mapped = remap_rule_label(row["Arie"])
        if mapped is None:
            skipped += 1
            continue
        out[name] = mapped
    if not out:
        raise SystemExit(f"No usable agreement labels in {xlsx_path}")
    if skipped:
        print(f"  rule GT: skipped {skipped} agreement row(s) with empty/unmapped labels")
    return out


def load_expert_rule_ground_truth(xlsx_path: Path) -> dict[str, dict[str, str]]:
    """Map expert -> (image filename -> 7-class label) over ALL rows of Habitat_7_Class.xlsx."""
    if not xlsx_path.is_file():
        raise SystemExit(f"Missing agreement spreadsheet: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    for col in ("Image",) + EXPERTS:
        if col not in df.columns:
            raise SystemExit(f"{xlsx_path} must contain column: {col}")
    out: dict[str, dict[str, str]] = {}
    for expert in EXPERTS:
        labels: dict[str, str] = {}
        skipped = 0
        for _, row in df.iterrows():
            if pd.isna(row["Image"]):
                continue
            mapped = remap_rule_label(row[expert])
            if mapped is None:
                skipped += 1
                continue
            labels[str(row["Image"]).strip()] = mapped
        if skipped:
            print(f"  rule GT ({expert}): skipped {skipped} row(s) with empty/unmapped labels")
        out[expert] = labels
    return out


def filter_frames_to_agreement(frames: dict, agree_names: set[str]) -> dict:
    """Keep only JSON frames whose filenames are in the expert-agreement set."""
    agree_lower = {n.lower(): n for n in agree_names}
    filtered = {}
    for name, ann in frames.items():
        key = name.lower()
        if key in agree_lower:
            filtered[name] = ann
    return filtered


def resolve_image_path(images_dir: Path, name: str) -> Path | None:
    direct = images_dir / name
    if direct.is_file():
        return direct
    want = name.lower()
    for entry in images_dir.iterdir():
        if entry.is_file() and entry.name.lower() == want:
            return entry
    return None


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def remap_frame_label(label: str) -> str | None:
    if label is None:
        return None
    key = str(label).strip()
    mapped = FRAME_LABEL_REMAP.get(key, key)
    return mapped if mapped in C_TO_I_FRAME else None


def remap_patch_label(label: str) -> str | None:
    """Map expert point label to model class; skip undefined / unknown labels."""
    if label is None:
        return None
    key = str(label).strip()
    if not key:
        return None
    # Do not evaluate undefined / unscored points in the patch classifier.
    if key == UNDEFINED_POINT or key.lower().startswith("undefined"):
        return None
    mapped = PATCH_LABEL_REMAP.get(key, key)
    return mapped if mapped in C_TO_I_PATCH else None


def extract_patch_at_center(image_rgb: np.ndarray, cx: int, cy: int, patch_stride: int) -> np.ndarray | None:
    """Crop patch_stride x patch_stride cell whose center is (cx, cy)."""
    half = patch_stride // 2
    x0 = int(cx) - half
    y0 = int(cy) - half
    x1 = x0 + patch_stride
    y1 = y0 + patch_stride
    h, w = image_rgb.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    patch = image_rgb[y0:y1, x0:x1]
    if patch.shape[0] != patch_stride or patch.shape[1] != patch_stride:
        return None
    return patch


def load_frame_classifier(path: Path):
    if not path.is_file():
        raise SystemExit(f"Frame classifier not found: {path}")
    return tf.keras.models.load_model(str(path), compile=False)


def load_patch_classifier(path: Path):
    if not path.is_file():
        raise SystemExit(f"Patch classifier not found: {path}")
    custom_objects = {"LayerScale": LayerScale, "F1Score": tfa.metrics.F1Score}
    try:
        return tf.keras.models.load_model(
            str(path), compile=False, custom_objects=custom_objects
        )
    except Exception:
        return tf.keras.models.load_model(str(path), compile=False)


def load_urchin_detector():
    return GetYoloModel()


def predict_frame_probs(model, image_rgb: np.ndarray) -> np.ndarray:
    input_size = int(model.input.shape[1])
    image = tf.convert_to_tensor(image_rgb)
    image = tf.image.resize(image, (input_size, input_size))
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)
    # Same preprocess as get_evaluation_patch_frame.py (model_name="inception")
    image = preprocess_input_inception_v3(image)
    return model.predict(image, verbose=0)[0]


def predict_frame_label(model, image_rgb: np.ndarray) -> str:
    return I_TO_C_FRAME[int(np.argmax(predict_frame_probs(model, image_rgb)))]


def predict_patch_labels(
    model,
    patches_rgb: list[np.ndarray],
    batch_size: int = 16,
    already_resized: bool = False,
    return_confidence: bool = False,
):
    """
    Predict patch classes. Expects RGB patches; resizes on CPU unless already_resized.
    If return_confidence=True, returns (labels, max_probs) else labels only.
    """
    if not patches_rgb:
        return ([], []) if return_confidence else []
    input_size = int(model.input.shape[1])
    if already_resized:
        resized = np.stack(patches_rgb, axis=0).astype(np.float32)
    else:
        resized = np.stack(
            [
                cv2.resize(p, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                for p in patches_rgb
            ],
            axis=0,
        ).astype(np.float32)
    tensor = preprocess_input_convnext(resized)
    probs = model.predict(tensor, batch_size=min(batch_size, len(patches_rgb)), verbose=0)
    idxs = np.argmax(probs, axis=1)
    labels = [I_TO_C_PATCH[int(i)] for i in idxs]
    if return_confidence:
        return labels, np.max(probs, axis=1).astype(float).tolist()
    return labels


def predict_points(
    model, image_rgb: np.ndarray, frame_ann: dict, batch_size: int = 16
) -> tuple[list[str | None], list[float | None]]:
    """
    Predict every annotator center of frame_ann (the --patch grid).
    Returns (labels, max_probs) aligned 1:1 with frame_ann['points'];
    None where the crop falls outside the image or patch_stride is missing.
    """
    points = frame_ann.get("points") or []
    patch_stride = int(frame_ann.get("patch_stride") or 0)
    labels: list[str | None] = [None] * len(points)
    confs: list[float | None] = [None] * len(points)
    if patch_stride <= 0:
        return labels, confs
    input_size = int(model.input.shape[1])
    batch: list[np.ndarray] = []
    idx: list[int] = []
    for j, point in enumerate(points):
        patch = extract_patch_at_center(
            image_rgb, int(point["x"]), int(point["y"]), patch_stride
        )
        if patch is None:
            continue
        batch.append(
            cv2.resize(patch, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        )
        idx.append(j)
    if batch:
        preds, cs = predict_patch_labels(
            model, batch, batch_size=batch_size, already_resized=True, return_confidence=True
        )
        for j, lab, conf in zip(idx, preds, cs):
            labels[j] = lab
            confs[j] = conf
    return labels, confs


def patch_ratios_from_predictions(
    labels: list[str | None],
    confs: list[float | None],
    threshold: float = PATCH_CONF_THRESHOLD,
) -> dict[str, float]:
    """
    Habibot patch percentages (demo.get_patch_level_prediction_ratios), computed on
    the annotator-center predictions instead of Habibot's own grid:
      - predictions with max prob < threshold are Unscorable (% of all patches)
      - Urchin patches are dropped from the denominator of the class percentages
    """
    ratios = {
        "Carpophyllum": 0.0,
        "Ecklonia": 0.0,
        "Foliose algae": 0.0,
        "Other canopy": 0.0,
        "Unconsolidated": 0.0,
        "BrLfa": 0.0,
        "Unscorable": 0.0,
    }
    valid = [(lab, c) for lab, c in zip(labels, confs) if lab is not None]
    if not valid:
        return ratios
    confident = [lab for lab, c in valid if c >= threshold]
    non_urchin = [lab for lab in confident if lab != "Urchin"]
    for lab in non_urchin:
        ratios[lab] += 100.0 / len(non_urchin)
    ratios["Unscorable"] = 100.0 * (len(valid) - len(confident)) / len(valid)
    return ratios


class _RuleContext:
    """Minimal stand-in for Habibot so demo's rule function runs without its models."""

    i_to_c_frame = dict(I_TO_C_FRAME)


def apply_habibot_rules(frame_probs, patch_ratios: dict, urchin_boxes) -> tuple[str, str, list]:
    from demo import Habibot

    return Habibot.apply_recommendation_rules(
        _RuleContext(), frame_probs, patch_ratios, urchin_boxes
    )


def predict_urchin_present(detector, image_rgb: np.ndarray) -> bool:
    boxes = detector(image_rgb)
    return len(boxes) > 0


def gt_urchin_present(frame_ann: dict) -> bool:
    """Binary GT: any expert urchin box counts as present (including undefined_urchin)."""
    boxes = frame_ann.get("urchins") or []
    return len(boxes) > 0


def save_classification_outputs(
    y_true: list[str],
    y_pred: list[str],
    out_dir: Path,
    labels: list[str],
    title_prefix: str,
    extra_meta: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    payload = {
        "n_samples": len(y_true),
        "labels": labels,
        "accuracy": report.get("accuracy"),
        "macro_avg": report.get("macro avg"),
        "weighted_avg": report.get("weighted avg"),
        "classification_report": report,
    }
    if extra_meta:
        payload.update(extra_meta)
    report_path = out_dir / "classification_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    # Classification-report heatmap (per-class precision/recall/f1)
    report_data = {
        cls: {
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1-score": report[cls]["f1-score"],
        }
        for cls in labels
        if cls in report
    }
    report_df = pd.DataFrame(report_data).transpose()
    plt.figure(figsize=(8, max(4, 0.45 * len(labels) + 2)))
    sns.heatmap(report_df, annot=True, cmap="Blues", fmt=".2f", cbar=False)
    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    plt.title(f"{title_prefix} classification report")
    plt.tight_layout()
    plt.savefig(out_dir / "classification_report.png")
    plt.close()

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(max(8, 1.6 * len(labels) + 4), max(6, 1.4 * len(labels) + 3)))
    ax = sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Reds",
        cbar=True,
        linewidths=0.3,
        linecolor="black",
    )
    plt.title(f"{title_prefix} confusion matrix", fontsize=14, pad=12)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    print(f"  n_samples: {len(y_true)}")
    print(f"  accuracy:  {report.get('accuracy')}")
    print(f"  macro avg: {report.get('macro avg')}")
    print(f"  saved to:  {out_dir}")
    return report


def _round_metric(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _per_image_record(name: str, y_true: list[str], y_pred: list[str]) -> dict:
    """Accuracy plus macro precision, recall, and F1 for one image."""
    n_scored = len(y_true)
    record = {
        "image": name,
        "n_scored": n_scored,
        "n_correct": 0,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    if n_scored == 0:
        return record
    n_correct = sum(gt == pred for gt, pred in zip(y_true, y_pred))
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    record.update(
        {
            "n_correct": n_correct,
            "accuracy": _round_metric(n_correct / n_scored),
            "precision": _round_metric(precision),
            "recall": _round_metric(recall),
            "f1": _round_metric(f1),
        }
    )
    return record


def save_per_image_accuracy(records: list[dict], out_dir: Path, title: str) -> None:
    """One accuracy per image, plotted for the whole set with mean and SD."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = [r for r in records if r["accuracy"] is not None]
    acc = np.array([r["n_correct"] / r["n_scored"] for r in scored], dtype=float)
    n = int(acc.size)
    mean = float(acc.mean()) if n else None
    std = float(acc.std(ddof=1)) if n > 1 else 0.0
    omitted = len(records) - n

    summary = {
        "n_images": len(records),
        "n_images_scored": n,
        "n_images_unscored": omitted,
        "mean": None if mean is None else round(mean, 4),
        "std": None if mean is None else round(std, 4),
        "min": None if n == 0 else round(float(acc.min()), 4),
        "max": None if n == 0 else round(float(acc.max()), 4),
        "images": records,
    }
    (out_dir / "per_image_accuracy.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if n == 0:
        print("  per-image accuracy: no scored images")
        return

    order = np.argsort(acc)
    ranked = acc[order]
    x = np.arange(1, n + 1)
    lo = max(0.0, mean - std)
    hi = min(1.0, mean + std)

    with plt.rc_context(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "font.size": 11,
        }
    ):
        fig, axes = plt.subplots(
            1, 2, figsize=(12.2, 4.8), sharey=True, gridspec_kw={"width_ratios": [3.2, 0.7]}
        )
        ax = axes[0]
        ax.axhspan(lo, hi, color="#C44536", alpha=0.12, label="Mean ± 1 SD", zorder=0)
        ax.axhline(mean, color="#C44536", linewidth=1.6, label=f"Mean {mean:.3f}", zorder=2)
        ax.plot(x, ranked, color="#1F4E79", linewidth=1.1, zorder=3)
        ax.scatter(x, ranked, s=12, color="#1F4E79", zorder=4)
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Images, sorted from lowest to highest accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend(frameon=False, loc="lower right")
        note = f"SD {std:.3f}    n = {n}"
        if omitted:
            note += f"\n{omitted} image(s) with no scored points omitted"
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D0D0D0"},
        )

        box = axes[1]
        drawn = box.boxplot(
            [acc],
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#C44536",
                "markersize": 6,
            },
            medianprops={"color": "#1F4E79", "linewidth": 1.6},
            whiskerprops={"color": "#1F4E79"},
            capprops={"color": "#1F4E79"},
            flierprops={
                "marker": "o",
                "markersize": 3.5,
                "markerfacecolor": "#1F4E79",
                "markeredgecolor": "none",
                "alpha": 0.45,
            },
        )
        for patch in drawn["boxes"]:
            patch.set(facecolor="#D6E3F0", edgecolor="#1F4E79", linewidth=1.2)
        box.set_title("Accuracy")
        box.set_xticks([1])
        box.set_xticklabels([f"{mean:.2f} ± {std:.2f}"])
        box.tick_params(axis="y", left=False, labelleft=False)

        fig.tight_layout()
        fig.savefig(out_dir / "per_image_accuracy.png", dpi=160)
        plt.close(fig)

    print(f"  per-image accuracy: mean {mean:.4f}  std {std:.4f}  (n={n})")
    print(f"  saved to:  {out_dir / 'per_image_accuracy.png'}")


def _read_image(images_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    img_path = resolve_image_path(images_dir, name)
    if img_path is None:
        print(f"  missing image: {name}")
        return None
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"  could not read: {name}")
        return None
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class FrameComponent:
    """Frame-classifier metrics + debug; shared by --frame and --rule."""

    def __init__(self, out_dir: Path, debug: bool):
        self.out_dir = out_dir
        self.debug = debug
        self.y_true: list[str] = []
        self.y_pred: list[str] = []
        self.skipped = 0
        self.debug_saved = 0

    def add(self, name: str, bgr: np.ndarray, ann: dict, frame_probs) -> str | None:
        gt = remap_frame_label(ann.get("frame_label"))
        if gt is None:
            self.skipped += 1
            return None
        pred = I_TO_C_FRAME[int(np.argmax(frame_probs))]
        self.y_true.append(gt)
        self.y_pred.append(pred)
        if self.debug:
            render_debug_pair(
                image_bgr=bgr,
                frame_ann=ann,
                mode="frame",
                gt_frame_label=gt,
                pred_frame_label=pred,
                pred_point_labels=None,
                pred_urchin_xywh=None,
                out_path=self.out_dir / "debug" / f"{Path(name).stem}.png",
                image_name=name,
            )
            self.debug_saved += 1
        return pred

    def save(self) -> None:
        if not self.y_true:
            raise SystemExit("No frame samples to evaluate.")
        print(f"  frame skipped (no/unmapped GT or unreadable): {self.skipped}")
        if self.debug:
            print(f"  frame debug plots saved: {self.debug_saved}")
        save_classification_outputs(
            self.y_true, self.y_pred, self.out_dir, FRAME_LABELS, title_prefix="Frame"
        )


class PatchComponent:
    """
    Patch-classifier metrics + debug on the annotator grid; shared by --patch and --rule.
      argmax/       — every defined GT point scored by argmax
      conf-cutoff/  — only points with max(prob) >= PATCH_CONF_THRESHOLD
    """

    def __init__(self, out_dir: Path, debug: bool):
        self.argmax_dir = out_dir / "argmax"
        self.conf_dir = out_dir / "conf-cutoff"
        self.debug = debug
        self.y_true_argmax: list[str] = []
        self.y_pred_argmax: list[str] = []
        self.y_true_conf: list[str] = []
        self.y_pred_conf: list[str] = []
        self.n_defined = 0
        self.n_abstained = 0
        self.skipped_points = 0
        self.skipped_images = 0
        self.debug_saved = 0
        self.per_image_argmax: list[dict] = []
        self.per_image_conf: list[dict] = []

    def add(
        self,
        name: str,
        bgr: np.ndarray,
        ann: dict,
        labels: list[str | None],
        confs: list[float | None],
    ) -> list[str]:
        """Accumulate metrics; return conf-cutoff labels aligned with ann['points']."""
        points = ann.get("points") or []
        debug_argmax: list[str] = []
        debug_conf: list[str] = []
        gt_argmax: list[str] = []
        pred_argmax: list[str] = []
        gt_conf: list[str] = []
        pred_conf: list[str] = []
        for point, lab, conf in zip(points, labels, confs):
            if lab is None:
                self.skipped_points += 1
                debug_argmax.append(UNDEFINED_POINT)
                debug_conf.append(UNDEFINED_POINT)
                continue
            confident = conf >= PATCH_CONF_THRESHOLD
            debug_argmax.append(lab)
            debug_conf.append(lab if confident else UNSCORABLE)
            gt = remap_patch_label(point.get("label"))
            if gt is None:
                self.skipped_points += 1
                continue
            self.n_defined += 1
            self.y_true_argmax.append(gt)
            self.y_pred_argmax.append(lab)
            gt_argmax.append(gt)
            pred_argmax.append(lab)
            if confident:
                self.y_true_conf.append(gt)
                self.y_pred_conf.append(lab)
                gt_conf.append(gt)
                pred_conf.append(lab)
            else:
                self.n_abstained += 1
        self.per_image_argmax.append(_per_image_record(name, gt_argmax, pred_argmax))
        self.per_image_conf.append(_per_image_record(name, gt_conf, pred_conf))

        if self.debug:
            stem = Path(name).stem
            render_debug_pair(
                image_bgr=bgr,
                frame_ann=ann,
                mode="patch",
                gt_frame_label=None,
                pred_frame_label=None,
                pred_point_labels=debug_argmax,
                pred_urchin_xywh=None,
                out_path=self.argmax_dir / "debug" / f"{stem}.png",
                image_name=name,
                extra_footer="mode=argmax",
            )
            render_debug_pair(
                image_bgr=bgr,
                frame_ann=ann,
                mode="patch",
                gt_frame_label=None,
                pred_frame_label=None,
                pred_point_labels=debug_conf,
                pred_urchin_xywh=None,
                out_path=self.conf_dir / "debug" / f"{stem}.png",
                image_name=name,
                extra_footer=f"mode=conf-cutoff @{PATCH_CONF_THRESHOLD}",
                extra_legend=[UNSCORABLE],
            )
            self.debug_saved += 1
        return debug_conf

    def save(self) -> None:
        if not self.y_true_argmax:
            raise SystemExit("No patch samples to evaluate.")
        n = self.n_defined
        coverage = (n - self.n_abstained) / n if n else 0.0
        print(f"  patch skipped_images: {self.skipped_images}")
        print(f"  patch skipped_points: {self.skipped_points} (undefined / unmapped / OOB)")
        print(f"  defined GT points: {n}")
        print(
            f"  conf-cutoff: scored={n - self.n_abstained}  abstained={self.n_abstained}  "
            f"coverage={coverage:.4f}"
        )
        if self.debug:
            print(f"  patch debug plots saved (per mode): {self.debug_saved}")

        print("\n  -- Patch argmax --")
        save_classification_outputs(
            self.y_true_argmax,
            self.y_pred_argmax,
            self.argmax_dir,
            PATCH_LABELS,
            title_prefix="Patch (argmax)",
            extra_meta={"mode": "argmax", "n_defined_gt_points": n},
        )
        save_per_image_accuracy(
            self.per_image_argmax, self.argmax_dir, "Patch accuracy per image (argmax)"
        )

        print("\n  -- Patch conf-cutoff --")
        if not self.y_true_conf:
            print(
                f"  WARNING: no points with confidence >= {PATCH_CONF_THRESHOLD}; "
                "skipping conf-cutoff metrics."
            )
            return
        save_classification_outputs(
            self.y_true_conf,
            self.y_pred_conf,
            self.conf_dir,
            PATCH_LABELS,
            title_prefix=f"Patch (conf>={PATCH_CONF_THRESHOLD})",
            extra_meta={
                "mode": "conf-cutoff",
                "confidence_threshold": PATCH_CONF_THRESHOLD,
                "n_defined_gt_points": n,
                "n_scored": len(self.y_true_conf),
                "n_abstained": self.n_abstained,
                "coverage": coverage,
                "note": (
                    "Low-confidence predictions (max prob < threshold) abstain and "
                    "are excluded from metrics; see coverage."
                ),
            },
        )
        save_per_image_accuracy(
            self.per_image_conf,
            self.conf_dir,
            f"Patch accuracy per image (conf ≥ {PATCH_CONF_THRESHOLD})",
        )


class UrchinComponent:
    """Binary urchin-presence metrics + debug; shared by --urchin and --rule."""

    def __init__(self, out_dir: Path, debug: bool):
        self.out_dir = out_dir
        self.debug = debug
        self.y_true: list[str] = []
        self.y_pred: list[str] = []
        self.debug_saved = 0

    def add(self, name: str, bgr: np.ndarray, ann: dict, boxes) -> tuple[str, str]:
        gt = "urchin" if gt_urchin_present(ann) else "no_urchin"
        pred = "urchin" if len(boxes) > 0 else "no_urchin"
        self.y_true.append(gt)
        self.y_pred.append(pred)
        if self.debug:
            render_debug_pair(
                image_bgr=bgr,
                frame_ann=ann,
                mode="urchin",
                gt_frame_label=None,
                pred_frame_label=None,
                pred_point_labels=None,
                pred_urchin_xywh=boxes,
                out_path=self.out_dir / "debug" / f"{Path(name).stem}.png",
                image_name=name,
                extra_footer=f"GT={gt} Pred={pred}",
            )
            self.debug_saved += 1
        return gt, pred

    def save(self) -> None:
        if not self.y_true:
            raise SystemExit("No urchin samples to evaluate.")
        if self.debug:
            print(f"  urchin debug plots saved: {self.debug_saved}")
        save_classification_outputs(
            self.y_true, self.y_pred, self.out_dir, URCHIN_LABELS, title_prefix="Urchin (binary)"
        )


def evaluate_frame(
    frames: dict, images_dir: Path, out_dir: Path, debug: bool = False
) -> None:
    print("\n=== Frame classifier evaluation ===")
    if debug:
        print(f"  Debug plots: {out_dir / 'debug'}")
    model = load_frame_classifier(FRAME_MODEL_PATH)
    comp = FrameComponent(out_dir, debug)

    for name in tqdm(sorted(frames.keys()), desc="frame+debug" if debug else "frame"):
        ann = frames[name]
        if remap_frame_label(ann.get("frame_label")) is None:
            comp.skipped += 1
            continue
        loaded = _read_image(images_dir, name)
        if loaded is None:
            comp.skipped += 1
            continue
        bgr, rgb = loaded
        comp.add(name, bgr, ann, predict_frame_probs(model, rgb))
        del bgr, rgb

    comp.save()


def evaluate_patch(
    frames: dict, images_dir: Path, out_dir: Path, batch_size: int = 16, debug: bool = False
) -> None:
    """
    Write two result trees under out_dir:
      argmax/       — all defined GT points, predicted by argmax
      conf-cutoff/  — only points with max(prob) >= PATCH_CONF_THRESHOLD
    Debug (if enabled) is written under each tree's debug/ folder.
    """
    comp = PatchComponent(out_dir, debug)
    print("\n=== Patch classifier evaluation ===")
    print(f"  argmax      -> {comp.argmax_dir}")
    print(f"  conf-cutoff -> {comp.conf_dir}  (threshold={PATCH_CONF_THRESHOLD})")
    if debug:
        print(f"  Debug plots: {comp.argmax_dir / 'debug'} and {comp.conf_dir / 'debug'}")
    model = load_patch_classifier(PATCH_MODEL_PATH)

    for i, name in enumerate(tqdm(sorted(frames.keys()), desc="patch+debug" if debug else "patch")):
        ann = frames[name]
        if int(ann.get("patch_stride") or 0) <= 0:
            comp.skipped_images += 1
            continue
        loaded = _read_image(images_dir, name)
        if loaded is None:
            comp.skipped_images += 1
            continue
        bgr, rgb = loaded
        labels, confs = predict_points(model, rgb, ann, batch_size=batch_size)
        comp.add(name, bgr, ann, labels, confs)
        del bgr, rgb
        if (i + 1) % 25 == 0:
            gc.collect()

    comp.save()


def evaluate_urchin(
    frames: dict, images_dir: Path, out_dir: Path, debug: bool = False
) -> None:
    print("\n=== Urchin detector evaluation (binary presence) ===")
    if debug:
        print(f"  Debug plots: {out_dir / 'debug'}")
    detector = load_urchin_detector()
    comp = UrchinComponent(out_dir, debug)
    skipped = 0

    for name in tqdm(sorted(frames.keys()), desc="urchin+debug" if debug else "urchin"):
        ann = frames[name]
        loaded = _read_image(images_dir, name)
        if loaded is None:
            skipped += 1
            continue
        bgr, rgb = loaded
        comp.add(name, bgr, ann, detector(rgb))
        del bgr, rgb

    print(f"  skipped: {skipped}")
    comp.save()


# Fixed BGR colors so the same class name is always the same color (GT & Pred, all modes)
URCHIN_BOX_COLOR_BGR = (0, 0, 220)
CLASS_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    # Frame / rule habitat classes
    "Reef-Urchin-Barren": (180, 119, 31),
    "Reef-BrLfa": (14, 127, 255),
    "Reef-Kelp": (44, 160, 44),
    "Reef-FnEc": (40, 39, 214),
    "Unconsolidated": (189, 103, 148),
    "Reef-Partial-BrLfa": (75, 86, 140),
    "Reef-Partial-Urchin-Barren": (194, 119, 227),
    # Patch classes (distinct from habitat hues where possible)
    "Carpophyllum": (255, 144, 30),
    "Ecklonia": (0, 165, 255),
    "Foliose algae": (35, 199, 112),
    "Other canopy": (36, 28, 237),
    "Urchin": (60, 76, 180),
    "BrLfa": (200, 130, 230),
    UNDEFINED_POINT: (0, 204, 255),
    UNSCORABLE: (160, 160, 160),  # grey — Habibot-style abstain
    "urchin box": URCHIN_BOX_COLOR_BGR,
}
_FALLBACK_PALETTE_BGR = [
    (127, 127, 127),
    (34, 189, 188),
    (207, 190, 23),
    (100, 100, 255),
    (180, 180, 60),
]


def _canonical_label(label: str) -> str:
    key = str(label).strip()
    if key.lower().startswith("undefined"):
        return UNDEFINED_POINT
    if key == "urchin box":
        return key
    return (
        PATCH_LABEL_REMAP.get(key)
        or FRAME_LABEL_REMAP.get(key)
        or RULE_LABEL_REMAP.get(key)
        or key
    )


def _label_color_bgr(label: str, labels: list[str] | None = None) -> tuple[int, int, int]:
    """Same class name -> same BGR color on GT and Pred (and across debug modes)."""
    key = _canonical_label(label)
    if key in CLASS_COLOR_BGR:
        return CLASS_COLOR_BGR[key]
    # Fall back stably if an unexpected label appears
    idx = abs(hash(key)) % len(_FALLBACK_PALETTE_BGR)
    return _FALLBACK_PALETTE_BGR[idx]


def _fit_panel_scale(
    image_bgr: np.ndarray, max_w: int, max_h: int
) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        resized = cv2.resize(
            image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        return resized, scale
    return image_bgr.copy(), 1.0


def _draw_grid(out: np.ndarray, patch_stride: int, thickness: int = 3) -> None:
    if patch_stride <= 0:
        return
    gh, gw = out.shape[:2]
    color = (40, 220, 220)
    for x in range(0, gw + 1, max(1, patch_stride)):
        cv2.line(out, (x, 0), (x, gh), color, thickness, cv2.LINE_AA)
    for y in range(0, gh + 1, max(1, patch_stride)):
        cv2.line(out, (0, y), (gw, y), color, thickness, cv2.LINE_AA)


def _draw_points(
    out: np.ndarray,
    points: list[tuple[int, int, str]],
    legend_labels: list[str],
    radius: int,
) -> None:
    for x, y, lab in points:
        color = _label_color_bgr(lab, legend_labels)
        cv2.circle(out, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), radius + 2, (0, 0, 0), 2, cv2.LINE_AA)


def _draw_urchin_xyxy(
    out: np.ndarray, boxes_xyxy: list[tuple[int, int, int, int]], thickness: int = 3
) -> None:
    for x1, y1, x2, y2 in boxes_xyxy:
        cv2.rectangle(
            out, (x1, y1), (x2, y2), URCHIN_BOX_COLOR_BGR, thickness, cv2.LINE_AA
        )


def _draw_legend_column(
    height: int,
    legend_labels: list[str],
    title: str = "Legend (shared)",
    width: int = 340,
) -> np.ndarray:
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    n = max(1, len(legend_labels))
    row_h = max(22, min(36, (height - 90) // n))
    font = max(0.45, min(0.75, row_h / 42.0))
    y = 36
    cv2.putText(
        panel, title, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA
    )
    y += 34
    for lab in legend_labels:
        color = _label_color_bgr(lab, legend_labels)
        cv2.rectangle(panel, (14, y - 18), (42, y + 4), color, -1, cv2.LINE_AA)
        cv2.rectangle(panel, (14, y - 18), (42, y + 4), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            panel,
            lab,
            (54, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        y += row_h
        if y > height - 12:
            break
    return panel


def _header_bar(
    image_bgr: np.ndarray, title: str, subtitle: str, font_scale: float = 1.1
) -> np.ndarray:
    header_h = 100
    h, w = image_bgr.shape[:2]
    canvas = np.full((h + header_h, w, 3), 24, dtype=np.uint8)
    canvas[header_h:] = image_bgr
    cv2.putText(
        canvas,
        title,
        (14, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    # Shrink subtitle if very long so it stays readable
    sub_scale = max(0.55, min(font_scale - 0.2, 0.95))
    if len(subtitle) > 70:
        sub_scale = max(0.5, sub_scale - 0.15)
    cv2.putText(
        canvas,
        subtitle,
        (14, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        sub_scale,
        (200, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_debug_side_by_side(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    legend_labels: list[str],
    out_path: Path,
    footer: str | list[str],
    match: bool | None = None,
) -> None:
    """Compose GT | Pred | shared legend. Panels must already be same display size."""
    footer_lines = [footer] if isinstance(footer, str) else list(footer)
    th = max(left_bgr.shape[0], right_bgr.shape[0])
    tw = max(left_bgr.shape[1], right_bgr.shape[1])

    def pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == th and img.shape[1] == tw:
            return img
        canvas = np.full((th, tw, 3), 24, dtype=np.uint8)
        canvas[: img.shape[0], : img.shape[1]] = img
        return canvas

    left, right = pad(left_bgr), pad(right_bgr)
    legend = _draw_legend_column(th, legend_labels)
    gap = 12
    footer_h = 22 + 36 * len(footer_lines)
    total_w = left.shape[1] + gap + right.shape[1] + gap + legend.shape[1]
    canvas = np.full((th + footer_h, total_w, 3), 24, dtype=np.uint8)
    x = 0
    canvas[:th, x : x + left.shape[1]] = left
    x += left.shape[1] + gap
    canvas[:th, x : x + right.shape[1]] = right
    x += right.shape[1] + gap
    canvas[:th, x : x + legend.shape[1]] = legend

    color = (200, 220, 255)
    if match is True:
        color = (120, 220, 120)
    elif match is False:
        color = (80, 80, 255)
    for k, line in enumerate(footer_lines):
        cv2.putText(
            canvas,
            line,
            (14, th + 38 + 36 * k),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color if k == 0 else (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def _lookup_frame(frames: dict, name: str) -> dict:
    if name in frames:
        return frames[name]
    for k, v in frames.items():
        if k.lower() == name.lower():
            return v
    return {}


def _display_point_label(raw_label) -> str:
    """Normalize GT point labels to the shared legend vocabulary."""
    if raw_label is None:
        return UNDEFINED_POINT
    key = str(raw_label).strip()
    if not key or key.lower().startswith("undefined"):
        return UNDEFINED_POINT
    mapped = PATCH_LABEL_REMAP.get(key, key)
    return mapped if mapped in C_TO_I_PATCH else key


def render_debug_pair(
    image_bgr: np.ndarray,
    frame_ann: dict,
    mode: str,
    gt_frame_label: str | None,
    pred_frame_label: str | None,
    pred_point_labels: list[str] | None,
    pred_urchin_xywh: list | None,
    out_path: Path,
    image_name: str,
    extra_footer: str = "",
    extra_legend: list[str] | None = None,
    left_subtitle: str | None = None,
    right_subtitle: str | None = None,
    footer_lines: list[str] | None = None,
    panel_w: int = 900,
    panel_h: int = 760,
) -> None:
    """
    Build GT|Pred on the SAME Test-Frame-365 image and annotator patch centers.
    Overlays are drawn AFTER downscaling so grid/points stay thick and readable.
    mode: 'frame' | 'patch' | 'urchin' | 'rule'
    pred_point_labels must be 1:1 with frame_ann['points'] when mode uses patches.
    """
    frame_ann = frame_ann or {}
    patch_stride = int(frame_ann.get("patch_stride") or 0)
    points = frame_ann.get("points") or []
    urchins = frame_ann.get("urchins") or []

    show_frame = mode in ("frame", "rule")
    show_patch = mode in ("patch", "rule")
    show_urchin = mode in ("urchin", "rule")

    if mode == "frame":
        legend_labels = list(FRAME_LABELS)
        font_scale = 1.35
    elif mode == "patch":
        legend_labels = list(PATCH_LABELS) + [UNDEFINED_POINT]
        font_scale = 1.05
    elif mode == "urchin":
        legend_labels = ["urchin box"]
        font_scale = 1.25
    else:
        # rule: habitat labels are text-only in the header — legend is patches + urchin
        legend_labels = list(PATCH_LABELS) + [UNDEFINED_POINT, "urchin box"]
        font_scale = 0.95
    if extra_legend:
        legend_labels = list(dict.fromkeys(legend_labels + list(extra_legend)))

    if show_patch:
        if pred_point_labels is None or len(pred_point_labels) != len(points):
            raise RuntimeError(
                f"Debug patch count mismatch for {image_name}: "
                f"GT={len(points)} "
                f"pred={0 if pred_point_labels is None else len(pred_point_labels)}. "
                "Pred must use the same annotator centers as GT."
            )

    # Downscale ONCE; draw overlays at display resolution (visible thick grid)
    base, scale = _fit_panel_scale(image_bgr, panel_w, panel_h)
    stride_d = max(1, int(round(patch_stride * scale))) if patch_stride > 0 else 0
    # Drawn on already-resized panel → keep lines thick enough to see
    grid_thickness = 4
    point_radius = max(7, min(16, stride_d // 8 if stride_d else 10))
    box_thickness = 4

    def scaled_pts(labels: list[str]) -> list[tuple[int, int, str]]:
        out = []
        for i, p in enumerate(points):
            out.append(
                (
                    int(round(int(p["x"]) * scale)),
                    int(round(int(p["y"]) * scale)),
                    labels[i],
                )
            )
        return out

    gt_point_labels = [_display_point_label(p.get("label")) for p in points]

    # --- GT panel ---
    left = base.copy()
    if show_patch:
        _draw_grid(left, stride_d, thickness=grid_thickness)
        _draw_points(left, scaled_pts(gt_point_labels), legend_labels, point_radius)
    if show_urchin:
        gt_boxes = [
            (
                int(round(u["xtl"] * scale)),
                int(round(u["ytl"] * scale)),
                int(round(u["xbr"] * scale)),
                int(round(u["ybr"] * scale)),
            )
            for u in urchins
        ]
        _draw_urchin_xyxy(left, gt_boxes, thickness=box_thickness)

    if mode == "frame":
        left_sub = f"Frame GT: {gt_frame_label}"
    elif mode == "patch":
        left_sub = f"Patches GT: {len(points)}  (stride={patch_stride})"
    elif mode == "urchin":
        left_sub = f"Urchin boxes GT: {len(urchins)}"
    else:
        left_sub = (
            f"Habitat GT: {gt_frame_label} | patches: {len(points)} | "
            f"urchins: {len(urchins)}"
        )
    if left_subtitle is not None:
        left_sub = left_subtitle
    left = _header_bar(left, "GT", left_sub, font_scale=font_scale)

    # --- Pred panel (identical base image + identical centers) ---
    right = base.copy()
    if show_patch:
        _draw_grid(right, stride_d, thickness=grid_thickness)
        _draw_points(right, scaled_pts(list(pred_point_labels)), legend_labels, point_radius)
    if show_urchin:
        pred_boxes = []
        for box in pred_urchin_xywh or []:
            cx, cy, bw, bh = [float(v) for v in box[:4]]
            pred_boxes.append(
                (
                    int(round((cx - bw / 2) * scale)),
                    int(round((cy - bh / 2) * scale)),
                    int(round((cx + bw / 2) * scale)),
                    int(round((cy + bh / 2) * scale)),
                )
            )
        _draw_urchin_xyxy(right, pred_boxes, thickness=box_thickness)

    if mode == "frame":
        right_sub = f"Frame Pred: {pred_frame_label}"
    elif mode == "patch":
        right_sub = f"Patches Pred: {len(points)}  (same centers)"
    elif mode == "urchin":
        right_sub = f"Urchin boxes Pred: {len(pred_urchin_xywh or [])}"
    else:
        right_sub = (
            f"Habitat Pred: {pred_frame_label} | patches: {len(points)} | "
            f"urchins: {len(pred_urchin_xywh or [])}"
        )
    if right_subtitle is not None:
        right_sub = right_subtitle
    right = _header_bar(right, "Prediction", right_sub, font_scale=font_scale)

    match = None
    if show_frame and gt_frame_label is not None and pred_frame_label is not None:
        match = gt_frame_label == pred_frame_label
    footer = f"{image_name}  |  image {image_bgr.shape[1]}x{image_bgr.shape[0]}"
    if show_patch:
        footer += f"  |  points GT=Pred={len(points)}"
    if show_frame:
        footer += f"  |  GT: {gt_frame_label} vs Pred: {pred_frame_label}"
        footer += "  |  MATCH" if match else "  |  MISMATCH"
    if extra_footer:
        footer += f"  |  {extra_footer}"

    save_debug_side_by_side(
        left, right, legend_labels, out_path, [footer] + list(footer_lines or []), match=match
    )


def evaluate_rule(
    rule_gt: dict[str, str],
    frames: dict,
    images_dir: Path,
    out_dir: Path,
    debug: bool = False,
    batch_size: int = 16,
    expert_gt: dict[str, dict[str, str]] | None = None,
) -> None:
    """
    Rule-based habitat classifier vs Habitat_7_Class.xlsx, built from the SAME component
    predictions as --frame / --patch / --urchin (original Test-Frame-365 image, annotator
    patch centers). demo.Habibot.apply_recommendation_rules supplies the rule logic only.

    Outputs out_dir/ (rule metrics + predictions.csv on the agreed images); with debug,
    out_dir/debug/ shows everything that fed the rule on one figure. With expert_gt,
    out_dir/<expert>/ also holds metrics vs that expert's labels on every image they
    labelled (no debug plots). Component metrics are not written here — use
    --frame / --patch / --urchin, which share the same prediction code.
    """
    expert_gt = expert_gt or {}
    print("\n=== Rule-based classifier evaluation ===")
    print("  GT: Habitat_7_Class.xlsx (Check=True agreed labels)")
    for expert, expert_labels in expert_gt.items():
        print(f"  GT ({expert}): {len(expert_labels)} images -> {out_dir / expert}")
    print(
        "  Pred: demo.Habibot.apply_recommendation_rules on --frame/--patch/--urchin "
        "predictions (annotator grid, original image)"
    )
    if debug:
        print(f"  Debug plots: {out_dir / 'debug'}")

    frame_model = load_frame_classifier(FRAME_MODEL_PATH)
    patch_model = load_patch_classifier(PATCH_MODEL_PATH)
    detector = load_urchin_detector()

    y_true: list[str] = []
    y_pred: list[str] = []
    rows: list[dict] = []
    expert_true: dict[str, list[str]] = {e: [] for e in expert_gt}
    expert_pred: dict[str, list[str]] = {e: [] for e in expert_gt}
    skipped_no_json = 0
    skipped_image = 0
    pred_outside = 0
    debug_saved = 0

    names = set(rule_gt)
    for expert_labels in expert_gt.values():
        names |= set(expert_labels)

    desc = "rule+debug" if debug else "rule"
    for i, name in enumerate(tqdm(sorted(names, key=str.lower), desc=desc)):
        ann = _lookup_frame(frames, name)
        if not ann:
            print(f"  skip (no JSON annotation): {name}")
            skipped_no_json += 1
            continue
        loaded = _read_image(images_dir, name)
        if loaded is None:
            skipped_image += 1
            continue
        bgr, rgb = loaded

        frame_probs = predict_frame_probs(frame_model, rgb)
        labels, confs = predict_points(patch_model, rgb, ann, batch_size=batch_size)
        urchin_boxes = detector(rgb)
        ratios = patch_ratios_from_predictions(labels, confs)
        frame_after_urchin, recommendation, rule_numbers = apply_habibot_rules(
            frame_probs, ratios, urchin_boxes
        )
        pred = remap_rule_label(recommendation) or str(recommendation).strip()
        if pred not in RULE_LABELS:
            pred_outside += 1
        for expert, expert_labels in expert_gt.items():
            if name in expert_labels:
                expert_true[expert].append(expert_labels[name])
                expert_pred[expert].append(pred)

        if name not in rule_gt:
            del bgr, rgb
            if (i + 1) % 25 == 0:
                gc.collect()
            continue
        gt = rule_gt[name]
        y_true.append(gt)
        y_pred.append(pred)

        frame_pred = I_TO_C_FRAME[int(np.argmax(frame_probs))]
        conf_labels = [
            UNDEFINED_POINT if lab is None
            else (lab if conf >= PATCH_CONF_THRESHOLD else UNSCORABLE)
            for lab, conf in zip(labels, confs)
        ]
        urchin_gt = "urchin" if gt_urchin_present(ann) else "no_urchin"
        urchin_pred = "urchin" if len(urchin_boxes) > 0 else "no_urchin"
        frame_gt = remap_frame_label(ann.get("frame_label"))
        frame_conf = float(np.max(frame_probs))

        rows.append(
            {
                "image": name,
                "rule_gt": gt,
                "rule_pred": pred,
                "rule_match": gt == pred,
                "rules": " ".join(rule_numbers),
                "frame_gt_json": frame_gt,
                "frame_pred": I_TO_C_FRAME[int(np.argmax(frame_probs))],
                "frame_conf": round(frame_conf, 4),
                "frame_after_urchin_rules": frame_after_urchin,
                "urchin_gt": urchin_gt,
                "urchin_pred": urchin_pred,
                "n_urchin_boxes_pred": len(urchin_boxes),
                "n_patches_predicted": sum(lab is not None for lab in labels),
                **{f"pct_{k}": round(v, 2) for k, v in ratios.items()},
            }
        )

        if debug:
            canopy = sum(
                ratios[k] for k in ("Carpophyllum", "Ecklonia", "Other canopy", "Foliose algae")
            )
            render_debug_pair(
                image_bgr=bgr,
                frame_ann=ann,
                mode="rule",
                gt_frame_label=gt,
                pred_frame_label=pred,
                pred_point_labels=conf_labels,
                pred_urchin_xywh=urchin_boxes,
                out_path=out_dir / "debug" / f"{Path(name).stem}.png",
                image_name=name,
                extra_legend=[UNSCORABLE],
                left_subtitle=(
                    f"Habitat GT: {gt} | Frame GT: {frame_gt} | "
                    f"urchins: {len(ann.get('urchins') or [])}"
                ),
                right_subtitle=(
                    f"Rule Pred: {pred} | Frame Pred: {frame_pred} ({frame_conf:.2f}) | "
                    f"urchins: {len(urchin_boxes)}"
                ),
                footer_lines=[
                    f"Frame after urchin rules: {frame_after_urchin}  |  rules: "
                    f"{' '.join(rule_numbers)}  |  urchin GT={urchin_gt} Pred={urchin_pred}",
                    f"Patch % (conf>={PATCH_CONF_THRESHOLD}, urchin excluded): "
                    f"BrLfa {ratios['BrLfa']:.1f}  canopy {canopy:.1f}  "
                    f"Uncons {ratios['Unconsolidated']:.1f}  Unscorable {ratios['Unscorable']:.1f}",
                ],
            )
            debug_saved += 1

        del bgr, rgb
        if (i + 1) % 25 == 0:
            gc.collect()

    if not y_true:
        raise SystemExit("No rule-based samples to evaluate.")
    print(f"  skipped (no JSON annotation): {skipped_no_json}")
    print(f"  skipped (image missing/unreadable): {skipped_image}")
    if pred_outside:
        print(f"  predictions outside the 7 excel classes: {pred_outside}")
    if debug:
        print(f"  rule debug plots saved: {debug_saved}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)

    def _labels_for(t: list[str], p: list[str]) -> list[str]:
        out = list(RULE_LABELS)
        for lab in sorted(set(t) | set(p)):
            if lab not in out:
                out.append(lab)
        return out

    meta = {
        "pipeline": (
            "frame/patch/urchin predictions on the annotator grid "
            "(same as --frame/--patch/--urchin) + demo.apply_recommendation_rules"
        ),
        "patch_conf_threshold": PATCH_CONF_THRESHOLD,
    }
    print("\n  -- Rule-based --")
    save_classification_outputs(
        y_true,
        y_pred,
        out_dir,
        _labels_for(y_true, y_pred),
        title_prefix="Rule-based",
        extra_meta={"ground_truth": "agreed labels (Check=True)", **meta},
    )

    for expert in expert_gt:
        t, p = expert_true[expert], expert_pred[expert]
        if not t:
            print(f"  no samples for expert {expert}")
            continue
        print(f"\n  -- Rule-based vs {expert} (all images) --")
        save_classification_outputs(
            t,
            p,
            out_dir / expert,
            _labels_for(t, p),
            title_prefix=f"Rule-based vs {expert}",
            extra_meta={"ground_truth": f"{expert} labels (all images)", **meta},
        )


def _round_minutes(value) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _task_minutes(patches, urchins, frames, n_images: int) -> dict:
    total = None
    if None not in (patches, urchins, frames):
        total = patches + urchins + frames
    minutes = {
        "Patches": _round_minutes(patches),
        "Urchins": _round_minutes(urchins),
        "Frames": _round_minutes(frames),
        "Total": _round_minutes(total),
    }
    per_image = {
        key: None if val is None or not n_images else round(val / n_images, 4)
        for key, val in minutes.items()
    }
    return {"minutes": minutes, "minutes_per_image": per_image}


def evaluate_time(time_path: Path, out_path: Path) -> None:
    """Summarise time_keeping.json into totals and per-image minutes, without image names."""
    if not time_path.is_file():
        raise SystemExit(f"Missing time keeping file: {time_path}")
    data = json.loads(time_path.read_text(encoding="utf-8"))
    sets_in = data.get("sets") or []
    if not sets_in:
        raise SystemExit(f"No sets in {time_path}")

    sets_out = []
    timed_n = 0
    sums = {"Patches": 0.0, "Urchins": 0.0, "Frames": 0.0}
    for block in sets_in:
        n_images = int(block["n_images"])
        patches, urchins, frames = block.get("Patches"), block.get("Urchins"), block.get("Frames")
        entry = {"image_set": block["image_set"], "n_images": n_images}
        entry.update(_task_minutes(patches, urchins, frames, n_images))
        sets_out.append(entry)
        if None not in (patches, urchins, frames):
            timed_n += n_images
            sums["Patches"] += patches
            sums["Urchins"] += urchins
            sums["Frames"] += frames

    annotation = {"n_images": timed_n}
    annotation.update(_task_minutes(sums["Patches"], sums["Urchins"], sums["Frames"], timed_n))

    seven = data.get("seven_class") or {}
    seven_n = int(seven.get("n_images") or 0)
    seven_out = {"n_images": seven_n, "minutes": {}, "minutes_per_image": {}}
    for expert in ("Arie", "Kelsey", "Celia"):
        val = seven.get(expert)
        seven_out["minutes"][expert] = _round_minutes(val)
        seven_out["minutes_per_image"][expert] = (
            None if val is None or not seven_n else round(float(val) / seven_n, 4)
        )

    hab = data.get("habibot") or {}
    hab_n = int(hab.get("n_images") or 0)
    hab_out = {"image_set": hab.get("image_set"), "n_images": hab_n}
    hab_out.update(
        _task_minutes(hab.get("Patches"), hab.get("Urchins"), hab.get("Frames"), hab_n)
    )
    note = str(hab.get("note") or "").strip()
    processing = None
    if note:
        try:
            processing = float(note.split()[0])
        except ValueError:
            processing = None
    hab_out["processing_minutes"] = _round_minutes(processing)
    hab_out["processing_minutes_per_image"] = (
        None if processing is None or not hab_n else round(processing / hab_n, 4)
    )

    payload = {
        "n_images": int(data.get("n_images") or sum(s["n_images"] for s in sets_out)),
        "annotation": annotation,
        "sets": sets_out,
        "seven_class": seven_out,
        "habibot": hab_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    ann = annotation["minutes"]
    print("\n=== Time keeping ===")
    print(f"  images: {payload['n_images']}  ({timed_n} with patch/urchin/frame times)")
    print(
        f"  annotation minutes: Patches {ann['Patches']}  Urchins {ann['Urchins']}  "
        f"Frames {ann['Frames']}  Total {ann['Total']}"
    )
    print(f"  per image: {annotation['minutes_per_image']}")
    print(
        "  7-class minutes: "
        + "  ".join(f"{k} {v}" for k, v in seven_out["minutes"].items())
    )
    print(f"  habibot minutes: {hab_out['minutes']}  processing {hab_out['processing_minutes']}")
    print(f"  saved to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Habibot models on Test-Frame-365. "
            "Frame, patch, and urchin use every annotated image. "
            "Rule agreement uses Check=True in Habitat_7_Class.xlsx."
        )
    )
    parser.add_argument("--frame", action="store_true", help="Evaluate frame classifier")
    parser.add_argument("--patch", action="store_true", help="Evaluate patch classifier")
    parser.add_argument(
        "--urchin", action="store_true", help="Evaluate urchin detector (binary presence)"
    )
    parser.add_argument(
        "--rule",
        action="store_true",
        help=(
            "Evaluate the rule-based classifier vs Habitat_7_Class.xlsx agreed labels "
            "(7 classes incl. partials), using the same frame/patch/urchin predictions "
            "as --frame/--patch/--urchin. Also writes rule/<expert>/ metrics vs each "
            "expert's labels on all images"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Save side-by-side GT|Pred debug plots for whichever tasks are selected. "
            "Both panels use the SAME Test-Frame-365 image and annotator patch grid. "
            "--frame/--patch/--urchin/--rule control what is drawn."
        ),
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help=(
            "Summarise time_keeping.json into results/stats_time.json "
            "(overall minutes and per-image minutes, no image names)"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run frame, patch, urchin, then rule evaluations",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to Test-Frame-365.json",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES,
        help="Folder with Test-Frame-365 images",
    )
    parser.add_argument(
        "--agreement-xlsx",
        type=Path,
        default=DEFAULT_AGREEMENT_XLSX,
        help="Excel with Image + Check; --rule agreement uses Check=True rows",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_ROOT,
        help="Root results folder (subdirs frame/patch/urchin/rule)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_frame = args.frame or args.all
    run_patch = args.patch or args.all
    run_urchin = args.urchin or args.all
    run_rule = args.rule or args.all
    run_time = bool(args.time)
    debug = bool(args.debug)

    if not (run_frame or run_patch or run_urchin or run_rule or run_time):
        print("Specify at least one of --frame, --patch, --urchin, --rule, --time, or --all")
        return 2

    json_path = args.json.resolve()
    images_dir = args.images.resolve()
    results_root = args.results.resolve()
    xlsx_path = args.agreement_xlsx.resolve()

    if run_time:
        evaluate_time(DEFAULT_TIME_JSON, results_root / "stats_time.json")
        if not (run_frame or run_patch or run_urchin or run_rule):
            return 0

    if not images_dir.is_dir():
        print(f"Missing images folder: {images_dir}")
        return 2

    if not json_path.is_file():
        print(f"Missing annotations JSON: {json_path}")
        return 2

    # Habibot loaders expect cwd-relative Urchin-Detector / trained_classifiers paths
    os.chdir(REPO_ROOT)
    configure_tf_memory()

    agree_names = load_agreement_image_names(xlsx_path) if run_rule else set()
    print(f"Images: {images_dir}")
    print(f"Results root: {results_root}")
    if debug:
        print("Debug plots enabled (same image + annotator grid on GT and Pred).")

    frames_all = load_annotations(json_path)
    frames = filter_frames_to_agreement(frames_all, agree_names) if run_rule else {}
    print(f"Loaded {len(frames_all)} annotated frames from {json_path}")
    if run_frame or run_patch or run_urchin:
        print(f"Frame/patch/urchin images: {len(frames_all)}")
    if run_rule:
        print(f"Rule agreement images (Check=True): {len(frames)}")
        if not frames:
            print("No frames left after agreement filter.")
            return 2

    if run_frame:
        evaluate_frame(frames_all, images_dir, results_root / "frame", debug=debug)
    if run_patch:
        evaluate_patch(frames_all, images_dir, results_root / "patch", debug=debug)
    if run_urchin:
        evaluate_urchin(frames_all, images_dir, results_root / "urchin", debug=debug)
    if run_rule:
        rule_gt = load_agreement_rule_ground_truth(xlsx_path)
        rule_gt = {
            name: lab
            for name, lab in rule_gt.items()
            if name.lower() in {n.lower() for n in agree_names}
            and resolve_image_path(images_dir, name) is not None
        }
        print(f"Rule-based GT samples: {len(rule_gt)}")
        expert_gt = {
            expert: {
                name: lab
                for name, lab in labels.items()
                if _lookup_frame(frames_all, name)
                and resolve_image_path(images_dir, name) is not None
            }
            for expert, labels in load_expert_rule_ground_truth(xlsx_path).items()
        }
        evaluate_rule(
            rule_gt,
            frames_all,
            images_dir,
            results_root / "rule",
            debug=debug,
            expert_gt=expert_gt,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
