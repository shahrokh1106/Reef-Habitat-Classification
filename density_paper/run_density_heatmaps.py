from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def load_saved_density_data_dict(
    inputs_dir: str | os.PathLike,
    saved_outputs_dir: str | os.PathLike,
    locations: Iterable[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Reconstruct the `combined_dict` (a.k.a. data_dict) produced by `get_density_dicts`,
    using already-saved per-location JSONs in `saved_outputs_dir` and the input CSVs in `inputs_dir`.

    Expected files:
    - inputs_dir/<LOCATION>.csv with columns including '#Label', 'Y', 'X'
    - saved_outputs_dir/<LOCATION>.json with mapping: image_name -> [prediction_label, urchin_count]
    """
    inputs_dir = Path(inputs_dir)
    saved_outputs_dir = Path(saved_outputs_dir)

    if not inputs_dir.exists():
        raise FileNotFoundError(f"inputs_dir not found: {inputs_dir}")
    if not saved_outputs_dir.exists():
        raise FileNotFoundError(f"saved_outputs_dir not found: {saved_outputs_dir}")

    if locations is None:
        csv_paths = sorted(inputs_dir.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No .csv files found in inputs_dir: {inputs_dir}")
        location_names = [p.stem for p in csv_paths]
    else:
        location_names = [str(x) for x in locations]
        if not location_names:
            raise ValueError("locations was provided but empty.")

    combined_dict: Dict[str, pd.DataFrame] = {}

    for location_name in location_names:
        csv_path = inputs_dir / f"{location_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing input CSV for location '{location_name}': {csv_path}"
            )
        json_path = saved_outputs_dir / f"{location_name}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing saved results JSON for location '{location_name}': {json_path}"
            )

        loc_df = pd.read_csv(csv_path).rename(columns={"#Label": "image_name"})
        if "image_name" not in loc_df.columns:
            raise ValueError(f"{csv_path} must contain '#Label' (renamed to 'image_name').")

        # Match `get_density_dicts`: remove the file extension from labels (e.g. 'foo.jpg' -> 'foo')
        loc_df["image_name"] = loc_df["image_name"].astype(str).str.slice(stop=-4)

        with open(json_path, "r", encoding="utf-8") as f:
            results_dict = json.load(f)

        data_list = []
        for image_name, details in results_dict.items():
            # details = [recommendation, len(urchin_boxes)]
            data_list.append([image_name, details[0], details[1]])
        data_df = pd.DataFrame(data_list, columns=["image_name", "predictions", "urchin_count"])

        merged_df = pd.merge(loc_df, data_df, on="image_name", how="inner")
        combined_dict[location_name] = merged_df

    return combined_dict


def get_density_heat_maps_metric_png_v2(
    combined_dict,
    output_path=".",
    png_file="heatmap_maps_v2.png",
):
    """
    V2 static plot: same points as `get_density_heat_maps`, but plotted in a local metric frame (meters)
    relative to the centroid (mean lat/lon). This addresses reviewer comments about lat/lon looking "off"
    and improves interpretability (distances in meters).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from collections import defaultdict

    output_png = os.path.join(output_path, png_file)

    # Collect all valid lat/lon points (Y=lat, X=lon)
    all_points = []
    for df in combined_dict.values():
        if "Y" in df.columns and "X" in df.columns:
            all_points.extend(df[["Y", "X"]].values.tolist())
    if len(all_points) == 0:
        raise ValueError("No points found in combined_dict (expected columns 'Y' and 'X').")

    lats = np.array([p[0] for p in all_points], dtype=float)
    lons = np.array([p[1] for p in all_points], dtype=float)
    if lats.size == 0 or lons.size == 0:
        raise ValueError("No valid lat/lon points to plot.")

    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))

    # Simple local projection (equirectangular) to meters: good for small areas
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * float(np.cos(np.deg2rad(lat0)))

    def _to_m(lat: float, lon: float) -> tuple[float, float]:
        dy = (float(lat) - lat0) * meters_per_deg_lat
        dx = (float(lon) - lon0) * meters_per_deg_lon
        return dx, dy  # (east_m, north_m)

    # Same fixed color mapping per class
    heatmap_colors = {
        "Reef-Urchin-Barren": "red",
        "Reef-Partial-Urchin-Barren": "orange",
        "Unconsolidated": "blue",
        "Reef-FnEc": "#99ff99",
        "Reef-Kelp": "green",
        "Reef-BrLfa": "purple",
        "Reef-Partial-BrLfa": "pink",
    }

    # Normalize labels (match the original function)
    def _normalize_label(lbl: str) -> str:
        if lbl == "Reef-Urchin-Barren (Review)":
            return "Reef-Urchin-Barren"
        if lbl in ("Reef-Partial-Urchin-Barren (Review)",):
            return "Reef-Partial-Urchin-Barren"
        if lbl in ("Reef-Partial-Grazed (Review)", "Reef-Partial-Grazed"):
            return "Reef-Partial-BrLfa"
        if lbl == "Reef-Grazed":
            return "Reef-BrLfa"
        if lbl == "Reef-Vegetated":
            return "Reef-FnEc"
        return lbl

    # 1) Urchin points (size by count)
    urchin_data = []
    for df in combined_dict.values():
        if "urchin_count" in df.columns:
            for _, row in df.iterrows():
                try:
                    cnt = float(row["urchin_count"])
                except Exception:
                    continue
                if np.isfinite(cnt):
                    urchin_data.append([row["Y"], row["X"], cnt])

    ur_e, ur_n, ur_sizes = [], [], []
    i = 0
    if urchin_data:
        for y, x, c in urchin_data:
            if i % 2 == 0:  # keep the original subsampling behavior
                e, n = _to_m(y, x)
                ur_e.append(e)
                ur_n.append(n)
                ur_sizes.append(float(c))
            i += 1
        ur_e = np.array(ur_e, dtype=float)
        ur_n = np.array(ur_n, dtype=float)
        ur_sizes = np.array(ur_sizes, dtype=float)
        if np.any(np.isfinite(ur_sizes)):
            smin, smax = np.nanmin(ur_sizes), np.nanmax(ur_sizes)
            if smax <= smin:
                smax = smin + 1.0
            ur_sizes = 300 + 100.0 * (ur_sizes - smin) / (smax - smin)
        else:
            ur_sizes = np.full_like(ur_e, 40.0)

    # 2) Class points
    class_points_m = defaultdict(list)  # cname -> list[(e_m, n_m)]
    for _, df in combined_dict.items():
        if "predictions" not in df.columns:
            continue
        for _, row in df.iterrows():
            pred = _normalize_label(row["predictions"])
            if pred not in heatmap_colors:
                continue
            e, n = _to_m(row["Y"], row["X"])
            class_points_m[pred].append((e, n))

    # Bounds in meters with small padding
    all_e, all_n = [], []
    for y, x in zip(lats, lons):
        e, n = _to_m(y, x)
        all_e.append(e)
        all_n.append(n)
    all_e = np.array(all_e, dtype=float)
    all_n = np.array(all_n, dtype=float)

    pad_e = max(1.0, 0.02 * max(1e-6, float(all_e.max() - all_e.min())))
    pad_n = max(1.0, 0.02 * max(1e-6, float(all_n.max() - all_n.min())))
    e_min, e_max = float(all_e.min() - pad_e), float(all_e.max() + pad_e)
    n_min, n_max = float(all_n.min() - pad_n), float(all_n.max() + pad_n)

    # Figure sizing (avoid extreme aspect ratios)
    span_n = (n_max - n_min)
    span_e = (e_max - e_min)
    aspect = span_e / max(1e-12, span_n)
    width = 7.0 * min(2.0, max(0.7, aspect))
    height = 7.0

    fig = plt.figure(figsize=(width, height), dpi=400)
    ax = plt.gca()

    if urchin_data and len(ur_e) > 0:
        ax.scatter(ur_e, ur_n, s=ur_sizes, facecolors="black", edgecolors="black", alpha=0.5)

    for cname, pts in class_points_m.items():
        if not pts:
            continue
        color = heatmap_colors[cname]
        ee = [p[0] for p in pts]
        nn = [p[1] for p in pts]
        ax.scatter(ee, nn, s=50, c=color, edgecolors="none", alpha=0.9, label=cname)

    fsize = 18
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(n_min, n_max)
    ax.tick_params(axis="x", labelsize=fsize - 4)
    ax.tick_params(axis="y", labelsize=fsize - 4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Distance East (m)", fontsize=fsize - 3)
    ax.set_ylabel("Distance North (m)", fontsize=fsize - 3)
    ax.set_title(
        "Predicted habitat classes (local metric coordinates)",
        fontsize=fsize + 1,
    )

    class_handles = [
        Patch(facecolor=heatmap_colors[c], edgecolor="none", label=c)
        for c in heatmap_colors.keys()
        if c in class_points_m and len(class_points_m[c]) > 0
    ]
    urchin_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        markerfacecolor="black",
        markersize=6,
        linewidth=0,
        alpha=0.6,
        label="Urchin Count (size)",
    )
    handles = class_handles + ([urchin_handle] if (urchin_data and len(ur_e) > 0) else [])
    if handles:
        ax.legend(handles=handles, loc="best", frameon=True, fontsize=fsize - 3)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)
    print(f"[Matplotlib] V2 (meters) static map saved to {output_png}")


def get_density_heat_maps_metric_png_v3(
    combined_dict,
    output_path=".",
    png_file="heatmap_maps_v3.png",
):
    """
    V3 static plot: local metric frame (meters) with origin at the SW corner (min lat/lon),
    so distances are non-negative (0..max).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from collections import defaultdict

    output_png = os.path.join(output_path, png_file)

    # Collect all valid lat/lon points (Y=lat, X=lon)
    all_points = []
    for df in combined_dict.values():
        if "Y" in df.columns and "X" in df.columns:
            all_points.extend(df[["Y", "X"]].values.tolist())
    if len(all_points) == 0:
        raise ValueError("No points found in combined_dict (expected columns 'Y' and 'X').")

    lats = np.array([p[0] for p in all_points], dtype=float)
    lons = np.array([p[1] for p in all_points], dtype=float)
    if lats.size == 0 or lons.size == 0:
        raise ValueError("No valid lat/lon points to plot.")

    lat_min0 = float(np.min(lats))
    lon_min0 = float(np.min(lons))

    # Use a representative latitude for lon scaling (small area; this is stable)
    lat_ref = float(np.mean(lats))
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * float(np.cos(np.deg2rad(lat_ref)))

    def _to_m_sw(lat: float, lon: float) -> tuple[float, float]:
        north_m = (float(lat) - lat_min0) * meters_per_deg_lat
        east_m = (float(lon) - lon_min0) * meters_per_deg_lon
        return east_m, north_m  # (east_m, north_m), expected >= 0

    heatmap_colors = {
        "Reef-Urchin-Barren": "red",
        "Reef-Partial-Urchin-Barren": "orange",
        "Unconsolidated": "blue",
        "Reef-FnEc": "#99ff99",
        "Reef-Kelp": "green",
        "Reef-BrLfa": "purple",
        "Reef-Partial-BrLfa": "pink",
    }

    def _normalize_label(lbl: str) -> str:
        if lbl == "Reef-Urchin-Barren (Review)":
            return "Reef-Urchin-Barren"
        if lbl in ("Reef-Partial-Urchin-Barren (Review)",):
            return "Reef-Partial-Urchin-Barren"
        if lbl in ("Reef-Partial-Grazed (Review)", "Reef-Partial-Grazed"):
            return "Reef-Partial-BrLfa"
        if lbl == "Reef-Grazed":
            return "Reef-BrLfa"
        if lbl == "Reef-Vegetated":
            return "Reef-FnEc"
        return lbl

    # 1) Urchin points (size by count)
    urchin_data = []
    for df in combined_dict.values():
        if "urchin_count" in df.columns:
            for _, row in df.iterrows():
                try:
                    cnt = float(row["urchin_count"])
                except Exception:
                    continue
                if np.isfinite(cnt):
                    urchin_data.append([row["Y"], row["X"], cnt])

    ur_e, ur_n, ur_sizes = [], [], []
    i = 0
    if urchin_data:
        for y, x, c in urchin_data:
            if i % 2 == 0:  # keep the original subsampling behavior
                e, n = _to_m_sw(y, x)
                ur_e.append(e)
                ur_n.append(n)
                ur_sizes.append(float(c))
            i += 1
        ur_e = np.array(ur_e, dtype=float)
        ur_n = np.array(ur_n, dtype=float)
        ur_sizes = np.array(ur_sizes, dtype=float)
        if np.any(np.isfinite(ur_sizes)):
            smin, smax = np.nanmin(ur_sizes), np.nanmax(ur_sizes)
            if smax <= smin:
                smax = smin + 1.0
            ur_sizes = 300 + 100.0 * (ur_sizes - smin) / (smax - smin)
        else:
            ur_sizes = np.full_like(ur_e, 40.0)

    # 2) Class points
    class_points_m = defaultdict(list)  # cname -> list[(e_m, n_m)]
    for _, df in combined_dict.items():
        if "predictions" not in df.columns:
            continue
        for _, row in df.iterrows():
            pred = _normalize_label(row["predictions"])
            if pred not in heatmap_colors:
                continue
            e, n = _to_m_sw(row["Y"], row["X"])
            class_points_m[pred].append((e, n))

    # Bounds in meters with small padding
    all_e, all_n = [], []
    for y, x in zip(lats, lons):
        e, n = _to_m_sw(y, x)
        all_e.append(e)
        all_n.append(n)
    all_e = np.array(all_e, dtype=float)
    all_n = np.array(all_n, dtype=float)

    pad_e = max(1.0, 0.02 * max(1e-6, float(all_e.max() - all_e.min())))
    pad_n = max(1.0, 0.02 * max(1e-6, float(all_n.max() - all_n.min())))
    e_min, e_max = float(all_e.min() - pad_e), float(all_e.max() + pad_e)
    n_min, n_max = float(all_n.min() - pad_n), float(all_n.max() + pad_n)

    # Figure sizing
    span_n = (n_max - n_min)
    span_e = (e_max - e_min)
    aspect = span_e / max(1e-12, span_n)
    width = 7.0 * min(2.0, max(0.7, aspect))
    height = 7.0

    fig = plt.figure(figsize=(width, height), dpi=400)
    ax = plt.gca()

    if urchin_data and len(ur_e) > 0:
        ax.scatter(ur_e, ur_n, s=ur_sizes, facecolors="black", edgecolors="black", alpha=0.5)

    for cname, pts in class_points_m.items():
        if not pts:
            continue
        color = heatmap_colors[cname]
        ee = [p[0] for p in pts]
        nn = [p[1] for p in pts]
        ax.scatter(ee, nn, s=50, c=color, edgecolors="none", alpha=0.9, label=cname)

    fsize = 18
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(n_min, n_max)
    ax.tick_params(axis="x", labelsize=fsize - 4)
    ax.tick_params(axis="y", labelsize=fsize - 4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Distance East from SW corner (m)", fontsize=fsize - 3)
    ax.set_ylabel("Distance North from SW corner (m)", fontsize=fsize - 3)
    ax.set_title(
        "Predicted habitat classes (metric coords, SW origin)",
        fontsize=fsize + 1,
    )

    class_handles = [
        Patch(facecolor=heatmap_colors[c], edgecolor="none", label=c)
        for c in heatmap_colors.keys()
        if c in class_points_m and len(class_points_m[c]) > 0
    ]
    urchin_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        markerfacecolor="black",
        markersize=6,
        linewidth=0,
        alpha=0.6,
        label="Urchin Count (size)",
    )
    handles = class_handles + ([urchin_handle] if (urchin_data and len(ur_e) > 0) else [])
    if handles:
        ax.legend(handles=handles, loc="best", frameon=True, fontsize=fsize - 3)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)
    print(f"[Matplotlib] V3 (SW origin, meters) static map saved to {output_png}")


def get_density_heat_maps_metric_png_v4(
    combined_dict,
    output_path=".",
    png_file="heatmap_maps_v4.png",
):
    """
    V4 static plot: same as V3 (SW-corner origin, meters), but makes urchin-size encoding
    easier to interpret by:
    - using a more visible size scaling (marker area ~ sqrt(count))
    - adding an explicit size legend with example counts
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from collections import defaultdict

    output_png = os.path.join(output_path, png_file)

    # Collect all valid lat/lon points (Y=lat, X=lon)
    all_points = []
    for df in combined_dict.values():
        if "Y" in df.columns and "X" in df.columns:
            all_points.extend(df[["Y", "X"]].values.tolist())
    if len(all_points) == 0:
        raise ValueError("No points found in combined_dict (expected columns 'Y' and 'X').")

    lats = np.array([p[0] for p in all_points], dtype=float)
    lons = np.array([p[1] for p in all_points], dtype=float)
    if lats.size == 0 or lons.size == 0:
        raise ValueError("No valid lat/lon points to plot.")

    lat_min0 = float(np.min(lats))
    lon_min0 = float(np.min(lons))

    # Use a representative latitude for lon scaling (small area; stable)
    lat_ref = float(np.mean(lats))
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * float(np.cos(np.deg2rad(lat_ref)))

    def _to_m_sw(lat: float, lon: float) -> tuple[float, float]:
        north_m = (float(lat) - lat_min0) * meters_per_deg_lat
        east_m = (float(lon) - lon_min0) * meters_per_deg_lon
        return east_m, north_m  # (east_m, north_m)

    heatmap_colors = {
        "Reef-Urchin-Barren": "red",
        "Reef-Partial-Urchin-Barren": "orange",
        "Unconsolidated": "blue",
        "Reef-FnEc": "#99ff99",
        "Reef-Kelp": "green",
        "Reef-BrLfa": "purple",
        "Reef-Partial-BrLfa": "pink",
    }

    def _normalize_label(lbl: str) -> str:
        if lbl == "Reef-Urchin-Barren (Review)":
            return "Reef-Urchin-Barren"
        if lbl in ("Reef-Partial-Urchin-Barren (Review)",):
            return "Reef-Partial-Urchin-Barren"
        if lbl in ("Reef-Partial-Grazed (Review)", "Reef-Partial-Grazed"):
            return "Reef-Partial-BrLfa"
        if lbl == "Reef-Grazed":
            return "Reef-BrLfa"
        if lbl == "Reef-Vegetated":
            return "Reef-FnEc"
        return lbl

    # 1) Urchin points (size by count)
    urchin_data = []
    for df in combined_dict.values():
        if "urchin_count" in df.columns:
            for _, row in df.iterrows():
                try:
                    cnt = float(row["urchin_count"])
                except Exception:
                    continue
                if np.isfinite(cnt):
                    urchin_data.append([row["Y"], row["X"], cnt])

    ur_e, ur_n, ur_sizes, ur_counts = [], [], [], []
    i = 0
    if urchin_data:
        for y, x, c in urchin_data:
            if i % 2 == 0:  # keep original subsampling behavior
                e, n = _to_m_sw(y, x)
                ur_e.append(e)
                ur_n.append(n)
                ur_counts.append(float(c))
            i += 1

        ur_e = np.array(ur_e, dtype=float)
        ur_n = np.array(ur_n, dtype=float)
        ur_counts = np.array(ur_counts, dtype=float)

        # More visible scaling than V1/V2/V3: marker area ~ sqrt(count)
        # This keeps large counts noticeable without exploding the biggest points.
        cmin = float(np.nanmin(ur_counts))
        cmax = float(np.nanmax(ur_counts))
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= 0:
            ur_sizes = np.full_like(ur_counts, 80.0, dtype=float)
        else:
            c = np.clip(ur_counts, 0.0, None)
            s_raw = np.sqrt(c)
            s0, s1 = float(np.nanmin(s_raw)), float(np.nanmax(s_raw))
            if s1 <= s0:
                s1 = s0 + 1.0
            # Scatter 's' is area in points^2. Use a wide range so size differences are obvious.
            s_min, s_max = 60.0, 1400.0
            ur_sizes = s_min + (s_max - s_min) * (s_raw - s0) / (s1 - s0)

    # 2) Class points
    class_points_m = defaultdict(list)  # cname -> list[(e_m, n_m)]
    for _, df in combined_dict.items():
        if "predictions" not in df.columns:
            continue
        for _, row in df.iterrows():
            pred = _normalize_label(row["predictions"])
            if pred not in heatmap_colors:
                continue
            e, n = _to_m_sw(row["Y"], row["X"])
            class_points_m[pred].append((e, n))

    # Bounds in meters with small padding
    all_e, all_n = [], []
    for y, x in zip(lats, lons):
        e, n = _to_m_sw(y, x)
        all_e.append(e)
        all_n.append(n)
    all_e = np.array(all_e, dtype=float)
    all_n = np.array(all_n, dtype=float)
    pad_e = max(1.0, 0.02 * max(1e-6, float(all_e.max() - all_e.min())))
    pad_n = max(1.0, 0.02 * max(1e-6, float(all_n.max() - all_n.min())))
    e_min, e_max = float(all_e.min() - pad_e), float(all_e.max() + pad_e)
    n_min, n_max = float(all_n.min() - pad_n), float(all_n.max() + pad_n)

    # Figure sizing
    span_n = (n_max - n_min)
    span_e = (e_max - e_min)
    aspect = span_e / max(1e-12, span_n)
    width = 7.0 * min(2.0, max(0.7, aspect))
    height = 7.0

    fig = plt.figure(figsize=(width, height), dpi=400)
    ax = plt.gca()

    # Plot urchins first (semi-transparent) so class points remain visible
    if urchin_data and len(ur_e) > 0:
        ax.scatter(
            ur_e,
            ur_n,
            s=ur_sizes,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            alpha=0.75,
        )

    # Plot classes
    for cname, pts in class_points_m.items():
        if not pts:
            continue
        color = heatmap_colors[cname]
        ee = [p[0] for p in pts]
        nn = [p[1] for p in pts]
        ax.scatter(ee, nn, s=50, c=color, edgecolors="none", alpha=0.9, label=cname)

    fsize = 18
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(n_min, n_max)
    ax.tick_params(axis="x", labelsize=fsize - 4)
    ax.tick_params(axis="y", labelsize=fsize - 4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Distance East from SW corner (m)", fontsize=fsize - 3)
    ax.set_ylabel("Distance North from SW corner (m)", fontsize=fsize - 3)
    ax.set_title("Predicted habitat classes (metric coords, SW origin)", fontsize=fsize + 1)

    # Legend 1: class colors
    class_handles = [
        Patch(facecolor=heatmap_colors[c], edgecolor="none", label=c)
        for c in heatmap_colors.keys()
        if c in class_points_m and len(class_points_m[c]) > 0
    ]
    leg1 = None
    if class_handles:
        leg1 = ax.legend(handles=class_handles, loc="upper right", frameon=True, fontsize=fsize - 5, title="Class")
        leg1.get_title().set_fontsize(fsize - 5)
        ax.add_artist(leg1)

    # Legend 2: urchin size examples
    if urchin_data and len(ur_e) > 0 and len(ur_counts) > 0:
        # Pick a few representative counts
        reps = np.array(
            [
                float(np.nanpercentile(ur_counts, 10)),
                float(np.nanpercentile(ur_counts, 50)),
                float(np.nanpercentile(ur_counts, 90)),
            ],
            dtype=float,
        )
        reps = np.unique(np.round(reps).astype(int))
        reps = reps[reps > 0]
        if reps.size > 0:
            # Map representative counts to the same size scaling
            s_raw = np.sqrt(np.clip(reps.astype(float), 0.0, None))
            s0, s1 = float(np.nanmin(np.sqrt(np.clip(ur_counts, 0.0, None)))), float(np.nanmax(np.sqrt(np.clip(ur_counts, 0.0, None))))
            if s1 <= s0:
                s1 = s0 + 1.0
            s_min, s_max = 60.0, 1400.0
            rep_sizes = s_min + (s_max - s_min) * (s_raw - s0) / (s1 - s0)

            size_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="black",
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                    markersize=float(np.sqrt(s) / 2.0),  # convert area-ish to marker size for legend
                    linewidth=0,
                    label=f"{int(c)}",
                )
                for c, s in zip(reps, rep_sizes)
            ]
            leg2 = ax.legend(
                handles=size_handles,
                loc="lower right",
                frameon=True,
                fontsize=fsize - 5,
                title="Urchin count\n(marker size)",
            )
            leg2.get_title().set_fontsize(fsize - 6)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)
    print(f"[Matplotlib] V4 (interpretable urchin sizes) static map saved to {output_png}")


def get_density_heat_maps_two_panel_v5(
    combined_dict,
    output_path=".",
    png_file="heatmap_maps_v5.png",
    hex_cell_size_m: float = 5.0,
):
    """
    V5 paper-style figure: two panels with shared metric axes (meters, SW-corner origin).

    - Left: habitat class points (categorical, colored)
    - Right: urchin abundance as a hexbin heatmap (sum of counts per hex) with colorbar
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from collections import defaultdict
    from matplotlib.colors import LogNorm

    output_png = os.path.join(output_path, png_file)

    # Collect all lat/lon points for bounds
    all_points = []
    for df in combined_dict.values():
        if "Y" in df.columns and "X" in df.columns:
            all_points.extend(df[["Y", "X"]].values.tolist())
    if len(all_points) == 0:
        raise ValueError("No points found in combined_dict (expected columns 'Y' and 'X').")

    lats = np.array([p[0] for p in all_points], dtype=float)
    lons = np.array([p[1] for p in all_points], dtype=float)
    if lats.size == 0 or lons.size == 0:
        raise ValueError("No valid lat/lon points to plot.")

    # SW-corner origin
    lat_min0 = float(np.min(lats))
    lon_min0 = float(np.min(lons))

    # Local meters conversion (stable for small areas)
    lat_ref = float(np.mean(lats))
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * float(np.cos(np.deg2rad(lat_ref)))

    def _to_m_sw(lat: float, lon: float) -> tuple[float, float]:
        north_m = (float(lat) - lat_min0) * meters_per_deg_lat
        east_m = (float(lon) - lon_min0) * meters_per_deg_lon
        return east_m, north_m

    heatmap_colors = {
        "Reef-Urchin-Barren": "red",
        "Reef-Partial-Urchin-Barren": "orange",
        "Unconsolidated": "blue",
        "Reef-FnEc": "#99ff99",
        "Reef-Kelp": "green",
        "Reef-BrLfa": "purple",
        "Reef-Partial-BrLfa": "pink",
    }

    def _normalize_label(lbl: str) -> str:
        if lbl == "Reef-Urchin-Barren (Review)":
            return "Reef-Urchin-Barren"
        if lbl in ("Reef-Partial-Urchin-Barren (Review)",):
            return "Reef-Partial-Urchin-Barren"
        if lbl in ("Reef-Partial-Grazed (Review)", "Reef-Partial-Grazed"):
            return "Reef-Partial-BrLfa"
        if lbl == "Reef-Grazed":
            return "Reef-BrLfa"
        if lbl == "Reef-Vegetated":
            return "Reef-FnEc"
        return lbl

    # Build class points + urchin points in meters
    class_points_m = defaultdict(list)
    ur_e, ur_n, ur_counts = [], [], []
    i = 0
    for _, df in combined_dict.items():
        if "Y" not in df.columns or "X" not in df.columns:
            continue

        # class points
        if "predictions" in df.columns:
            for _, row in df.iterrows():
                pred = _normalize_label(row["predictions"])
                if pred in heatmap_colors:
                    e, n = _to_m_sw(row["Y"], row["X"])
                    class_points_m[pred].append((e, n))

        # urchin points (subsample to match existing behavior)
        if "urchin_count" in df.columns:
            for _, row in df.iterrows():
                try:
                    cnt = float(row["urchin_count"])
                except Exception:
                    continue
                if not np.isfinite(cnt):
                    continue
                if i % 2 == 0:
                    e, n = _to_m_sw(row["Y"], row["X"])
                    ur_e.append(e)
                    ur_n.append(n)
                    ur_counts.append(cnt)
                i += 1

    ur_e = np.array(ur_e, dtype=float)
    ur_n = np.array(ur_n, dtype=float)
    ur_counts = np.array(ur_counts, dtype=float)

    # Bounds with padding (meters)
    all_e = np.array([_to_m_sw(y, x)[0] for y, x in zip(lats, lons)], dtype=float)
    all_n = np.array([_to_m_sw(y, x)[1] for y, x in zip(lats, lons)], dtype=float)
    pad_e = max(1.0, 0.02 * max(1e-6, float(all_e.max() - all_e.min())))
    pad_n = max(1.0, 0.02 * max(1e-6, float(all_n.max() - all_n.min())))
    e_min, e_max = float(all_e.min() - pad_e), float(all_e.max() + pad_e)
    n_min, n_max = float(all_n.min() - pad_n), float(all_n.max() + pad_n)

    # Figure sizing (wide, 2 panels)
    span_n = (n_max - n_min)
    span_e = (e_max - e_min)
    aspect = span_e / max(1e-12, span_n)
    panel_w = 6.2 * min(2.0, max(0.85, aspect))
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(panel_w * 2.0, 6.5), dpi=400, sharex=True, sharey=True
    )

    # ----- Panel A: habitat classes -----
    for cname, pts in class_points_m.items():
        if not pts:
            continue
        ee = [p[0] for p in pts]
        nn = [p[1] for p in pts]
        ax0.scatter(ee, nn, s=28, c=heatmap_colors[cname], edgecolors="none", alpha=0.95, label=cname)

    ax0.set_title("A) Habitat class (rule-based)", fontsize=15)
    ax0.set_xlabel("Distance East from SW corner (m)", fontsize=13)
    ax0.set_ylabel("Distance North from SW corner (m)", fontsize=13)
    ax0.set_xlim(e_min, e_max)
    ax0.set_ylim(n_min, n_max)
    ax0.set_aspect("equal", adjustable="box")
    ax0.grid(True, alpha=0.2)

    class_handles = [
        Patch(facecolor=heatmap_colors[c], edgecolor="none", label=c)
        for c in heatmap_colors.keys()
        if c in class_points_m and len(class_points_m[c]) > 0
    ]
    if class_handles:
        ax0.legend(handles=class_handles, loc="upper left", frameon=True, fontsize=9)

    # ----- Panel B: urchins as hexbin heatmap -----
    ax1.set_title("B) Urchin abundance (hexbin)", fontsize=15)
    ax1.set_xlabel("Distance East from SW corner (m)", fontsize=13)
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.2)

    if ur_e.size == 0:
        ax1.text(
            0.5,
            0.5,
            "No urchin data",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            fontsize=13,
        )
    else:
        # Choose gridsize from desired cell size (meters). hexbin gridsize ~ number of hexes across x.
        gx = int(max(10, min(80, round((e_max - e_min) / max(1e-6, hex_cell_size_m)))))
        # Sum urchin counts per hex; use log color scaling to handle skew.
        hb = ax1.hexbin(
            ur_e,
            ur_n,
            C=ur_counts,
            reduce_C_function=np.sum,
            gridsize=gx,
            cmap="viridis",
            mincnt=1,
            norm=LogNorm(vmin=1),
        )
        cbar = fig.colorbar(hb, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label("Urchin count (sum per hex, log scale)", fontsize=12)

    ax1.set_xlim(e_min, e_max)
    ax1.set_ylim(n_min, n_max)

    fig.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)
    print(f"[Matplotlib] V5 (two-panel) figure saved to {output_png}")


def get_density_heat_maps_metric_png_v6(
    combined_dict,
    output_path=".",
    png_file="heatmap_maps_v6.png",
):
    """
    V6: refinement of V4 for paper readability.

    Changes vs V4:
    - Habitat-class legend is moved outside the axes (no overlap with transect points).
    - Urchin legend uses exactly three references: min / mean / max counts, with marker sizes
      that match the plotted scatter marker areas.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from collections import defaultdict

    output_png = os.path.join(output_path, png_file)

    # Collect all valid lat/lon points (Y=lat, X=lon)
    all_points = []
    for df in combined_dict.values():
        if "Y" in df.columns and "X" in df.columns:
            all_points.extend(df[["Y", "X"]].values.tolist())
    if len(all_points) == 0:
        raise ValueError("No points found in combined_dict (expected columns 'Y' and 'X').")

    lats = np.array([p[0] for p in all_points], dtype=float)
    lons = np.array([p[1] for p in all_points], dtype=float)
    if lats.size == 0 or lons.size == 0:
        raise ValueError("No valid lat/lon points to plot.")

    lat_min0 = float(np.min(lats))
    lon_min0 = float(np.min(lons))

    # Use a representative latitude for lon scaling (small area; stable)
    lat_ref = float(np.mean(lats))
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * float(np.cos(np.deg2rad(lat_ref)))

    def _to_m_sw(lat: float, lon: float) -> tuple[float, float]:
        north_m = (float(lat) - lat_min0) * meters_per_deg_lat
        east_m = (float(lon) - lon_min0) * meters_per_deg_lon
        return east_m, north_m  # (east_m, north_m)

    heatmap_colors = {
        "Reef-Urchin-Barren": "red",
        "Reef-Partial-Urchin-Barren": "orange",
        "Unconsolidated": "blue",
        "Reef-FnEc": "#99ff99",
        "Reef-Kelp": "green",
        "Reef-BrLfa": "purple",
        "Reef-Partial-BrLfa": "pink",
    }

    def _normalize_label(lbl: str) -> str:
        if lbl == "Reef-Urchin-Barren (Review)":
            return "Reef-Urchin-Barren"
        if lbl in ("Reef-Partial-Urchin-Barren (Review)",):
            return "Reef-Partial-Urchin-Barren"
        if lbl in ("Reef-Partial-Grazed (Review)", "Reef-Partial-Grazed"):
            return "Reef-Partial-BrLfa"
        if lbl == "Reef-Grazed":
            return "Reef-BrLfa"
        if lbl == "Reef-Vegetated":
            return "Reef-FnEc"
        return lbl

    # 1) Urchin points (size by count)
    urchin_data = []
    for df in combined_dict.values():
        if "urchin_count" in df.columns:
            for _, row in df.iterrows():
                try:
                    cnt = float(row["urchin_count"])
                except Exception:
                    continue
                if np.isfinite(cnt):
                    urchin_data.append([row["Y"], row["X"], cnt])

    ur_e, ur_n, ur_counts = [], [], []
    i = 0
    if urchin_data:
        for y, x, c in urchin_data:
            if i % 2 == 0:  # keep original subsampling behavior
                e, n = _to_m_sw(y, x)
                ur_e.append(e)
                ur_n.append(n)
                ur_counts.append(float(c))
            i += 1

    ur_e = np.array(ur_e, dtype=float)
    ur_n = np.array(ur_n, dtype=float)
    ur_counts = np.array(ur_counts, dtype=float)

    # Marker size scaling (same style as V4, but kept here explicitly)
    # Scatter 's' is marker area in points^2.
    s_min, s_max = 60.0, 1400.0
    if ur_counts.size > 0 and np.any(np.isfinite(ur_counts)):
        c = np.clip(ur_counts, 0.0, None)
        s_raw = np.sqrt(c)
        s0, s1 = float(np.nanmin(s_raw)), float(np.nanmax(s_raw))
        if s1 <= s0:
            s1 = s0 + 1.0
        ur_sizes = s_min + (s_max - s_min) * (s_raw - s0) / (s1 - s0)
    else:
        ur_sizes = np.array([], dtype=float)

    # 2) Class points
    class_points_m = defaultdict(list)  # cname -> list[(e_m, n_m)]
    for _, df in combined_dict.items():
        if "predictions" not in df.columns:
            continue
        for _, row in df.iterrows():
            pred = _normalize_label(row["predictions"])
            if pred not in heatmap_colors:
                continue
            e, n = _to_m_sw(row["Y"], row["X"])
            class_points_m[pred].append((e, n))

    # Bounds in meters with small padding
    all_e, all_n = [], []
    for y, x in zip(lats, lons):
        e, n = _to_m_sw(y, x)
        all_e.append(e)
        all_n.append(n)
    all_e = np.array(all_e, dtype=float)
    all_n = np.array(all_n, dtype=float)
    pad_e = max(1.0, 0.02 * max(1e-6, float(all_e.max() - all_e.min())))
    pad_n = max(1.0, 0.02 * max(1e-6, float(all_n.max() - all_n.min())))
    e_min, e_max = float(all_e.min() - pad_e), float(all_e.max() + pad_e)
    n_min, n_max = float(all_n.min() - pad_n), float(all_n.max() + pad_n)

    # Figure sizing (+ extra width for outside legends)
    span_n = (n_max - n_min)
    span_e = (e_max - e_min)
    aspect = span_e / max(1e-12, span_n)
    width = 7.0 * min(2.0, max(0.7, aspect))
    height = 7.0
    fig = plt.figure(figsize=(width * 1.35, height), dpi=400)
    ax = plt.gca()

    # Plot urchins first (outline circles) so class points remain visible
    if ur_e.size > 0:
        ax.scatter(
            ur_e,
            ur_n,
            s=ur_sizes,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            alpha=0.8,
            zorder=1,
        )

    # Plot classes
    for cname, pts in class_points_m.items():
        if not pts:
            continue
        color = heatmap_colors[cname]
        ee = [p[0] for p in pts]
        nn = [p[1] for p in pts]
        ax.scatter(ee, nn, s=50, c=color, edgecolors="none", alpha=0.9, zorder=2)

    fsize = 18
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(n_min, n_max)
    ax.tick_params(axis="x", labelsize=fsize - 4)
    ax.tick_params(axis="y", labelsize=fsize - 4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Distance East from SW corner (m)", fontsize=fsize - 3)
    ax.set_ylabel("Distance North from SW corner (m)", fontsize=fsize - 3)
    ax.set_title("Predicted habitat classes (metric coords, SW origin)", fontsize=fsize + 1)

    # Legend: habitat classes (outside axes so it doesn't cover points)
    class_handles = [
        Patch(facecolor=heatmap_colors[c], edgecolor="none", label=c)
        for c in heatmap_colors.keys()
        if c in class_points_m and len(class_points_m[c]) > 0
    ]
    if class_handles:
        leg_classes = ax.legend(
            handles=class_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=fsize - 7,
            title="Habitat class",
        )
        leg_classes.get_title().set_fontsize(fsize - 7)
        ax.add_artist(leg_classes)

    # Legend: urchin sizes (min/mean/max) with marker sizes matching scatter areas
    if ur_counts.size > 0 and ur_sizes.size > 0:
        cmin = float(np.nanmin(ur_counts))
        cmean = float(np.nanmean(ur_counts))
        cmax = float(np.nanmax(ur_counts))
        # Round display numbers for readability
        reps = [("min", int(round(cmin))), ("mean", int(round(cmean))), ("max", int(round(cmax)))]

        def _size_for_count(count: float) -> float:
            # apply the same scaling used for ur_sizes
            s_raw = math.sqrt(max(0.0, float(count)))
            return s_min + (s_max - s_min) * (s_raw - s0) / (s1 - s0)

        import math

        rep_handles = []
        for name, cval in reps:
            s_area = _size_for_count(cval)
            # Convert scatter area (points^2) to Line2D marker size (points, ~diameter)
            ms = 2.0 * math.sqrt(max(1e-12, s_area) / math.pi)
            rep_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="black",
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                    markersize=ms,
                    linewidth=0,
                    label=f"{name}: {cval}",
                )
            )

        leg_urchin = ax.legend(
            handles=rep_handles,
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=fsize - 7,
            title="Urchin count (marker size)",
        )
        leg_urchin.get_title().set_fontsize(fsize - 8)

    ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)
    print(f"[Matplotlib] V6 (clean legends, min/mean/max urchins) saved to {output_png}")


def get_density_heat_maps_metric_png_v7(
    combined_dict,
    output_path=".",
    png_file="heatmap_maps_v7.png",
):
    """
    V7: V6 but fixes urchin legend readability.

    - Keeps legend marker sizes EXACTLY matched to plotted scatter areas.
    - Increases legend vertical spacing based on the largest marker diameter (so circles don't overlap).
    """
    import os
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from collections import defaultdict

    output_png = os.path.join(output_path, png_file)

    # Collect all valid lat/lon points (Y=lat, X=lon)
    all_points = []
    for df in combined_dict.values():
        if "Y" in df.columns and "X" in df.columns:
            all_points.extend(df[["Y", "X"]].values.tolist())
    if len(all_points) == 0:
        raise ValueError("No points found in combined_dict (expected columns 'Y' and 'X').")

    lats = np.array([p[0] for p in all_points], dtype=float)
    lons = np.array([p[1] for p in all_points], dtype=float)
    if lats.size == 0 or lons.size == 0:
        raise ValueError("No valid lat/lon points to plot.")

    lat_min0 = float(np.min(lats))
    lon_min0 = float(np.min(lons))

    # Use a representative latitude for lon scaling (small area; stable)
    lat_ref = float(np.mean(lats))
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * float(np.cos(np.deg2rad(lat_ref)))

    def _to_m_sw(lat: float, lon: float) -> tuple[float, float]:
        north_m = (float(lat) - lat_min0) * meters_per_deg_lat
        east_m = (float(lon) - lon_min0) * meters_per_deg_lon
        return east_m, north_m  # (east_m, north_m)

    heatmap_colors = {
        "Reef-Urchin-Barren": "red",
        "Reef-Partial-Urchin-Barren": "orange",
        "Unconsolidated": "blue",
        "Reef-FnEc": "#99ff99",
        "Reef-Kelp": "green",
        "Reef-BrLfa": "purple",
        "Reef-Partial-BrLfa": "pink",
    }

    def _normalize_label(lbl: str) -> str:
        if lbl == "Reef-Urchin-Barren (Review)":
            return "Reef-Urchin-Barren"
        if lbl in ("Reef-Partial-Urchin-Barren (Review)",):
            return "Reef-Partial-Urchin-Barren"
        if lbl in ("Reef-Partial-Grazed (Review)", "Reef-Partial-Grazed"):
            return "Reef-Partial-BrLfa"
        if lbl == "Reef-Grazed":
            return "Reef-BrLfa"
        if lbl == "Reef-Vegetated":
            return "Reef-FnEc"
        return lbl

    # Urchin points (subsample to match original behavior)
    urchin_data = []
    for df in combined_dict.values():
        if "urchin_count" in df.columns:
            for _, row in df.iterrows():
                try:
                    cnt = float(row["urchin_count"])
                except Exception:
                    continue
                if np.isfinite(cnt):
                    urchin_data.append([row["Y"], row["X"], cnt])

    ur_e, ur_n, ur_counts = [], [], []
    i = 0
    if urchin_data:
        for y, x, c in urchin_data:
            if i % 2 == 0:
                e, n = _to_m_sw(y, x)
                ur_e.append(e)
                ur_n.append(n)
                ur_counts.append(float(c))
            i += 1
    ur_e = np.array(ur_e, dtype=float)
    ur_n = np.array(ur_n, dtype=float)
    ur_counts = np.array(ur_counts, dtype=float)

    # Size scaling (scatter 's' is marker AREA in points^2)
    s_min, s_max = 60.0, 1400.0
    if ur_counts.size > 0 and np.any(np.isfinite(ur_counts)):
        c = np.clip(ur_counts, 0.0, None)
        s_raw = np.sqrt(c)
        s0, s1 = float(np.nanmin(s_raw)), float(np.nanmax(s_raw))
        if s1 <= s0:
            s1 = s0 + 1.0
        ur_sizes = s_min + (s_max - s_min) * (s_raw - s0) / (s1 - s0)
    else:
        # still define for legend helper
        s0, s1 = 0.0, 1.0
        ur_sizes = np.array([], dtype=float)

    def _area_for_count(count: float) -> float:
        # EXACT same mapping used for ur_sizes above
        s_raw = math.sqrt(max(0.0, float(count)))
        return s_min + (s_max - s_min) * (s_raw - s0) / (s1 - s0)

    def _ms_from_area(s_area: float) -> float:
        # Convert scatter area (points^2) to marker diameter (points) for Line2D legend.
        # For circles: area = pi*(d/2)^2  => d = 2*sqrt(area/pi)
        return 2.0 * math.sqrt(max(1e-12, float(s_area)) / math.pi)

    # Class points
    class_points_m = defaultdict(list)
    for _, df in combined_dict.items():
        if "predictions" not in df.columns:
            continue
        for _, row in df.iterrows():
            pred = _normalize_label(row["predictions"])
            if pred not in heatmap_colors:
                continue
            e, n = _to_m_sw(row["Y"], row["X"])
            class_points_m[pred].append((e, n))

    # Bounds
    all_e = np.array([_to_m_sw(y, x)[0] for y, x in zip(lats, lons)], dtype=float)
    all_n = np.array([_to_m_sw(y, x)[1] for y, x in zip(lats, lons)], dtype=float)
    pad_e = max(1.0, 0.02 * max(1e-6, float(all_e.max() - all_e.min())))
    pad_n = max(1.0, 0.02 * max(1e-6, float(all_n.max() - all_n.min())))
    e_min, e_max = float(all_e.min() - pad_e), float(all_e.max() + pad_e)
    n_min, n_max = float(all_n.min() - pad_n), float(all_n.max() + pad_n)

    # Figure sizing (+ extra width for outside legends)
    span_n = (n_max - n_min)
    span_e = (e_max - e_min)
    aspect = span_e / max(1e-12, span_n)
    width = 7.0 * min(2.0, max(0.7, aspect))
    height = 7.0
    fig = plt.figure(figsize=(width * 1.4, height), dpi=400)
    ax = plt.gca()

    # Plot urchins (outline circles)
    if ur_e.size > 0:
        ax.scatter(
            ur_e,
            ur_n,
            s=ur_sizes,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            alpha=0.8,
            zorder=1,
        )

    # Plot classes
    for cname, pts in class_points_m.items():
        if not pts:
            continue
        color = heatmap_colors[cname]
        ee = [p[0] for p in pts]
        nn = [p[1] for p in pts]
        ax.scatter(ee, nn, s=50, c=color, edgecolors="none", alpha=0.9, zorder=2)

    fsize = 24
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(n_min, n_max)
    ax.tick_params(axis="x", labelsize=fsize - 4)
    ax.tick_params(axis="y", labelsize=fsize - 4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Distance East from SW corner (m)", fontsize=fsize - 3)
    ax.set_ylabel("Distance North from SW corner (m)", fontsize=fsize - 3)
    ax.set_title("Predicted habitat classes (metric coords, SW origin)", fontsize=fsize + 1)

    # Habitat legend INSIDE (choose a corner with minimal point overlap)
    class_handles = [
        Patch(facecolor=heatmap_colors[c], edgecolor="none", label=c)
        for c in heatmap_colors.keys()
        if c in class_points_m and len(class_points_m[c]) > 0
    ]
    # Urchin legend INSIDE, horizontal at bottom-right.
    # IMPORTANT: use actual scatter handles so legend circle sizes match the plot EXACTLY.
    if ur_counts.size > 0 and np.any(np.isfinite(ur_counts)):
        cmin = float(np.nanmin(ur_counts))
        cmean = float(np.nanmean(ur_counts))
        cmax = float(np.nanmax(ur_counts))
        reps = [("min", int(round(cmin))), ("mean", int(round(cmean))), ("max", int(round(cmax)))]

        rep_diam_pts = []
        rep_handles = []
        rep_labels = []
        for name, cval in reps:
            s_area = _area_for_count(cval)
            rep_diam_pts.append(_ms_from_area(s_area))
            rep_handles.append(
                ax.scatter(
                    [],
                    [],
                    s=s_area,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.0,
                )
            )
            rep_labels.append(f"{name}: {cval}")

        legend_fontsize = fsize - 4
        max_d = float(max(rep_diam_pts)) if rep_diam_pts else 12.0
        # `labelspacing` also affects the title-to-entries gap; keep it moderate.
        labelspacing = 0.5
        # borderpad is in units of fontsize; ensure the biggest marker isn't clipped by the box
        borderpad = max(0.35, ((max_d / 2.0) / max(1.0, legend_fontsize)) * 1.05)

        leg_urchin = ax.legend(
            rep_handles,
            rep_labels,
            loc="lower left",
            # Wider horizontal legend inside bottom-right area (x0, y0, w, h in axes fraction)
            bbox_to_anchor=(0.42, 0.01, 0.56, 0.14),
            borderaxespad=0.0,
            frameon=True,
            fontsize=legend_fontsize,
            title="Urchin count (marker size)",
            markerscale=1.0,
            labelspacing=labelspacing,
            borderpad=borderpad,
            handletextpad=0.60,
            handlelength=1.0,
            columnspacing=1.25,
            scatterpoints=1,
            ncol=3,
        )
        leg_urchin.get_title().set_fontsize(fsize - 3)
        ax.add_artist(leg_urchin)

    # Now place the habitat legend, inside, avoiding points (and the urchin legend if present).
    if class_handles:
        # Score candidate placements by how many points they cover.
        # Use ALL points (not just class points) to avoid covering any plotted data.
        pts_disp = ax.transData.transform(np.column_stack([all_e, all_n]))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        urchin_bbox = None
        if "leg_urchin" in locals():
            try:
                urchin_bbox = leg_urchin.get_window_extent(renderer=renderer)
            except Exception:
                urchin_bbox = None

        candidates = [
            ("upper left", (0.02, 0.98)),
            ("upper right", (0.98, 0.98)),
            ("center left", (0.02, 0.65)),
            ("center right", (0.98, 0.65)),
        ]

        best = None
        best_score = None
        for loc, anchor in candidates:
            tmp_leg = ax.legend(
                handles=class_handles,
                loc=loc,
                bbox_to_anchor=anchor,
                borderaxespad=0.0,
                frameon=True,
                fontsize=fsize - 4,
                title="Habitat class",
            )
            tmp_leg.get_title().set_fontsize(fsize - 3)
            fig.canvas.draw()
            bbox = tmp_leg.get_window_extent(renderer=renderer)

            x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
            x = pts_disp[:, 0]
            y = pts_disp[:, 1]
            covered = int(np.sum((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)))

            overlap_urchin = 0
            if urchin_bbox is not None:
                ix0 = max(x0, urchin_bbox.x0)
                iy0 = max(y0, urchin_bbox.y0)
                ix1 = min(x1, urchin_bbox.x1)
                iy1 = min(y1, urchin_bbox.y1)
                if ix1 > ix0 and iy1 > iy0:
                    overlap_urchin = 1

            score = (covered, overlap_urchin)
            tmp_leg.remove()

            if best_score is None or score < best_score:
                best_score = score
                best = (loc, anchor)

        loc, anchor = best if best is not None else ("upper left", (0.02, 0.98))
        leg_classes = ax.legend(
            handles=class_handles,
            loc=loc,
            bbox_to_anchor=anchor,
            borderaxespad=0.0,
            frameon=True,
            fontsize=fsize - 4,
            title="Habitat class",
        )
        leg_classes.get_title().set_fontsize(fsize - 3)
        ax.add_artist(leg_classes)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[Matplotlib] V7 (fixed urchin legend spacing) saved to {output_png}")


def get_density_heat_maps(combined_dict, output_path=".", output_file="heatmap_maps.html", png_file="heatmap_maps.png"):
    import os
    import numpy as np
    import folium
    from folium import FeatureGroup
    from branca.element import MacroElement, Template
    from collections import defaultdict

    # -------------------------
    # Validate / collect points
    # -------------------------
    output_html = os.path.join(output_path, output_file)
    output_png  = os.path.join(output_path, png_file)

    all_points = []
    for df in combined_dict.values():
        if 'Y' in df.columns and 'X' in df.columns:
            all_points.extend(df[['Y', 'X']].values.tolist())

    if len(all_points) == 0:
        raise ValueError("No points found in combined_dict (expected columns 'Y' and 'X').")

    # Center for Folium
    map_center = [np.mean([p[0] for p in all_points]), np.mean([p[1] for p in all_points])]

    # -------------------------
    # Fixed color mapping per class
    # -------------------------
    heatmap_colors = {
        "Reef-Urchin-Barren": "red",
        "Reef-Partial-Urchin-Barren": "orange",
        "Unconsolidated": "blue",
        "Reef-FnEc": "#99ff99",
        "Reef-Kelp": "green",
        "Reef-BrLfa": "purple",
        "Reef-Partial-BrLfa": "pink",
    }

    # -------------------------
    # Build Folium map (HTML)
    # -------------------------
    m = folium.Map(location=map_center, zoom_start=7)

    # 1) Urchin count as POINTS
    urchin_data = []
    for df in combined_dict.values():
        if 'urchin_count' in df.columns:
            for _, row in df.iterrows():
                cnt = row['urchin_count']
                # robust float conversion, skip NaNs
                try:
                    cnt = float(cnt)
                except Exception:
                    continue
                if np.isfinite(cnt):
                    urchin_data.append([row['Y'], row['X'], cnt])

    if urchin_data:
        max_urchin = max(x[2] for x in urchin_data) or 1.0
        fg_urchin = FeatureGroup(name="Urchin Count (points)", show=True)
        for y, x, count in urchin_data:
            val = count / max_urchin
            radius = 2.0 + 8.0 * val  # 2..10 px
            folium.CircleMarker(
                location=[y, x],
                radius=radius,
                color="black",
                weight=1,
                fill=True,
                fill_opacity=0.6,
                fill_color="black",
                tooltip=f"Urchin count: {count:.0f}",
            ).add_to(fg_urchin)
        fg_urchin.add_to(m)

    # 2) Multiclass predictions as POINTS
    def _normalize_label(lbl: str) -> str:
        if lbl == "Reef-Urchin-Barren (Review)":
            return "Reef-Urchin-Barren"
        if lbl in ("Reef-Partial-Urchin-Barren (Review)",):
            return "Reef-Partial-Urchin-Barren"
        if lbl in ("Reef-Partial-Grazed (Review)", "Reef-Partial-Grazed"):
            return "Reef-Partial-BrLfa"
        if lbl == "Reef-Grazed":
            return "Reef-BrLfa"
        if lbl == "Reef-Vegetated":
            return "Reef-FnEc"
        return lbl

    class_points = defaultdict(list)
    for key, df in combined_dict.items():
        if 'predictions' not in df.columns:
            continue
        for _, row in df.iterrows():
            pred = _normalize_label(row['predictions'])
            if pred in heatmap_colors:
                class_points[pred].append((row['Y'], row['X']))

    # Add one point layer per class
    for class_name, pts in class_points.items():
        color = heatmap_colors[class_name]
        fg = FeatureGroup(name=class_name, show=True)
        for (y, x) in pts:
            folium.CircleMarker(
                location=[y, x],
                radius=3.5,
                color=color,
                weight=1,
                fill=True,
                fill_opacity=0.8,
                fill_color=color,
                tooltip=f"{class_name}",
            ).add_to(fg)
        fg.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Legend (HTML)
    legend_entries = list(heatmap_colors.items())
    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 280px; height: auto; 
                background-color: white; z-index:9999; padding: 10px;
                font-size:14px; border-radius:5px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
        <b>Legend</b><br>
        <div><div style="width: 12px; height: 12px; background: black; display: inline-block; margin-right:6px;"></div> Urchin Count (point size)</div>
    """
    for class_name, color in legend_entries:
        legend_html += f"""
        <div><div style="width: 12px; height: 12px; background: {color}; display: inline-block; margin-right:6px;"></div> {class_name}</div>
        """
    legend_html += "</div>{% endmacro %}"
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    # Save HTML
    os.makedirs(output_path, exist_ok=True)
    m.save(output_html)
    print(f"[Folium] Map saved to {output_html}")

    # -------------------------
    # Static PNG with Matplotlib
    # -------------------------
    # Collect arrays for plotting
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # tight bounds with small padding
    lats = np.array([p[0] for p in all_points], dtype=float)
    lons = np.array([p[1] for p in all_points], dtype=float)
    if lats.size == 0 or lons.size == 0:
        raise ValueError("No valid lat/lon points to plot.")

    # pad roughly to zoom in; adjust as needed
    pad_lat = max(1e-4, 0.02 * max(1e-6, lats.max() - lats.min()))
    pad_lon = max(1e-4, 0.02 * max(1e-6, lons.max() - lons.min()))
    lat_min, lat_max = lats.min() - pad_lat, lats.max() + pad_lat
    lon_min, lon_max = lons.min() - pad_lon, lons.max() + pad_lon

    # Prepare per-class arrays
    class_arrays = {}
    for cname, pts in class_points.items():
        if len(pts) == 0:
            continue
        arr = np.array(pts, dtype=float)  # (N, 2) -> [lat, lon]
        class_arrays[cname] = (arr[:, 0], arr[:, 1])  # (lats, lons)

    # Prepare urchin arrays and scale
    ur_lats, ur_lons, ur_sizes = [], [], []
    urchin_count = len(urchin_data)
    # urchin_data = []
    i =0 
    if urchin_data:
        for y, x, c in urchin_data:
            if i%2==0:
                ur_lats.append(float(y))
                ur_lons.append(float(x))
                ur_sizes.append(float(c))
            i+=1
        ur_lats = np.array(ur_lats, dtype=float)
        ur_lons = np.array(ur_lons, dtype=float)
        ur_sizes = np.array(ur_sizes, dtype=float)
        # Normalize sizes to a reasonable pixel range (20..120 pts^2 for scatter)
        if np.any(np.isfinite(ur_sizes)):
            smin, smax = np.nanmin(ur_sizes), np.nanmax(ur_sizes)
            if smax <= smin:
                smax = smin + 1.0
            ur_sizes = 300+ 100.0 * (ur_sizes - smin) / (smax - smin)
        else:
            ur_sizes = np.full_like(ur_lats, 40.0)

    # Aspect ratio handling for figure dims (wider if longitude span is larger)
    span_lat = (lat_max - lat_min)
    span_lon = (lon_max - lon_min)
    aspect = span_lon / max(1e-12, span_lat)
    width = 7.0 * min(2.0, max(0.7, aspect))  # clamp to avoid extreme shapes
    height = 7.0

    fig = plt.figure(figsize=(width, height), dpi=400)
    ax = plt.gca()

    # Plot urchins on top (black, size by count)
    if urchin_data:
        ax.scatter(ur_lons, ur_lats, s=ur_sizes, facecolors='black', edgecolors='black', alpha=0.5)

    # Plot classes
    for cname, (clats, clons) in class_arrays.items():
        color = heatmap_colors[cname]
        ax.scatter(clons, clats, s=50, c=color, edgecolors='none', alpha=0.9, label=cname)

    fsize = 18

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.tick_params(axis='x', labelsize=fsize-4)       # for x tick labels
    ax.tick_params(axis='y', labelsize=fsize-4)       # for y tick labels
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Longitude",fontsize=fsize-3)
    ax.set_ylabel("Latitude",fontsize=fsize-3)
    ax.set_title(f"Predicted habitat classes by the rule-based classifier",fontsize=fsize+1)
    # Build legend: class color boxes + urchin marker
    class_handles = [Patch(facecolor=heatmap_colors[c], edgecolor='none', label=c) for c in heatmap_colors.keys() if c in class_arrays]
    urchin_handle = Line2D([0], [0], marker='o', color='black', markerfacecolor='black',
                           markersize=6, linewidth=0, alpha=0.6, label='Urchin Count (size)')
    handles = class_handles + ([urchin_handle] if urchin_data else [])

    if handles:
        ax.legend(handles=handles, loc='best', frameon=True, fontsize=fsize-3)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    print(f"[Matplotlib] Static map saved to {output_png}")
  

def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Reproduce density heatmap outputs from saved density-samples JSONs."
    )
    parser.add_argument(
        "--location",
        default="Black_Rocks",
        help="Location name to reproduce (expects <LOCATION>.csv and <LOCATION>.json).",
    )
    parser.add_argument(
        "--inputs-dir",
        default=str(repo_root / "dataset" / "density-samples" / "inputs"),
        help="Folder containing <LOCATION>.csv files (and location folders).",
    )
    parser.add_argument(
        "--saved-outputs-dir",
        default=str(repo_root / "dataset" / "density-samples" / "outputs"),
        help="Folder containing saved <LOCATION>.json files produced earlier.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(repo_root / "density_paper" / "output"),
        help="Folder to write reproduced outputs into.",
    )
    parser.add_argument(
        "--html",
        default="heatmap_maps.html",
        help="Output HTML filename.",
    )
    parser.add_argument(
        "--png",
        default="heatmap_maps.png",
        help="Output PNG filename.",
    )
    parser.add_argument(
        "--png-v2",
        default="heatmap_maps_v2.png",
        help="Output PNG filename for V2 metric plot (meters).",
    )
    parser.add_argument(
        "--png-v3",
        default="heatmap_maps_v3.png",
        help="Output PNG filename for V3 metric plot (meters, SW-corner origin; all positive).",
    )
    parser.add_argument(
        "--png-v4",
        default="heatmap_maps_v4.png",
        help="Output PNG filename for V4 metric plot (meters, clearer urchin size legend).",
    )
    parser.add_argument(
        "--png-v5",
        default="heatmap_maps_v5.png",
        help="Output PNG filename for V5 two-panel paper-style figure.",
    )
    parser.add_argument(
        "--png-v6",
        default="heatmap_maps_v6.png",
        help="Output PNG filename for V6 (v4 refined: outside class legend + min/mean/max urchin legend).",
    )
    parser.add_argument(
        "--png-v7",
        default="heatmap_maps_v7.png",
        help="Output PNG filename for V7 (v6 refined: non-overlapping urchin legend, exact sizes).",
    )
    args = parser.parse_args()

    data_dict = load_saved_density_data_dict(
        args.inputs_dir,
        args.saved_outputs_dir,
        locations=[args.location],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    get_density_heat_maps(data_dict, output_path=str(out_dir), output_file=args.html, png_file=args.png)
    get_density_heat_maps_metric_png_v2(data_dict, output_path=str(out_dir), png_file=args.png_v2)
    get_density_heat_maps_metric_png_v3(data_dict, output_path=str(out_dir), png_file=args.png_v3)
    get_density_heat_maps_metric_png_v4(data_dict, output_path=str(out_dir), png_file=args.png_v4)
    get_density_heat_maps_two_panel_v5(data_dict, output_path=str(out_dir), png_file=args.png_v5)
    get_density_heat_maps_metric_png_v6(data_dict, output_path=str(out_dir), png_file=args.png_v6)
    get_density_heat_maps_metric_png_v7(data_dict, output_path=str(out_dir), png_file=args.png_v7)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

