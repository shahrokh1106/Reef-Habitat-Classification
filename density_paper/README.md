## density_paper

Reproduce the density map outputs (HTML + PNG) **from the saved density results** in `dataset/density-samples/outputs/*.json`.

### What this uses

- **Inputs**: `dataset/density-samples/inputs/*.csv` (lat/lon + `#Label`)
- **Saved results**: `dataset/density-samples/outputs/<LOCATION>.json` (prediction + urchin count per image)
- **Outputs (reproduced)**: `density_paper/output/heatmap_maps.html` and `density_paper/output/heatmap_maps.png`

### Run

From the repo root:

```bash
python density_paper/run_density_heatmaps.py
```

Optional overrides:

```bash
python density_paper/run_density_heatmaps.py --location Black_Rocks --out-dir density_paper/output --html heatmap_maps.html --png heatmap_maps.png
```

