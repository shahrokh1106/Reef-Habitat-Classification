# Dataset preparation

This document describes how to download annotations, build train/validation/test splits, and prepare image datasets for Habibot.

Run all commands from the **repository root** unless noted otherwise.

---

## Patch dataset

The patch classifier is trained on **square crops** cut around annotated point labels on full frames. Scripts live in `dataset/python/`.

### Overview pipeline

```text
Annotation CSVs  →  split CSVs  →  download frames  →  crop patches  →  dataset/patches/
     (step 1)         (step 2)        (step 3)           (step 4)
```

Expected final layout for training and evaluation:

```text
dataset/
  dataset_csv_files/
    training_datasets/
      train_df.csv
      valid_df.csv
      test_df.csv
      test_ex_df.csv
  frames/
    {media_id}_{filename}.JPG
  patches/
    train/
      Bare rock/
      Turf/
      Ecklonia radiata/
      ...
    valid/
      ...
    test/
      ...
    test_extra/          # optional held-out campaigns
      ...
```

`python/trainer_supervised_patch_classifier.py` reads patches from `dataset/patches/` with subfolders `train`, `valid`, and `test`.

### Build the patch dataset

From the repo root:

```bash
# 1. Create train / valid / test / test_extra splits
python dataset/python/split_dataset_csv.py

# 2. Download full frames
python dataset/python/download_frames.py

# 3. Crop patches around annotated points
python dataset/python/get_patches.py
```

---

### How patches are used at training time

Patch **images on disk** keep the original annotation class folder names (e.g. `Bare rock`, `Turf`, `Ecklonia radiata`, `Urchin`, …).

When training (`python/trainer_supervised_patch_classifier.py` → `Multiclassifier`), classes are filtered and merged **in code** (not by renaming folders beforehand):


| Setting                                   | Classes                                                               |
| ----------------------------------------- | --------------------------------------------------------------------- |
| **Removed**                               | `Mobile invertebrate`, `Unscorable`, `Sessile invertebrate community` |
| **Merged into** `grazed rock` **(BrLfa)** | `Bare rock`, `Turf`, `Encrusting algae`, `Filamentous algae`          |


After merging, the patch classifier uses **7 training classes**

### Verify patch counts

Please check that patches were created correctly:

```bash
python dataset/python/plot_label_frequencies_patch_dataset.py
```
---

## Frame dataset

The frame classifier is trained on **whole-image** habitat labels. Scripts live in `dataset/python/`.

### Overview pipeline

```text
Whole-frame CSVs + frame_annotation.json  →  download frames  →  group by class  →  train/val/test folders  →  frame7_dataset_cleaned/
```

Expected final layout for training and evaluation:

```text
dataset/frame7_dataset_cleaned/
  train/
    reef_barren/
    reef_grazed/
    reef_kelp/
    reef_vegetated/
    unconsolidated/
    reef_partial_barren/
    reef_partial_grazed/
  val/
    ...
  test/
    ...
```

`python/trainer_supervised_frame_classifier.py` reads frames from `dataset/frame7_dataset_cleaned/` with subfolders `train`, `val`, and `test`.

### Build the frame dataset

Place whole-frame annotation CSVs and `frame_annotation.json` in `dataset/dataset_csv_files/csvs_whole_frame_annotations/`. Then from the repo root:

```bash
# 1. Download whole-frame images
python dataset/python/download_frames_with_whole_annotations.py

# 2. Group images into class folders
python dataset/python/split_frames_with_whole_annotations.py
```

This writes images to `dataset/frames_with_whole_annotations/` and copies them into `dataset/grouped_frames_with_whole_annotations/{class}/`.

Organize those class folders into `dataset/frame7_dataset_cleaned/` with `train/`, `val/`, and `test/` splits. **No script in this repo creates `frame7_dataset_cleaned/`** — arrange the split locally (or use a prepared copy of the dataset).

---

### How frames are used at training time

At training, images in `reef_partial_barren/` and `reef_partial_grazed/` are **skipped**. The frame classifier is trained on **5 classes**:

| Folder name | Training label |
|-------------|----------------|
| `reef_barren` | Reef-Barren |
| `reef_grazed` | Reef-Grazed |
| `reef_kelp` | Reef-Kelp |
| `reef_vegetated` | Reef-Vegetated |
| `unconsolidated` | Unconsolidated |

### Verify frame counts

Please check that frames were organized correctly:

```bash
python dataset/python/plot_label_frequencies_frame_dataset.py
```

Saves label frequency plots to `dataset/figs_dataset_analysis/`.