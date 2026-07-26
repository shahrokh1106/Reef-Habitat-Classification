import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def normalize_habibot_labels(labels):
    out = []
    for item in labels:
        item = "Reef-BrLfa" if item == "Reef-Grazed" else item
        item = "Reef-Partial-BrLfa" if item == "Reef-Partial-Grazed" else item
        item = "Reef-FnEc" if item == "Reef-Vegetated" else item
        out.append(item)
    return out


def is_no_agreement(value):
    if isinstance(value, str):
        return value.strip().lower() == "no agreement"
    return False


def is_true(value):
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def get_classification_report_plot(label_pred_list, label_truth_list, path, model_name="_"):
    y_preds = np.asarray(list(label_pred_list))
    y_truths = np.asarray(list(label_truth_list))
    report = classification_report(y_truths, y_preds, zero_division=0, output_dict=True)
    report_data = {
        cls: {
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1-score": report[cls]["f1-score"],
        }
        for cls in list(np.unique(y_truths))
    }
    report_df = pd.DataFrame(report_data).transpose()
    plt.figure(figsize=(8, 5))
    sns.heatmap(report_df, annot=True, cmap="Blues", fmt=".2f", cbar=False)
    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    plt.grid("off")
    plt.tight_layout()
    plt.savefig(os.path.join(path, "classification_report_" + model_name + ".png"))
    plt.close()
    return report


def get_confusion_matrix_plot(label_pred_list, label_truth_list, path, model_name="_"):
    y_preds = np.asarray(label_pred_list)
    y_truths = np.asarray(label_truth_list)
    cm = confusion_matrix(y_truths, y_preds)
    cm_df = pd.DataFrame(cm, index=list(np.unique(y_truths)), columns=list(np.unique(y_truths)))
    plt.figure(figsize=(50, 25))
    ax = sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Reds",
        annot_kws={"size": 45},
        cbar=True,
        linewidths=0.0029,
        linecolor="black",
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=45)
    plt.title("Confusion Matrix on the Test Set", fontsize=40, pad=20)
    plt.ylabel("Actual Values", fontsize=40)
    plt.xlabel("Predicted Values", fontsize=40)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=45)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=45)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "confusion_matrix_" + model_name + ".png"))
    plt.close()


def load_expert_truth_by_image():
    xlsx_path = os.path.join("dataset", "dataset_csv_files", "User_Agreement_2.xlsx")
    xlsx = pd.read_excel(xlsx_path).iloc[:329]
    xlsx = xlsx[xlsx["Unnamed: 3"].to_numpy() != "*"]
    xlsx = xlsx[xlsx["Incorrect Recommendation Condition"] != 4]
    return xlsx.set_index("Image Name")["Unnamed: 13"]


if __name__ == "__main__":
    csv_path = os.path.join("dataset", "dataset_csv_files", "User_Agreement_3.csv")
    results_dir = "results_295"
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    expert_truth = load_expert_truth_by_image()

    print("Total images in User_Agreement_3.csv:", len(df))

    labels = []
    new_preds = []
    no_agreement_num = 0
    correct_predictions_num = 0
    incorrect_predictions_num = 0

    for _, row in df.iterrows():
        image_name = row["Image Name"]
        correct_rec = row["Correct Recommendation"]

        if is_no_agreement(correct_rec):
            no_agreement_num += 1
            continue

        pred = row["Recommendation"]
        new_preds.append(pred)

        if is_true(correct_rec):
            labels.append(pred)
            correct_predictions_num += 1
        else:
            labels.append(expert_truth.loc[image_name])
            incorrect_predictions_num += 1

    print("Images with Correct Recommendation = No agreement:", no_agreement_num)
    print("Images under evaluation (metrics):", len(labels))
    print("Expert-marked correct recommendations:", correct_predictions_num)
    print("Expert-marked incorrect recommendations:", incorrect_predictions_num)

    new_preds = normalize_habibot_labels(new_preds)
    labels = normalize_habibot_labels(labels)

    print(Counter(labels))

    model_name = "eval_295"
    cls_report = get_classification_report_plot(new_preds, labels, results_dir, model_name=model_name)
    print("accuracy:", cls_report["accuracy"])
    print("macro avg:", cls_report["macro avg"])
    print("weighted avg:", cls_report["weighted avg"])
    get_confusion_matrix_plot(new_preds, labels, results_dir, model_name=model_name)

    label_counts = Counter(labels)
    plt.figure(figsize=(18, 12))
    bars = plt.barh(list(label_counts.keys()), list(label_counts.values()))
    plt.ylabel("Class Names", fontsize=25)
    plt.xlabel("Frequency", fontsize=20)
    plt.title("Label Frequency (Eval-Frame-295, n=290)", fontsize=20)
    plt.tick_params(axis="y", labelsize=20)
    plt.tick_params(axis="x", labelsize=12)
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(int(width)),
            va="center",
            fontsize=20,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "label_frequency.png"))
    plt.close()

    print("Results saved to:", os.path.abspath(results_dir))
