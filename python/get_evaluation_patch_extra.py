import os
import shutil
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from multiclassifier import (
    MulticlassClassifier,
    get_paths_labels_from_dir,
    create_dataset,
)

SEED = 2
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

TEST_EXTRA_PATH = r"E:\shah\habibot\patches\test_extra"
PATCHES_ROOT_PATH = r"E:\shah\habibot\patches"

REMOVE_CLASSES = [
    "Mobile invertebrate",
    "Unscorable",
    "Sessile invertebrate community",
]
TO_BE_COMBINED = [
    "Bare rock",
    "Turf",
    "Encrusting algae",
    "Filamentous algae",
    "grazed rock",
]

CARPOPHYLLUM_FOLDER_NAME = "Carpophyllum_spp_incorrect"
CARPOPHYLLUM_CSV_NAME = "Carpophyllum_spp_incorrect.csv"


def is_carpophyllum_ground_truth(label):
    if label is None:
        return False
    normalized = str(label).strip().lower()
    return normalized.startswith("carpophyllum")


def save_carpophyllum_misclassified_patches(
    extra_paths,
    y_pred,
    y_true,
    i_to_c,
    out_dir,
):
    """Copy Carpophyllum spp. patches the model misclassified; write confidence CSV."""
    incorrect_dir = os.path.join(out_dir, CARPOPHYLLUM_FOLDER_NAME)
    os.makedirs(incorrect_dir, exist_ok=True)

    rows = []
    n = len(extra_paths)
    for i in range(n):
        true_name = i_to_c[int(np.argmax(y_true[i]))]
        if not is_carpophyllum_ground_truth(true_name):
            continue
        pred_idx = int(np.argmax(y_pred[i]))
        pred_name = i_to_c[pred_idx]
        pred_confidence = float(y_pred[i][pred_idx])
        if pred_name == true_name:
            continue

        src_path = extra_paths[i]
        base = os.path.basename(src_path)
        stem, ext = os.path.splitext(base)
        dest_name = (
            f"{stem}__true-{true_name}__pred-{pred_name}__conf-{pred_confidence:.4f}{ext}"
        )
        dest_name = dest_name.replace(" ", "_").replace("/", "-")
        dest_path = os.path.join(incorrect_dir, dest_name)
        shutil.copy2(src_path, dest_path)

        row = {
            "source_path": src_path,
            "saved_filename": dest_name,
            "true_label": true_name,
            "predicted_label": pred_name,
            "predicted_confidence": pred_confidence,
        }
        for class_idx, class_name in i_to_c.items():
            row[f"prob_{class_name}"] = float(y_pred[i][class_idx])
        rows.append(row)

    csv_path = os.path.join(incorrect_dir, CARPOPHYLLUM_CSV_NAME)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(
        "Carpophyllum spp. misclassified patches:",
        len(rows),
        "saved in",
        incorrect_dir,
    )
    if len(rows):
        print("CSV:", csv_path)
    return incorrect_dir, csv_path


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
    plt.title("Classification Report on the Test Extra Set")
    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    plt.grid("off")
    plt.tight_layout()
    plt.savefig(os.path.join(path, "classification_report_" + model_name + ".png"))
    plt.close()

    report_csv_df = pd.DataFrame(report).transpose()
    numeric_cols = report_csv_df.select_dtypes(include=[np.number]).columns
    report_csv_df[numeric_cols] = report_csv_df[numeric_cols].round(2)
    report_csv_df.to_csv(
        os.path.join(path, "classification_report_" + model_name + ".csv")
    )
    return report


def get_confusion_matrix_plot(label_pred_list, label_truth_list, path, model_name="_"):
    y_preds = np.asarray(label_pred_list)
    y_truths = np.asarray(label_truth_list)
    cm = confusion_matrix(y_truths, y_preds)
    cm_df = pd.DataFrame(
        cm, index=list(np.unique(y_truths)), columns=list(np.unique(y_truths))
    )
    plt.figure(figsize=(30, 25))
    ax = sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Reds",
        annot_kws={"size": 35},
        cbar=True,
        linewidths=0.0029,
        linecolor="black",
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    plt.title("Confusion Matrix on the Test Extra Set", fontsize=40, pad=20)
    plt.ylabel("Actual Values", fontsize=40)
    plt.xlabel("Predicted Values", fontsize=40)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=35)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=35)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "confusion_matrix_" + model_name + ".png"))
    plt.close()


def remap_extra_labels_to_train_indices(extra_labels, extra_class_names, train_class_names):
    """Align test_extra integer labels with indices used at training time."""
    idx_to_name = {v: k for k, v in extra_class_names.items()}
    remapped = []
    for lab in extra_labels:
        class_name = idx_to_name[lab]
        if class_name not in train_class_names:
            raise KeyError(
                "Class {!r} in test_extra is not in the training class map: {}".format(
                    class_name, list(train_class_names.keys())
                )
            )
        remapped.append(train_class_names[class_name])
    return remapped


def prepare_test_extra_dataset(
    test_extra_path,
    train_class_names,
    patch_model_name,
    batch_size=16,
    patch_size_ratio=1,
):
    extra_paths, extra_labels, extra_class_names = get_paths_labels_from_dir(
        data_dir=test_extra_path,
        remove_classes=REMOVE_CLASSES,
        to_be_combined=TO_BE_COMBINED,
        outlier_classes=[],
        sample_ratios={},
        verbose=False,
    )
    extra_labels_train_idx = remap_extra_labels_to_train_indices(
        extra_labels, extra_class_names, train_class_names
    )
    num_classes = len(train_class_names)
    extra_labels_encoded = to_categorical(extra_labels_train_idx, num_classes)
    test_extra_ds = create_dataset(
        extra_paths,
        extra_labels_encoded,
        batch_size,
        patch_model_name,
        patch_size_ratio,
        is_training=False,
    )
    return test_extra_ds, extra_labels_encoded, extra_paths


def get_patch_model_evaluations_test_extra(
    patches_path,
    test_extra_path,
    patch_model_name,
    patch_model_path,
):
    class LayerScale(layers.Layer):
        """Layer scale module.

        References:
        - https://arxiv.org/abs/2103.17239

        Args:
        init_values (float): Initial value for layer scale. Should be within
            [0, 1].
        projection_dim (int): Projection dimensionality.

        Returns:
        Tensor multiplied to the scale.
        """

        def __init__(self, init_values, projection_dim, **kwargs):
            super().__init__(**kwargs)
            self.init_values = init_values
            self.projection_dim = projection_dim

        def build(self, input_shape):
            self.gamma = tf.Variable(
                self.init_values * tf.ones((self.projection_dim,))
            )

        def call(self, x):
            return x * self.gamma

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "init_values": self.init_values,
                    "projection_dim": self.projection_dim,
                }
            )
            return config

    MyClassifier = MulticlassClassifier(
        data_dir=patches_path,
        patch_size_ratio=1,
        remove_classes=REMOVE_CLASSES,
        to_be_combined=TO_BE_COMBINED,
        outlier_classes=[],
        sample_ratios={},
        batch_size=16,
        model_name=patch_model_name,
        epochs=100,
        initial_learning_rate=0.001,
        decay_steps=1236,
        decay_rate=0.9,
        lr_schedule_flag=False,
        full_training_flag=True,
        dense_number=512,
        dropout_ratio=0.3,
        loss_function="categorical_crossentropy",
        last_activation="softmax",
        patience=15,
        model_save_path=os.path.dirname(patch_model_path),
        verbose=False,
    )

    test_extra_ds, test_extra_labels_, extra_paths = prepare_test_extra_dataset(
        test_extra_path=test_extra_path,
        train_class_names=MyClassifier.train_class_names,
        patch_model_name=patch_model_name,
        batch_size=16,
        patch_size_ratio=1,
    )

    model = tf.keras.models.load_model(
        patch_model_path, compile=False, custom_objects={"LayerScale": LayerScale}
    )

    model.trainable = False
    i_to_c = {v: k for k, v in MyClassifier.train_class_names.items()}
    i_to_c[6] = "BrLfa"

    y_pred = model.predict(test_extra_ds)
    y_hat = np.zeros_like(y_pred)
    y_hat[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1
    y_true = test_extra_labels_
    pred_labels = []
    truth_labels = []
    for i in range(len(y_hat)):
        pred_labels.append(i_to_c[int(np.argmax(y_hat[i]))])
        truth_labels.append(i_to_c[int(np.argmax(y_true[i]))])
    eval_name = "patch_classifier_" + patch_model_name + "_test_extra"
    out_dir = os.path.dirname(patch_model_path)
    report = get_classification_report_plot(
        pred_labels, truth_labels, out_dir, eval_name
    )
    get_confusion_matrix_plot(pred_labels, truth_labels, out_dir, model_name=eval_name)
    save_carpophyllum_misclassified_patches(
        extra_paths, y_pred, test_extra_labels_, i_to_c, out_dir
    )
    print("*********************** Patch Classifier (Test Extra) ***********************")
    print(report["accuracy"])
    print(report["macro avg"])
    print("Results saved in:", out_dir)
    print("********************************************************************************")


if __name__ == "__main__":
    patch_model_name = "convnextB"
    patch_model_path = os.path.join(
        "trained_classifiers", "patch_classifier", "patch_classifier.h5"
    )
    get_patch_model_evaluations_test_extra(
        PATCHES_ROOT_PATH,
        TEST_EXTRA_PATH,
        patch_model_name,
        patch_model_path,
    )
