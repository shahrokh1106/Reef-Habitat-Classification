import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import json
import cv2
import pandas as pd
import shutil
from sklearn import svm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from collections import Counter
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from tensorflow.keras.applications import InceptionV3 
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import random
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import spektral
from spektral.layers import GCNConv
from tensorflow.keras.applications import InceptionV3
import tensorflow.keras.backend as K
import copy
import albumentations as A
from albumentations.core.composition import OneOf
import itertools
from spektral.layers import GATConv
from tqdm import tqdm 

SEED = 2
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ACTUAL_CLASS_NAMES = {"reef_barren": "Reef-Barren",
                      "reef_grazed": "Reef-Grazed",
                      "reef_kelp": "Reef-Kelp", 
                      "reef_partial_barren": "Reef-Partial-Barren",
                      "reef_partial_grazed":"Reef-Partial-Grazed",
                      "reef_vegetated": "Reef-Vegetated",
                      "unconsolidated": "Unconsolidated",
                      "reef_partial": "Reef-Partial"
                    }

def get_classification_report_plot(label_pred_list, label_truth_list,path, model_name = "_"):
  
    y_preds = np.asarray(list(label_pred_list))
    y_truths = np.asarray(list(label_truth_list))
    report = classification_report(y_truths, y_preds, zero_division=0, output_dict=True)
    report_data = {cls: {'precision': report[cls]['precision'], 
                        'recall': report[cls]['recall'], 
                        'f1-score': report[cls]['f1-score']} for cls in list(np.unique(y_truths))}
    
    
    report_df = pd.DataFrame(report_data).transpose()
    plt.figure(figsize=(8, 5))
    sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f', cbar=False)
    plt.title('Classification Report on the Test Set')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.grid("off")
    plt.tight_layout()
    plt.savefig(os.path.join(path, "classification_report_"+model_name+".png"))
    plt.close()


def get_confusion_matrix_plot (label_pred_list, label_truth_list, path, model_name = "_"):
    y_preds = np.asarray(label_pred_list)
    y_truths = np.asarray(label_truth_list)
    cm = confusion_matrix(y_truths, y_preds)
    cm_df = pd.DataFrame(cm, index = list(np.unique(y_truths)), columns = list(np.unique(y_truths)))
    plt.figure(figsize=(20,15))
    ax =sns.heatmap(cm_df, annot = True, fmt='d', cmap='Blues',annot_kws={"size": 16},cbar=False)
    plt.title('Confusion Matrix on the Test Set', fontsize=30)
    plt.ylabel('Actual Values', fontsize=30)
    plt.xlabel('Predicted Values', fontsize=30)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "confusion_matrix_"+model_name+".png"))
    plt.close()

def plot_split_distribution(train_labels, valid_labels, test_labels,export_path):
    train_class_counts = Counter(train_labels)
    valid_class_counts = Counter(valid_labels)
    test_class_counts = Counter(test_labels)

    classes = sorted(set(train_class_counts.keys()).union(valid_class_counts.keys(), test_class_counts.keys()))

    train_counts = [train_class_counts.get(cls, 0) for cls in classes]
    valid_counts = [valid_class_counts.get(cls, 0) for cls in classes]
    test_counts = [test_class_counts.get(cls, 0) for cls in classes]

    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    x = range(len(classes))

    train_bars = plt.bar([p - bar_width for p in x], train_counts, width=bar_width, label='Train', align='center')
    valid_bars = plt.bar(x, valid_counts, width=bar_width, label='Validation', align='center')
    test_bars = plt.bar([p + bar_width for p in x], test_counts, width=bar_width, label='Test', align='center')

    for bar in train_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')
    
    for bar in valid_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')
    
    for bar in test_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

    plt.xlabel('Class Labels')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Train, Validation, and Test Sets')
    plt.xticks(x, classes, rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(export_path,"label_frequency_splits.png"), bbox_inches='tight')
    plt.close()

def get_learning_curves(history,export_path,model_name):
    # with open(os.path.join(export_path,"training_data.json"), "w") as file:
    #     json.dump(history, file, indent=4)
    loss = history['loss']
    val_loss =history['val_loss']
    precision = history['precision']
    val_precision = history['val_precision']
    recall = history['recall']
    val_recall = history['val_recall']
    acc = history['acc']
    val_acc = history['val_acc']
    f1 = history['f1_score']
    val_f1 = history['val_f1_score']
    epochs = len(loss)
    style.use("bmh")
    fontsize = 20
    fontsize_legend = 15
    plt.figure(figsize=(30, 15))

    #loss
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs+1), loss, label='Training Loss')
    plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
    plt.legend(loc='upper right', fontsize=fontsize_legend)
    plt.ylabel('Loss', fontsize=fontsize)
    plt.title('Training and Validation Loss', fontsize=fontsize)
    plt.xlabel('epoch', fontsize=fontsize)

    # f1
    plt.subplot(2, 2, 4)
    plt.plot(range(1, epochs+1), f1, label='Training Weighted F1-score')
    plt.plot(range(1, epochs+1), val_f1, label='Validation Weighted F1-score')
    plt.legend(loc='lower right', fontsize=fontsize_legend)
    plt.ylabel('Weighted F1-score', fontsize=fontsize)
    plt.title('Training and Validation Weighted F1-score', fontsize=fontsize)
    plt.xlabel('epoch', fontsize=fontsize)
    # plt.xlim(0, 100)

    # precision          
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs+1), precision, label='Training Precision')
    plt.plot(range(1, epochs+1), val_f1, label='Validation Precision')
    plt.legend(loc='lower right', fontsize=fontsize_legend)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.title('Training and Validation Precision', fontsize=fontsize)
    plt.xlabel('epoch', fontsize=fontsize)
    # plt.xlim(0, 100)

    # recall          
    plt.subplot(2, 2, 3)
    plt.plot(range(1, epochs+1), recall, label='Training Recall')
    plt.plot(range(1, epochs+1), val_recall, label='Validation Recall')
    plt.legend(loc='lower right', fontsize=fontsize_legend)
    plt.ylabel('Recall', fontsize=fontsize)
    plt.title('Training and Validation Recall', fontsize=fontsize)
    plt.xlabel('epoch', fontsize=fontsize)
    # plt.xlim(0, 100)

    plt.savefig(os.path.join(export_path,"train_val_scores_"+model_name+".png"), bbox_inches='tight')
    plt.close()
######################################################################################
class FullAdjacency(layers.Layer):
    def call(self, inputs):
        # inputs: (batch_size, num_nodes, feature_dim)
        num_nodes = tf.shape(inputs)[1]
        adj = tf.ones((num_nodes, num_nodes), dtype=tf.float32)
        return tf.tile(tf.expand_dims(adj, 0), [tf.shape(inputs)[0], 1, 1])
    
    def get_config(self):
        config = super(FullAdjacency, self).get_config()
        return config

class PatchExtraction(layers.Layer):
    def __init__(self, patch_size=4, **kwargs):
        super(PatchExtraction, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, feature_map):
        patch_size = self.patch_size
        batch_size = tf.shape(feature_map)[0]
        H = tf.shape(feature_map)[1]
        W = tf.shape(feature_map)[2]
        channels = feature_map.shape[-1]
        # Crop the feature map so its dimensions are divisible by patch_size
        H_new = (H // patch_size) * patch_size
        W_new = (W // patch_size) * patch_size
        feature_map_cropped = feature_map[:, :H_new, :W_new, :]
        num_patches_h = H_new // patch_size
        num_patches_w = W_new // patch_size

        # Reshape to extract patches
        patches = tf.reshape(feature_map_cropped,
                             (batch_size, num_patches_h, patch_size, num_patches_w, patch_size, channels))
        # Rearrange dimensions: (batch_size, num_patches_h, num_patches_w, patch_size, patch_size, channels)
        patches = tf.transpose(patches, perm=[0, 1, 3, 2, 4, 5])
        # Merge patch grid dimensions: (batch_size, num_patches_h*num_patches_w, patch_size, patch_size, channels)
        patches = tf.reshape(patches,
                             (batch_size, num_patches_h * num_patches_w, patch_size, patch_size, channels))
        # Pool over spatial dimensions (global average pooling per patch)
        pooled_patches = tf.reduce_mean(patches, axis=[2, 3])
        return pooled_patches

    def get_config(self):
        config = super(PatchExtraction, self).get_config()
        config.update({"patch_size": self.patch_size})
        return config
    

class SerializableGATConv(GATConv):
    def call(self, *args, **kwargs):
        # Remove the 'training' argument if it exists.
        kwargs.pop("training", None)
        return super(SerializableGATConv, self).call(*args, **kwargs)
    
    def get_config(self):
        config = super(SerializableGATConv, self).get_config()
        return config
class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs shape: (batch_size, num_patches, channels)
        # Compute attention weights for each patch using a dense layer.
        attn_scores = layers.Dense(1, activation='sigmoid')(inputs)  # shape: (batch_size, num_patches, 1)
        # Multiply patch features by their corresponding attention weights.
        return inputs * attn_scores

    def get_config(self):
        return super(SpatialAttention, self).get_config()


def get_model_inceptionv3_fpn_gat(number_of_classes,image_size):
    base_model = InceptionV3(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
    base_model.trainable = True
    feature_maps = {
        "P3": base_model.get_layer("mixed2").output,   
        "P4": base_model.get_layer("mixed6").output,  
        "P5": base_model.get_layer("mixed10").output   
    }
    P3 = layers.Conv2D(256, (1, 1), padding="same")(feature_maps["P3"])
    P4 = layers.Conv2D(256, (1, 1), padding="same")(feature_maps["P4"])
    P5 = layers.Conv2D(256, (1, 1), padding="same")(feature_maps["P5"])
    

    pooled_P3 = PatchExtraction(patch_size=8)(P3)  
    pooled_P4 = PatchExtraction(patch_size=4)(P4)  
    pooled_P5 = PatchExtraction(patch_size=2)(P5) 
    
    
    adj_P3 = FullAdjacency()(pooled_P3)
    adj_P4 = FullAdjacency()(pooled_P4)
    adj_P5 = FullAdjacency()(pooled_P5)
    
    # Apply a Graph Attention (GAT) layer to each branch.
    gat_out_P3 = SerializableGATConv(
        channels=64,
        attn_heads=8,
        concat_heads=False,
        dropout_rate=0.5,
        activation='relu'
    )([pooled_P3, adj_P3])
    gat_out_P3 = tf.reduce_mean(gat_out_P3, axis=1)
    gat_out_P3 = layers.Dense(512, activation="relu")(gat_out_P3)
    
    gat_out_P4 = SerializableGATConv(
        channels=64,
        attn_heads=8,
        concat_heads=False,
        dropout_rate=0.5,
        activation='relu'
    )([pooled_P4, adj_P4])
    gat_out_P4 = tf.reduce_mean(gat_out_P4, axis=1)
    gat_out_P4 = layers.Dense(512, activation="relu")(gat_out_P4)
    
    gat_out_P5 = SerializableGATConv(
        channels=64,
        attn_heads=8,
        concat_heads=False,
        dropout_rate=0.5,
        activation='relu'
    )([pooled_P5, adj_P5])
    gat_out_P5 = tf.reduce_mean(gat_out_P5, axis=1)
    gat_out_P5 = layers.Dense(512, activation="relu")(gat_out_P5)
    
    # Fuse the features from all three branches.
    fused_feature = tf.concat([gat_out_P3, gat_out_P4, gat_out_P5], axis=-1)
    fused_feature = layers.Dropout(0.2)(fused_feature)
    fused_feature = layers.Dense(256, activation="relu")(fused_feature)
    fused_feature = layers.Dropout(0.2)(fused_feature)
    final_output = layers.Dense(number_of_classes,activation="softmax")(fused_feature)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=final_output)
    return model
######################################################################################
def get_model_inceptionv3_fpn(number_of_classes,image_size):
    def get_feature_extractor(include_top=False, weights='imagenet'):
        base_model = InceptionV3(input_shape=(image_size, image_size, 3), include_top=include_top, weights=weights)
        if not include_top:
            base_model.trainable = False
        return base_model

    base_model = get_feature_extractor(include_top=False, weights='imagenet')
    base_model.trainable = True  

    feature_maps = {
        "P3": base_model.get_layer("mixed2").output,   
        "P4": base_model.get_layer("mixed6").output,  
        "P5": base_model.get_layer("mixed10").output   
    }

    P5 = layers.Conv2D(256, (1, 1), padding="same")(feature_maps["P5"])
    P4 = layers.Conv2D(256, (1, 1), padding="same")(feature_maps["P4"])
    P3 = layers.Conv2D(256, (1, 1), padding="same")(feature_maps["P3"])

    target_size = (tf.shape(P5)[1], tf.shape(P5)[2])

    P4_resized = tf.image.resize(P4, target_size, method='bilinear')
    P3_resized = tf.image.resize(P3, target_size, method='bilinear')

    fused_features = layers.Concatenate()([P3_resized, P4_resized, P5])
    fused_features = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(fused_features)

    pooled_features = layers.GlobalAveragePooling2D()(fused_features)

    hidden_layer = layers.Dense(512, activation="relu", name="hidden_layer_1")(pooled_features)
    hidden_layer = layers.Dropout(0.3)(hidden_layer)
    output_layer = layers.Dense(number_of_classes,activation="softmax", name="output")(hidden_layer)

    model = tf.keras.Model(inputs=base_model.input, outputs=output_layer, name="fpn_multiclass_classifier")
    return model

def get_model_inceptionv3(number_of_classes,image_size):
    def get_feature_extractor(include_top=False, weights='imagenet'):
        feature_extractor = InceptionV3(input_shape=(image_size, image_size, 3), include_top=include_top, weights=weights)
        if include_top == False:
            feature_extractor.trainable = False
        return feature_extractor

    base_model = get_feature_extractor(include_top=False, weights='imagenet')
    base_model.trainable = True

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu', name='hidden_layer_1'),
        layers.Dropout(0.3),
        layers.Dense(number_of_classes, activation="softmax", name='output')
    ], name="multiclass_frame_classifier")
    return model

def get_model_inceptionv3_v2(number_of_classes, image_size):
    base_model = InceptionV3(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
    base_model.trainable = True  
    input_layer = base_model.input
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', name='hidden_layer_1')(x)
    x = layers.Dropout(0.3)(x)
    output_layer = layers.Dense(number_of_classes, activation="softmax", name='output')(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="custom_inception_v3")
    return model

   
   



def compute_weights(train_labels, label_map):
    num_classes = len(label_map)
    train_labels_int = [label_map[label] for label in train_labels]
    train_labels_encoded = to_categorical(train_labels_int, num_classes)
    train_class_indices = np.argmax(train_labels_encoded, axis=1)
    class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_class_indices), y=train_class_indices)
    class_weight_dict = dict(enumerate(class_weights))
    return class_weight_dict

def get_image_paths_and_labels(dataset_path):
    train_dataset_path = os.path.join(dataset_path,"train")
    val_dataset_path = os.path.join(dataset_path,"val")
    test_dataset_path = os.path.join(dataset_path,"test")

    train_images, train_labels, valid_images, valid_labels, test_images, test_labels = [],[],[],[],[],[]

    # train
    class_paths = glob.glob(os.path.join(train_dataset_path, "*"))
    for class_path in class_paths:
        class_name = os.path.basename(class_path)
        if class_name == "reef_partial_barren" or class_name== "reef_partial_grazed":
            continue
        for path in glob.glob(os.path.join(class_path, "*")):
            train_images.append(path)
            train_labels.append(ACTUAL_CLASS_NAMES[class_name])

    # val
    class_paths = glob.glob(os.path.join(val_dataset_path, "*"))
    for class_path in class_paths:
        class_name = os.path.basename(class_path)
        if class_name == "reef_partial_barren" or class_name== "reef_partial_grazed":
            continue
        for path in glob.glob(os.path.join(class_path, "*")):
            valid_images.append(path)
            valid_labels.append(ACTUAL_CLASS_NAMES[class_name])

    # test
    class_paths = glob.glob(os.path.join(test_dataset_path, "*"))
    for class_path in class_paths:
        class_name = os.path.basename(class_path)
        if class_name == "reef_partial_barren" or class_name== "reef_partial_grazed":
            continue
        for path in glob.glob(os.path.join(class_path, "*")):
            test_images.append(path)
            test_labels.append(ACTUAL_CLASS_NAMES[class_name])
    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels

def make_datasets(train_images, train_labels, valid_images, valid_labels, test_images, test_labels,batch_size=32,image_size=512):
    
    def preprocess_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (image_size, image_size))
        image = preprocess_input_inception(image)
        return image, label
    
    def preprocess_image_aug(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (image_size, image_size))
        # Data Augmentation:
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        angle = tf.random.uniform([], minval=-0.2, maxval=0.2)
        image = tfa.image.rotate(image, angles=angle, interpolation='BILINEAR')
        crop_fraction = tf.random.uniform([], 0.8, 1.0)  # between 80% and 100% of the image size
        original_shape = tf.shape(image)[:2]
        crop_size = tf.cast(tf.cast(original_shape, tf.float32) * crop_fraction, tf.int32)
        image = tf.image.random_crop(image, size=[crop_size[0], crop_size[1], 3])
        image = tf.image.resize(image, (image_size, image_size))
        #######################################
        image = preprocess_input_inception(image)
        return image, label

    def create_dataset(filenames, labels,   batch_size, is_training=True):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        SHUFFLE_BUFFER_SIZE = 1024
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        if is_training == True:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
            dataset = dataset.map(lambda filename, label: preprocess_image_aug(filename, label), num_parallel_calls=AUTOTUNE)
        else:
            dataset = dataset.map(lambda filename, label: preprocess_image(filename, label), num_parallel_calls=AUTOTUNE)
        
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    label_map = {label: idx for idx, label in enumerate(sorted(set(train_labels + valid_labels + test_labels)))}
    
    combined = list(zip(train_images, train_labels))
    random.Random(SEED).shuffle(combined)  
    train_images, train_labels = zip(*combined)
    train_images = list(train_images)
    train_labels = list(train_labels)


    # Convert string labels to integers
    train_labels_int = [label_map[label] for label in train_labels]
    valid_labels_int = [label_map[label] for label in valid_labels]
    test_labels_int = [label_map[label] for label in test_labels]
    num_classes = len(label_map)

    train_labels_encoded = to_categorical(train_labels_int, num_classes)
    valid_labels_encoded = to_categorical(valid_labels_int, num_classes)
    test_labels_encoded = to_categorical(test_labels_int, num_classes)
    train_dataset =create_dataset(train_images, train_labels_encoded,   batch_size, is_training=True)
    valid_dataset =create_dataset(valid_images, valid_labels_encoded,   batch_size, is_training=False)
    test_dataset =create_dataset(test_images, test_labels_encoded,   batch_size, is_training=False)
    return train_dataset,valid_dataset, test_dataset, label_map

if __name__ == '__main__':
    image_size = 512
    batch_size = 32
    epochs = 100
    patience = 10
    initial_learning_rate = 0.0001
    dataset_name= "frame7_dataset_cleaned"
    dataset_path = os.path.join("dataset",dataset_name)
    # model_names = ["inception", "inception_fpn", "inception_fpn_gat"]
    model_names = ["inception_2"]
    output_folder = "supervised_frame_classifier_results"

    train_images, train_labels, valid_images, valid_labels, test_images, test_labels = get_image_paths_and_labels(dataset_path)
    train_dataset,valid_dataset, test_dataset, label_map= make_datasets(train_images, train_labels, valid_images, valid_labels, test_images, test_labels,batch_size=batch_size,image_size=image_size)
    print("Label Map - class to index: ",label_map)
    number_of_classes = len(label_map)
    for model_name in model_names:
        model_save_path = os.path.join(output_folder, model_name)
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = model_save_path+"/{}".format(t)+"_"+dataset_name
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        evaluation_path = os.path.join(export_path, "evaluation")
        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)
        plot_split_distribution(train_labels, valid_labels, test_labels,os.path.join(evaluation_path))
        class_weight_dict = compute_weights(train_labels, label_map)
        if model_name=="inception":
            model = get_model_inceptionv3(number_of_classes = number_of_classes,image_size=image_size)
        elif model_name == "inception_2":
            model = get_model_inceptionv3_v2(number_of_classes = number_of_classes,image_size=image_size)
        elif model_name=="inception_fpn":
            model = get_model_inceptionv3_fpn(number_of_classes = number_of_classes,image_size=image_size)
        elif model_name=="inception_fpn_gat":
            model = get_model_inceptionv3_fpn_gat(number_of_classes = number_of_classes,image_size=image_size)
        
        model.trainable = True
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
        callbacks = [EarlyStopping(patience=patience, verbose=1, restore_best_weights=True, monitor='val_loss'),lr_scheduler]
        metrics = ["acc",tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall'),tfa.metrics.F1Score(num_classes=number_of_classes, average='weighted')]
        model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=initial_learning_rate),loss="categorical_crossentropy",metrics=metrics)
        model.build(input_shape=(None, image_size, image_size, 3))
        callbacks = [EarlyStopping(patience=15, verbose=1, restore_best_weights=True, monitor='val_loss'), lr_scheduler]
        history = model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset, class_weight = class_weight_dict,callbacks=callbacks)
        history = history.history
        model.save(export_path)

        # testing 
        i_to_c = {v: k for k, v in label_map.items()}
        print("Label Map - index to class: ",i_to_c)
   
        model = tf.keras.models.load_model(export_path)
        pred_labels = []
        truth_labels= []
        for x, y in tqdm(test_dataset):
            preds = model.predict(x, verbose = "0")
            preds = [i_to_c[i] for i in np.argmax(preds, axis=1)]
            truths =[i_to_c[i] for i in np.argmax(y, axis=1)]
            pred_labels.extend(preds)
            truth_labels.extend(truths)
        get_confusion_matrix_plot(pred_labels,truth_labels,evaluation_path,model_name)
        get_classification_report_plot(pred_labels,truth_labels,evaluation_path,model_name)
        get_learning_curves(history,evaluation_path,model_name)
        print("TESTING DONE! AND THE RESULTS HAVE BENN SAVED IN: ", evaluation_path)



        


