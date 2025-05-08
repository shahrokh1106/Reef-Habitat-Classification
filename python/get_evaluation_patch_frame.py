import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import json
import cv2
import pandas as pd
import shutil
from sklearn import svm
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import random
import numpy as np
import tensorflow as tf
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm 
from multiclassifier import *


from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficient
from tensorflow.keras.applications.convnext import preprocess_input as preprocess_input_convnext
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB0, EfficientNetV2L
from tensorflow.keras.applications import InceptionV3 
from tensorflow.keras.applications import ConvNeXtLarge 
from tensorflow.keras.applications import ConvNeXtBase 
from tensorflow.keras.applications import ConvNeXtSmall
from tensorflow.keras.applications import Xception 
from tensorflow.keras.applications import DenseNet201 
from tensorflow.keras.applications import InceptionResNetV2



SEED = 2
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ACTUAL_CLASS_NAMES_frame = {"reef_barren": "Reef-Barren",
                      "reef_grazed": "Reef-Grazed",
                      "reef_kelp": "Reef-Kelp", 
                      "reef_partial_barren": "Reef-Partial-Barren",
                      "reef_partial_grazed":"Reef-Partial-Grazed",
                      "reef_vegetated": "Reef-Vegetated",
                      "unconsolidated": "Unconsolidated",
                      "reef_partial": "Reef-Partial"
                    }
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
            train_labels.append(ACTUAL_CLASS_NAMES_frame[class_name])

    # val
    class_paths = glob.glob(os.path.join(val_dataset_path, "*"))
    for class_path in class_paths:
        class_name = os.path.basename(class_path)
        if class_name == "reef_partial_barren" or class_name== "reef_partial_grazed":
            continue
        for path in glob.glob(os.path.join(class_path, "*")):
            valid_images.append(path)
            valid_labels.append(ACTUAL_CLASS_NAMES_frame[class_name])

    # test
    class_paths = glob.glob(os.path.join(test_dataset_path, "*"))
    for class_path in class_paths:
        class_name = os.path.basename(class_path)
        if class_name == "reef_partial_barren" or class_name== "reef_partial_grazed":
            continue
        for path in glob.glob(os.path.join(class_path, "*")):
            test_images.append(path)
            test_labels.append(ACTUAL_CLASS_NAMES_frame[class_name])
    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels

def make_datasets(model_name,train_images, train_labels, valid_images, valid_labels, test_images, test_labels,batch_size=32,image_size=512):
    
    def preprocess_image(image_path, label,model_name=model_name):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (image_size, image_size))
        if model_name=="inception":
            image = preprocess_input_inception(image)
        elif model_name=="efficient" or model_name=="efficientL":
            image = preprocess_input_efficient(image) 
        elif model_name =="resnet":
            image = preprocess_input_resnet(image) 
        elif model_name =="convnextB" or model_name =="convnextS":
            image = preprocess_input_convnext(image) 
        elif model_name =="xception":
            image = preprocess_input_xception(image) 
        elif model_name =="densenet":
            image = preprocess_input_densenet(image)  
        elif model_name =="inception_resnet":
            image = preprocess_input_inception_resnet_v2(image)
        else:
            raise Exception ("Model Name Not Found")
        return image, label
    
    def preprocess_image_aug(image_path, label, model_name=model_name):
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
        if model_name=="inception":
            image = preprocess_input_inception(image)
        elif model_name=="efficient" or model_name=="efficientL":
            image = preprocess_input_efficient(image) 
        elif model_name =="resnet":
            image = preprocess_input_resnet(image)
        elif model_name =="convnextB" or model_name =="convnextS":
            image = preprocess_input_convnext(image) 
        elif model_name =="xception":
            image = preprocess_input_xception(image) 
        elif model_name =="densenet":
            image = preprocess_input_densenet(image)  
        elif model_name =="inception_resnet":
            image = preprocess_input_inception_resnet_v2(image)
        else:
            raise Exception ("Model Name Not Found")
        return image, label

    def create_dataset(filenames, labels, model_name=model_name,  batch_size=batch_size, is_training=True):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        SHUFFLE_BUFFER_SIZE = 1024
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        if is_training == True:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
            dataset = dataset.map(lambda filename, label: preprocess_image_aug(filename, label,model_name=model_name), num_parallel_calls=AUTOTUNE)
        else:
            dataset = dataset.map(lambda filename, label: preprocess_image(filename, label,model_name=model_name), num_parallel_calls=AUTOTUNE)
        
        
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
    train_dataset =create_dataset(train_images, train_labels_encoded, model_name,  batch_size, is_training=True)
    valid_dataset =create_dataset(valid_images, valid_labels_encoded, model_name,  batch_size, is_training=False)
    test_dataset =create_dataset(test_images, test_labels_encoded, model_name,  batch_size, is_training=False)
    return train_dataset,valid_dataset, test_dataset, label_map

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
    return report



def get_confusion_matrix_plot (label_pred_list, label_truth_list, path, model_name = "_"):

    y_preds = np.asarray(label_pred_list)
    y_truths = np.asarray(label_truth_list)
    cm = confusion_matrix(y_truths, y_preds)
    cm_df = pd.DataFrame(cm, index = list(np.unique(y_truths)), columns = list(np.unique(y_truths)))
    plt.figure(figsize=(30,25))
    ax =sns.heatmap(cm_df, annot = True, fmt='d', cmap='Reds',annot_kws={"size": 35},cbar=True,linewidths=0.0029, linecolor="black")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30) 
    plt.title('Confusion Matrix on the Test Set', fontsize=40,pad=20)
    plt.ylabel('Actual Values', fontsize=40)
    plt.xlabel('Predicted Values', fontsize=40)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=35)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=35)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "confusion_matrix_"+model_name+".png"))
    plt.close()

def get_frame_model_evaluations(dataset_name,model_name,model_path):
    image_size = 512
    batch_size = 32
    dataset_path = os.path.join("dataset",dataset_name)
    train_images, train_labels, valid_images, valid_labels, test_images, test_labels = get_image_paths_and_labels(dataset_path)
    train_dataset,valid_dataset, test_dataset, label_map= make_datasets(model_name,train_images, train_labels, valid_images, valid_labels, test_images, test_labels,batch_size=batch_size,image_size=image_size)
    print("Label Map for the Frame Classifier - class to index: ",label_map)
    number_of_classes = len(label_map)
    i_to_c = {v: k for k, v in label_map.items()}
    model = tf.keras.models.load_model(model_path)
    model.trainable = False
    pred_labels = []
    truth_labels= []
    for x, y in tqdm(test_dataset):
        preds = model.predict(x, verbose = "0")
        preds = [i_to_c[i] for i in np.argmax(preds, axis=1)]
        truths =[i_to_c[i] for i in np.argmax(y, axis=1)]
        pred_labels.extend(preds)
        truth_labels.extend(truths)
    report = get_classification_report_plot(pred_labels,truth_labels,os.path.dirname(model_path),"classifier_"+model_name)
    get_confusion_matrix_plot (pred_labels, truth_labels, os.path.dirname(model_path), model_name = "frame_classifier_"+model_name)
    print("*********************** Frame Classifier ***********************")
    print(report["accuracy"])
    print(report["macro avg"])
    print("****************************************************************")

    



def get_patch_model_evaluations(patch_dataset_name,patch_model_name,patch_model_path):
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
    patches_path = os.path.join("dataset",patch_dataset_name) 
    MyClassifier = MulticlassClassifier(data_dir=patches_path, 
                                    patch_size_ratio = 1,
                                    remove_classes=["Mobile invertebrate", "Unscorable", "Sessile invertebrate community"],
                                    to_be_combined=["Bare rock", "Turf", "Encrusting algae","Filamentous algae", "grazed rock"],
                                    outlier_classes=[],
                                    sample_ratios={},
                                    batch_size=256,
                                    model_name=patch_model_name,
                                    epochs=100,
                                    initial_learning_rate=0.001,
                                    decay_steps=1236,
                                    decay_rate=0.9,
                                    lr_schedule_flag=False,
                                    full_training_flag=True,
                                    dense_number=512,
                                    dropout_ratio=0.3,
                                    loss_function='categorical_crossentropy',
                                    last_activation='softmax',
                                    patience=15,
                                    model_save_path=os.path.dirname(patch_model_path),
                                    verbose=False)  
    
    model=tf.keras.models.load_model(patch_model_path,compile=False, custom_objects={"LayerScale": LayerScale})
    model.trainable = False
    test_dataset = MyClassifier.test_ds
    i_to_c = {v: k for k, v in MyClassifier.train_class_names.items()}
    y_pred = model.predict(test_dataset)
    y_hat = np.zeros_like(y_pred)
    y_hat[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1
    y_true = MyClassifier.test_labels_
    pred_labels = []
    truth_labels= []
    for i in range(len(y_hat)):
        pred_labels.append(i_to_c[int(np.argmax(y_hat[i]))])
        truth_labels.append(i_to_c[int(np.argmax(y_true[i]))])
    report = get_classification_report_plot(pred_labels,truth_labels,os.path.dirname(patch_model_path),"patch_classifier_"+patch_model_name)
    get_confusion_matrix_plot (pred_labels, truth_labels, os.path.dirname(patch_model_path), model_name = "frame_classifier_"+patch_model_name)
    print("*********************** Patch Classifier ***********************")
    print(report["accuracy"])
    print(report["macro avg"])
    print("****************************************************************")
    


if __name__ == '__main__':
    frame_dataset_name= "frame7_dataset_cleaned"
    frame_model_name = "inception"
    frame_model_path = os.path.join("trained_classifiers","frame_classifier", "frame_classifer.h5")
    # get_frame_model_evaluations(frame_dataset_name,frame_model_name,frame_model_path)
    patch_dataset_name = "patches"
    patch_model_name = "convnextB"
    patch_model_path = os.path.join("trained_classifiers","patch_classifier", "patch_classifier.h5")
    get_patch_model_evaluations(patch_dataset_name, patch_model_name, patch_model_path)