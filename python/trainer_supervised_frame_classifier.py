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

def get_model(model_name, number_of_classes, image_size,full_training):
    if model_name=="inception":
        base_model = InceptionV3(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
    elif model_name=="efficient":
        base_model = EfficientNetB0(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
    elif model_name == "efficientL":
        base_model = EfficientNetV2L(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
    elif model_name == "resnet" :
        base_model = ResNet50(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
    elif model_name == "inception":
        base_model = InceptionV3(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
    elif model_name == "convnextB":
        base_model = ConvNeXtBase(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
    elif model_name == "convnextS":
        base_model = ConvNeXtSmall(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
    elif model_name == "xception":
        base_model = Xception(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
    elif model_name == "densenet":
        base_model = DenseNet201(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
    elif model_name == "inception_resnet":
        base_model = InceptionResNetV2(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
    else:
        raise Exception ("Model Name Not Found")
    base_model.trainable = full_training 
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

if __name__ == '__main__':
    full_training = True
    image_size = 512
    batch_size = 32
    epochs = 100
    patience = 15
    initial_learning_rate = 0.0001
    dataset_name= "frame7_dataset_cleaned"
    dataset_path = os.path.join("dataset",dataset_name)
    model_names = ["inception","efficient", "efficientL", "resnet","convnextB","convnextS","xception","densenet","inception_resnet"]
    model_names = ["inception_resnet"]
    output_folder = os.path.join("backbone_selection_results_frame","best_backbone_full_training")
    for model_name in model_names:
        train_images, train_labels, valid_images, valid_labels, test_images, test_labels = get_image_paths_and_labels(dataset_path)
        train_dataset,valid_dataset, test_dataset, label_map= make_datasets(model_name,train_images, train_labels, valid_images, valid_labels, test_images, test_labels,batch_size=batch_size,image_size=image_size)
        print("Label Map - class to index: ",label_map)
        number_of_classes = len(label_map)
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
        model = get_model(model_name,number_of_classes = number_of_classes,image_size=image_size,full_training=full_training)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
        metrics = ["acc",tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall'),tfa.metrics.F1Score(num_classes=number_of_classes, average='weighted')]
        model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=initial_learning_rate),loss="categorical_crossentropy",metrics=metrics)
        model.build(input_shape=(None, image_size, image_size, 3))
        callbacks = [EarlyStopping(patience=patience, verbose=1, restore_best_weights=True, monitor='val_loss'), lr_scheduler]
        history = model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset, class_weight = class_weight_dict,callbacks=callbacks)
        history = history.history
        try:
            model.save(os.path.join(export_path,model_name+".h5"))
        except:
            try:
                model.save_weights(os.path.join(export_path,model_name+".h5"))
            except:
                try:
                    model.save(export_path)
                except:
                    print(model_name,": could not be saved")

        # testing 
        i_to_c = {v: k for k, v in label_map.items()}
        print("Label Map - index to class: ",i_to_c)
   
        
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


    # convnextb = [0.89, 0.67, 0.87, 0.85, 0.81]
    # convnexts = [0.83, 0.69, 0.88, 0.82, 0.75]
    # densenet = [0.88, 0.71, 0.90, 0.83, 0.82]
    # efficientnet = [0.91, 0.74, 0.91, 0.83, 0.82]
    # efficientnetl =  [0.92, 0.75, 0.90, 0.83, 0.83]
    # inception = [0.86, 0.71, 0.91, 0.84, 0.79]
    # inception_resnet = [0.87, 0.65, 0.88, 0.81, 0.77]
    # resnet = [0.88, 0.74, 0.90, 0.84, 0.81]
    # xception = [0.85, 0.67, 0.87, 0.79, 0.74]

    # print(np.average(convnextb),np.average(convnexts),np.average(densenet),np.average(efficientnet),np.average(efficientnetl),np.average(inception), np.average(inception_resnet),np.average(resnet), np.average(xception))
    
    



        


