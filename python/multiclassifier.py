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


from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficient
from tensorflow.keras.applications.convnext import preprocess_input as preprocess_input_convnext
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nasnet


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB0, EfficientNetV2L
from tensorflow.keras.applications import InceptionV3 
from tensorflow.keras.applications import ConvNeXtLarge 
from tensorflow.keras.applications import ConvNeXtBase 
from tensorflow.keras.applications import ConvNeXtSmall
from tensorflow.keras.applications import Xception 
from tensorflow.keras.applications import DenseNet201 
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import NASNetLarge


from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from time import time
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from datetime import datetime
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tqdm import tqdm
import tensorflow_addons as tfa
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Layer
from tensorflow.keras.initializers import Constant 


class MultiHeadSelfAttention(layers.Layer):
        def __init__(self, num_heads, key_dim):
            super(MultiHeadSelfAttention, self).__init__()
            self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

        def call(self, inputs):
            # Self-attention: query, key, and value are the same
            return self.attention(inputs, inputs)
        

############################################################################################################################
def get_feature_extractor (model_name, include_top=False, weights='imagenet'):
    if model_name == "efficientL":
        feature_extractor = EfficientNetV2L(input_shape=(224, 224, 3), include_top=include_top, weights=weights)
    if model_name == "efficient":
        feature_extractor = EfficientNetB0(input_shape=(224, 224, 3), include_top=include_top, weights=weights)
    elif model_name == "resnet" :
        feature_extractor = ResNet50(input_shape=(224, 224, 3), include_top=include_top, weights=weights)
    elif model_name == "inception":
        feature_extractor = InceptionV3(input_shape=(299, 299, 3), include_top=include_top, weights=weights)
    elif model_name == "convnext":
        feature_extractor = ConvNeXtLarge(input_shape=(224, 224, 3), include_top=include_top, weights=weights)
    elif model_name == "convnextB":
        feature_extractor = ConvNeXtBase(input_shape=(224, 224, 3), include_top=include_top, weights=weights)
    elif model_name == "convnextS":
        feature_extractor = ConvNeXtSmall(input_shape=(224, 224, 3), include_top=include_top, weights=weights)
    elif model_name == "xception":
        feature_extractor = Xception(input_shape=(299, 299, 3), include_top=include_top, weights=weights)
    elif model_name == "densenet":
        feature_extractor = DenseNet201(input_shape=(224, 224, 3), include_top=include_top, weights=weights)
    elif model_name == "inception_resnet":
        feature_extractor = InceptionResNetV2(input_shape=(299, 299, 3), include_top=include_top, weights=weights)
    elif model_name == "vgg16":
        feature_extractor = VGG16(input_shape=(224, 224, 3), include_top=include_top, weights=weights)
    elif model_name == "vgg19":
        feature_extractor = VGG19(input_shape=(224, 224, 3), include_top=include_top, weights=weights)
    elif model_name == "nasnet":
        feature_extractor = NASNetLarge(input_shape=(224, 224, 3), include_top=include_top, weights=weights)
    
    if include_top==False:
        feature_extractor.trainable = False
    return feature_extractor
    
def preprocess_inception(image, label):
    image = preprocess_input_inception(image)  
    return image, label

def preprocess_efficient(image, label):
    image = preprocess_input_efficient(image) 
    return image, label

def preprocess_resnet(image, label):
    image = preprocess_input_resnet(image)  
    return image, label

def preprocess_convnext(image, label):
    image = preprocess_input_convnext(image)  
    return image, label

def preprocess_xception(image, label):
    image = preprocess_input_xception(image)  
    return image, label

def preprocess_densenet(image, label):
    image = preprocess_input_densenet(image)  
    return image, label

def preprocess_vgg16(image, label):
    image = preprocess_input_vgg16(image)  
    return image, label

def preprocess_vgg19(image, label):
    image = preprocess_input_vgg19(image)  
    return image, label
    
def preprocess_inception_resnet_v2(image, label):
    image = preprocess_input_inception_resnet_v2(image)  
    return image, label

def preprocess_nasnet(image, label):
    image = preprocess_input_nasnet(image)  
    return image, label
    
def parse_function(filename, label, model_name = "efficient", patch_size_ratio=1.0):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    
    CHANNELS = 3
    if  model_name == "efficient" or model_name == "efficientL":
        preprocess = preprocess_efficient
        IMAGE_SIZE = 224
    elif model_name == "resnet":
        preprocess = preprocess_resnet
        IMAGE_SIZE = 224
    elif model_name == "inception":
        preprocess = preprocess_inception
        IMAGE_SIZE = 299
    elif model_name == "convnext" or  model_name == "convnextB" or  model_name == "convnextS":
        preprocess = preprocess_convnext
        IMAGE_SIZE = 224
    elif model_name == "xception":
        preprocess = preprocess_xception
        IMAGE_SIZE = 299
    elif model_name == "densenet":
        preprocess = preprocess_densenet
        IMAGE_SIZE = 224
    elif model_name == "inception_resnet":
        preprocess = preprocess_inception_resnet_v2
        IMAGE_SIZE = 299
    elif model_name == "vgg16":
        preprocess = preprocess_vgg16
        IMAGE_SIZE = 224
    elif model_name == "vgg19":
        preprocess = preprocess_vgg19
        IMAGE_SIZE = 224
    elif model_name == "nasnet":
        preprocess = preprocess_nasnet
        IMAGE_SIZE = 224
        
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    image = tf.image.central_crop(image_decoded, patch_size_ratio)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    
    # preprocess
    image, label = preprocess(image, label)
    return image, label

def parse_function_aug(filename, label, model_name = "efficient", patch_size_ratio=1.0):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    CHANNELS = 3
    if  model_name == "efficient" or model_name == "efficientL":
        preprocess = preprocess_efficient
        IMAGE_SIZE = 224
    elif model_name == "resnet":
        preprocess = preprocess_resnet
        IMAGE_SIZE = 224
    elif model_name == "inception":
        preprocess = preprocess_inception
        IMAGE_SIZE = 299
    elif model_name == "convnext" or  model_name == "convnextB" or  model_name == "convnextS":
        preprocess = preprocess_convnext
        IMAGE_SIZE = 224
    elif model_name == "xception":
        preprocess = preprocess_xception
        IMAGE_SIZE = 299
    elif model_name == "densenet":
        preprocess = preprocess_densenet
        IMAGE_SIZE = 224
    elif model_name == "inception_resnet":
        preprocess = preprocess_inception_resnet_v2
        IMAGE_SIZE = 299
    elif model_name == "vgg16":
        preprocess = preprocess_vgg16
        IMAGE_SIZE = 224
    elif model_name == "vgg19":
        preprocess = preprocess_vgg19
        IMAGE_SIZE = 224
    elif model_name == "nasnet":
        preprocess = preprocess_nasnet
        IMAGE_SIZE = 224
        
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    image = tf.image.central_crop(image_decoded, patch_size_ratio)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    
    # Data Augmentation
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    angle = tf.random.uniform([], minval=-0.2, maxval=0.2)
    image = tfa.image.rotate(image, angles=angle, interpolation='BILINEAR')
    # preprocess
    image, label = preprocess(image, label)
    return image, label

def get_paths_labels_from_dir (data_dir, remove_classes = [], to_be_combined=[] ,outlier_classes = [],sample_ratios = {}, verbose = True, model_name = "inception", batch_size = 256):
    file_paths_dict = dict()
    labels_dict = dict()
    classes = os.listdir(data_dir)
    if verbose:
        print("-"*100)
        print("Number of classes to be removed: ", len(remove_classes))
        if len(remove_classes)!=0:
            print("Removed classes: ", remove_classes)
    for rc in remove_classes:
        classes.remove(rc)
    if verbose:
        print("-"*100)
        print("Original number of classes: ", len(classes))
    for class_index, class_name in enumerate(classes):
        file_paths = []
        class_dir = os.path.join(data_dir, class_name)
        if verbose:
            print("-"*100)
        if class_name in outlier_classes:
            if verbose:
                print("Number of samples in "+class_name+": ", len(glob.glob(os.path.join( os.path.join(data_dir, class_name), "*"))))
            inlier_class_paths,outlier_class_paths = remove_outliers (data_dir, class_name, model_name =model_name, batch_size = batch_size)
            if verbose:
                print("Number of samples in "+class_name+" after removing outlires: ", len(inlier_class_paths))
                
            if class_name in sample_ratios:
                ratio = sample_ratios[class_name]
            else:
                ratio = 1.0
            num_samples = int(len(inlier_class_paths) * ratio)
            inlier_class_paths = random.sample(inlier_class_paths, num_samples)
            if verbose:
                print("Number of samples in "+class_name+" after random subsampling with ratio "+str(ratio)+": ", len(inlier_class_paths))
            for file_name in inlier_class_paths:
                file_paths.append(os.path.join(class_dir, file_name))
        else:
            class_paths = os.listdir(class_dir)
            if verbose:
                print("Number of samples in "+class_name+": ", len(class_paths))
            if class_name in sample_ratios:
                ratio = sample_ratios[class_name]
            else:
                ratio = 1.0
            num_samples = int(len(class_paths) * ratio)
            class_paths = random.sample(class_paths, num_samples)
            if verbose:
                print("Number of samples in "+class_name+" after random subsampling with ratio "+str(ratio)+": ", len(class_paths))
            for file_name in class_paths:
                file_paths.append(os.path.join(class_dir, file_name))
                
        file_paths_dict.update({class_name:file_paths})
        
    if len(to_be_combined)<=1:
        final_file_paths = []
        final_labels = []
        class_names = {}
        for index,class_name in enumerate(file_paths_dict.keys()):
            if class_name not in class_names:
                class_names.update({class_name: index})
            final_file_paths+=file_paths_dict[class_name]
            final_labels+=[index for _ in range(len(file_paths_dict[class_name]))]
        if verbose:
            print("-"*100)
            print("Total number of samples in all classes: ", len(final_file_paths))
            print("Total number of classes: ", len(class_names))
            print("Classes: ", list(class_names.keys()))
            print("-"*100)
    else:
        final_file_paths = []
        final_labels = []
        class_names = {}
        index = 0
        for class_name in file_paths_dict.keys():
            if class_name not in to_be_combined[:-1]:
                if class_name not in class_names:
                    class_names.update({class_name: index})
                final_file_paths+=file_paths_dict[class_name]
                final_labels+=[index for _ in range(len(file_paths_dict[class_name]))]
                index+=1
        combined_classes_paths = []
        combined_classes_labels = []
        new_class_name = to_be_combined[-1]
        class_names.update({new_class_name: index})
        for class_name in to_be_combined[:-1]:
            combined_classes_paths+=file_paths_dict[class_name]
            combined_classes_labels+=[index for _ in range(len(file_paths_dict[class_name]))]
        final_file_paths+=combined_classes_paths
        final_labels+=combined_classes_labels
        if verbose:
            print("-"*100)
            print("Total number of samples in all classes: ", len(final_file_paths))
            print("Total number of classes: ", len(class_names))
            print("Classes: ", list(class_names.keys()))
            print("Note that these classes ", to_be_combined[:-1])
            print("have been combined into ", "['"+new_class_name+"']")
            print("-"*100)
            
    return final_file_paths, final_labels, class_names


def create_dataset(filenames, labels,   batch_size, model_name, patch_size_ratio=1, is_training=True):
     
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (batch_size, N_LABELS)
        is_training: boolean to indicate training mode
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    SHUFFLE_BUFFER_SIZE = 1024
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    
    if is_training == True:
        # dataset = dataset.cache()
        dataset = dataset.map(lambda filename, label: parse_function(filename, label, model_name,patch_size_ratio), num_parallel_calls=AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    else:
        dataset = dataset.map(lambda filename, label: parse_function_aug(filename, label, model_name,patch_size_ratio), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

def remove_outliers (data_dir, class_name, model_name, batch_size,verbose=False):
    classes = os.listdir(data_dir)
    class_index = classes.index(class_name)
    class_path = os.path.join(data_dir, class_name)
    patches_path = glob.glob(os.path.join(class_path, "*"))
    patches_ds = create_dataset(patches_path, np.zeros((len(patches_path),)),batch_size=batch_size,  model_name= model_name, is_training=False )
    feature_extractor = get_feature_extractor (model_name=model_name)
    features = []
    if verbose:
        print("Getting features from "+class_name+ " image patches for outlier detection")
        for f, l in tqdm(patches_ds.take(batch_size)):
            features.append(feature_extractor.predict(f, verbose = 0))
    else:
        for f, l in patches_ds.take(batch_size):
            features.append(feature_extractor.predict(f, verbose = 0))
    new = [features[index].reshape(features[index].shape[0], -1) for index in range (len(features))]
    new_features=  np.vstack(new)
    if_clf = IsolationForest(n_estimators=100,contamination=float(0.2),max_samples='auto')
    if_clf.fit(new_features)
    if_preds = if_clf.predict(new_features)
    patches_path_inliers = []
    patches_path_outliers = []
    for index , c in enumerate(if_preds):
        if c==1:
            patches_path_inliers.append(patches_path[index])
        if c==-1:
            patches_path_outliers.append(patches_path[index])
    return patches_path_inliers, patches_path_outliers
    
#################################################   MulticlassClassifier   ###########################################################################
class MulticlassClassifier():
    def __init__(self,data_dir,patch_size_ratio, remove_classes=[],to_be_combined=[],outlier_classes=[],sample_ratios={},
                 batch_size = 128, model_name = "inception",epochs = 100, initial_learning_rate = 1e-3,
                 decay_steps = 25, decay_rate = 0.9, lr_schedule_flag=False, full_training_flag = False,
                 dense_number = 1024,dropout_ratio = 0.2, loss_function = "categorical_crossentropy", last_activation= "softmax",
                 patience = 20, model_save_path = "multiclass_saved_models",
                 verbose=False):
        self.patch_size_ratio = patch_size_ratio
        self.train_data_dir = data_dir+"/train"
        self.valid_data_dir = data_dir+"/valid"
        self.test_data_dir = data_dir+"/test"
        self.remove_classes = remove_classes
        self.to_be_combined = to_be_combined
        self.outlier_classes = outlier_classes
        self.sample_ratios = sample_ratios
        self.batch_size = batch_size
        self.model_name = model_name
        self.verbose = verbose
        self.epochs = epochs
        self.initial_learning_rate = initial_learning_rate
        self.lr_schedule_flag = lr_schedule_flag
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.lr_schedule = ExponentialDecay(self.initial_learning_rate, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=True)
        self.full_training_flag=full_training_flag
        self.dense_number = dense_number
        self.dropout_ratio = dropout_ratio
        self.loss_function = loss_function
        self.last_activation = last_activation
        self.image_height,self.image_width = self.get_image_size()
        self.patience = patience
        if self.lr_schedule_flag:
            self.callbacks=[EarlyStopping(patience=self.patience, verbose=1, restore_best_weights=True, monitor='val_loss')]
        else:
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
            self.callbacks=[EarlyStopping(patience=self.patience, verbose=1, restore_best_weights=True, monitor='val_loss'),lr_scheduler]
        
        self.train_paths, self.train_labels, self.train_class_names = get_paths_labels_from_dir(data_dir = self.train_data_dir,
                                                                                        remove_classes=self.remove_classes,
                                                                                        to_be_combined=self.to_be_combined,outlier_classes=self.outlier_classes,
                                                                                        sample_ratios=self.sample_ratios,verbose=self.verbose)
    
        self.valid_paths, self.valid_labels, self.valid_class_names = get_paths_labels_from_dir(data_dir = self.valid_data_dir,
                                                                                        remove_classes=self.remove_classes,
                                                                                        to_be_combined=self.to_be_combined,outlier_classes=self.outlier_classes,
                                                                                        sample_ratios=self.sample_ratios,verbose=self.verbose)
    
        self.test_paths, self.test_labels, self.test_class_names = get_paths_labels_from_dir(data_dir = self.test_data_dir,
                                                                                        remove_classes=self.remove_classes,
                                                                                        to_be_combined=self.to_be_combined,outlier_classes=self.outlier_classes,
                                                                                        sample_ratios=self.sample_ratios,verbose=self.verbose)
        
        self.train_ds, self.train_labels_ = self.get_dataset_multi(data_paths = self.train_paths, data_labels = self.train_labels, is_training = True)
        self.valid_ds, self.valid_labels_ = self.get_dataset_multi(data_paths = self.valid_paths, data_labels = self.valid_labels, is_training = False)
        self.test_ds, self.test_labels_ = self.get_dataset_multi(data_paths = self.test_paths, data_labels = self.test_labels, is_training = False)

        self.class_weight_dict = self.get_class_weights ()
        self.model = self.get_model()
        self.model_save_path = model_save_path


    def get_dataset_multi(self, data_paths, data_labels, is_training):
        file_paths = data_paths
        labels_ = to_categorical(data_labels)
        
        if is_training:
            combined = list(zip(file_paths, labels_))
            random.seed(42)
            random.shuffle(combined)
            file_paths, labels_ = zip(*combined)
            file_paths = list(file_paths)
            labels_ = list(labels_)
                
        labels = np.asarray(labels_)
        dataset = create_dataset(file_paths, labels, self.batch_size, self.model_name,self.patch_size_ratio, is_training=False)
        return dataset,labels

    def get_class_weights (self):
        train_class_indices = np.argmax(self.train_labels_, axis=1)
        class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_class_indices), y=train_class_indices)
        class_weight_dict = dict(enumerate(class_weights))
        n_labels = len(class_weight_dict)
        return class_weight_dict

    def print_time(self, t):
        h = t//3600
        m = (t%3600)//60
        s = (t%3600)%60
        return '%dh:%dm:%ds'%(h,m,s)

    def get_image_size(self):
        if self.model_name == "inception" or self.model_name == "xception" or self.model_name == "inception_resnet":
            return 299,299
        else:
            return 224,224
        
    
    def get_model(self):
        metrics = ["acc",
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tfa.metrics.F1Score(num_classes=len(self.train_class_names) , average='weighted')]
        
        base_model = get_feature_extractor (self.model_name, include_top=False, weights='imagenet')
        base_model.trainable = self.full_training_flag

        # model = tf.keras.Sequential([
        #                             base_model,  # Backbone
        #                             layers.GlobalAveragePooling2D(),  # Reduce (batch_size, 8, 8, 2048) to (1, 2048)
        #                             layers.Reshape((1, -1)),  # Reshape to (batch_size, 1, 2048) for attention
        #                             MultiHeadSelfAttention(num_heads=4, key_dim=128),
        #                             layers.Flatten(),
        #                             layers.Dense(self.dense_number, activation='relu', name='hidden_layer_1'),
        #                             layers.Dropout(self.dropout_ratio),
        #                             layers.Dense(len(self.train_class_names), activation=self.last_activation, name='output')], name="multiclass_classifier")
        if self.dense_number!=0:
            model = tf.keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(self.dense_number, activation='relu', name='hidden_layer_1'),
                layers.Dropout(self.dropout_ratio),
                layers.Dense(len(self.train_class_names), activation=self.last_activation, name='output')], name="multiclass_classifier")
        else:
            model = tf.keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_ratio),
                layers.Dense(len(self.train_class_names), activation=self.last_activation, name='output')], name="multiclass_classifier")
        if self.lr_schedule_flag:
            model.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=self.lr_schedule),
                loss=self.loss_function,
                metrics=metrics)
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=self.initial_learning_rate),
                loss=self.loss_function,
                metrics=metrics)
        model.build(input_shape=(None, self.image_height, self.image_width, 3))
        model.summary()
        return model



    def get_evaluations(self):
        y_pred = self.model.predict(self.valid_ds)
        y_hat = np.zeros_like(y_pred)
        y_hat[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1
        y_true = self.valid_labels_
        
        
        classes = list(self.train_class_names.keys())
        loss = self.history['loss']
        val_loss =self.history['val_loss']
        precision = self.history['precision']
        val_precision = self.history['val_precision']
        recall = self.history['recall']
        val_recall = self.history['val_recall']
        acc = self.history['acc']
        val_acc = self.history['val_acc']
        f1 = self.history['f1_score']
        val_f1 = self.history['val_f1_score']
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
        evaluation_path = os.path.join(self.export_path,"evaluation")
        if not os.path.exists(evaluation_path):
            os.mkdir(evaluation_path)
        plt.savefig(os.path.join(evaluation_path,"train_val_scores.png"), bbox_inches='tight')
        plt.close()
        
        report = classification_report(y_true, y_hat, zero_division=0, target_names = classes, output_dict=True)
        report_data = {cls: {'precision': report[cls]['precision'], 
                             'recall': report[cls]['recall'], 
                             'f1-score': report[cls]['f1-score']} for cls in classes}
        
        # Convert report_data to DataFrame
        report_df = pd.DataFrame(report_data).transpose()
        # Plotting the classification report as a heatmap
        plt.figure(figsize=(8, 5))
        sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f', cbar=False)
        plt.title('Classification Report')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
        plt.savefig(os.path.join(evaluation_path,"classification_report.png"), bbox_inches='tight')
        plt.close()
        
        cm = confusion_matrix(y_true.argmax(axis=1), y_hat.argmax(axis=1))
        cm_df = pd.DataFrame(cm, index = classes, columns = classes)
        
        plt.figure(figsize=(20,15))
        ax =sns.heatmap(cm_df, annot = True, fmt='d', cmap='Blues',annot_kws={"size": 16},cbar=False)
        plt.title('Confusion Matrix', fontsize=30)
        plt.ylabel('Actual Values', fontsize=30)
        plt.xlabel('Predicted Values', fontsize=30)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
        plt.savefig(os.path.join(evaluation_path,"Confusion_Matrix.png"), bbox_inches='tight')
        plt.close()
        
        row_sums = cm_df.sum(axis=1)
        total_samples = row_sums.sum()
        plt.figure(figsize=(10, 6))
        bars = plt.bar(row_sums.index, row_sums.values, color='skyblue')
        plt.bar(row_sums.index, row_sums.values, color='skyblue')
        plt.xlabel('Classes')
        plt.ylabel('Number of samples')
        plt.title('Number of samples per class in the valid test')
        plt.xticks(rotation=90)
        for bar, value in zip(bars, row_sums.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(evaluation_path,"val_statistics.png"), bbox_inches='tight')
        plt.close()
    

    def fit(self):
        def convert_to_serializable(obj):
            if isinstance(obj, tf.Tensor):
                return obj.numpy().tolist()  
            elif isinstance(obj, np.ndarray):
                return obj.tolist()  
            return obj
        
        start = time()
        history = self.model.fit(self.train_ds, epochs=self.epochs, validation_data=self.valid_ds, class_weight = self.class_weight_dict,callbacks=self.callbacks)
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        print('\nTraining took {}'.format(self.print_time(time()-start)))
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = self.model_save_path+"/{}".format(t)
        self.export_path = export_path
        self.history = {key: np.array(value).tolist() for key, value in history.history.items()}
        try:
            self.model.save(os.path.join(export_path,self.model_name+".h5"))
        except:
            try:
                self.model.save_weights(os.path.join(export_path,self.model_name+".h5"))
            except:
                try:
                    self.model.save(export_path)
                except:
                    print(self.model_name,": could not be saved")
        
        specs = {"patch_size_ratio":self.patch_size_ratio,"remove_classes": self.remove_classes, "to_be_combined": self.to_be_combined,
                "outlier_classes": self.outlier_classes, "sample_ratios": self.sample_ratios, "batch_size": self.batch_size, "model_name":self.model_name, "epochs":self.epochs,
                "initial_learning_rate":self.initial_learning_rate, "decay_steps":self.decay_steps, "decay_rate":self.decay_rate,
                "lr_schedule_flag":self.lr_schedule_flag, "full_training_flag":self.full_training_flag, "dense_number":self.dense_number,
                "dropout_ratio": self.dropout_ratio, "number_of_classes":len(self.train_class_names), "loss_function":self.loss_function,
                "last_activation":self.last_activation,"patience":self.patience, "class_names": self.train_class_names, "weights": self.class_weight_dict}
        
        specs = {k:convert_to_serializable(v) for k,v in specs.items()}
        json.dump(self.history, open(export_path+"/history", 'w'))
        json.dump(specs, open(export_path+"/specs", 'w'))
        self.get_evaluations()
