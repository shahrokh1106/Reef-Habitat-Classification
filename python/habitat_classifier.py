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
from tqdm import tqdm 
from get_yolo_model import GetYoloModel
import time

from matplotlib.cm import get_cmap
SEED = 2
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class HabitatClassifier():
    def __init__(self,frame_classifier_path,patch_classifier_path,crop_ratio):
        self.frame_classifier_path = frame_classifier_path
        self.patch_classifier_path = patch_classifier_path
        self.frame_classifier = self.load_classifier(self.frame_classifier_path)
        self.frame_input_size = self.frame_classifier.input.shape[1]
        self.patch_classifier = self.load_classifier(self.patch_classifier_path)
        self.patch_input_size = self.patch_classifier.input.shape[1]
        self.urch_detector = self.load_urchin_detector()
        self.i_to_c_frame = {0: "Reef-Urchin-Barren", 1: "Reef-Grazed",2: "Reef-Kelp", 3: "Reef-Vegetated",4: "Unconsolidated"}
        self.c_to_i_frame = {value:key for key,value in self.i_to_c_frame.items()}
        self.i_to_c_patch = {0: 'Carpophyllum',1: 'Ecklonia',2: 'Foliose Algae',3: 'Other canopy',4: 'Unconsolidated',5: 'Urchin',6: 'Grazed rock'}
        self.c_to_i_patch = {value:key for key,value in self.i_to_c_patch.items()}
        self.crop_ratio = crop_ratio
        self.patch_conf_threshold = 0.7

    def load_classifier(self,path):
        model = tf.keras.models.load_model(path)
        return model
    
    def load_urchin_detector(self):
        model = GetYoloModel()
        return model
    
    def load_prepare_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_urchin_boxes(self,image):
        
        boxes = self.urch_detector(image)
        return boxes
    
    def get_frame_level_prediction_probs(self, image):
        image = tf.convert_to_tensor(image)
        image = tf.image.resize(image, (self.frame_input_size, self.frame_input_size))
        image = tf.expand_dims(image, axis=0)
        image = preprocess_input_inception(image)
        frame_probs = self.frame_classifier.predict(image,verbose = "0")
        return frame_probs
    
    def get_patch_level_prediction_ratios(self,image):
        def extract_grid_patches(image: np.ndarray):
            crop_size = int(((image.shape[1]+image.shape[0])/2)*self.crop_ratio)//2
            h, w, c = image.shape
            new_h = (h // crop_size) * crop_size
            new_w = (w // crop_size) * crop_size
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            patch_size = crop_size
            patches = []
            centers = []
            boxes = []
            for i in range(0, new_h, patch_size):
                for j in range(0, new_w, patch_size):
                    patch = image[i:i+patch_size, j:j+patch_size]
                    patches.append(patch)
                    x_min, y_min = j, i
                    x_max, y_max = min(j + patch_size, w), min(i + patch_size, h)
                    boxes.append((x_min, y_min, x_max, y_max))
                    x_center = (x_min + x_max) // 2
                    y_center = (y_min + y_max) // 2
                    centers.append((x_center, y_center))
            return image,patches,boxes,centers
        
        def preprocess_patches (patches,model_image_size):
            patches = tf.convert_to_tensor(patches)
            patches = tf.image.resize(patches, [model_image_size, model_image_size])
            patches = preprocess_input_inception(patches)  
            return patches
       
        image = self.load_prepare_image(image)
        img,patches,boxes,centers = extract_grid_patches(image)
        patches_array = np.stack(patches, axis=0)
        patches_tensor = preprocess_patches (patches_array,self.patch_input_size)
        patches_probs = self.patch_classifier.predict(patches_tensor,batch_size=len(patches),verbose=0)
        patches_preds_binary = np.zeros_like(patches_probs)
        patches_preds_binary[np.arange(len(patches_probs)), np.argmax(patches_probs, axis=1)] = 1
        patches_preds_categorial = np.argmax(patches_preds_binary == 1, axis=1)
        unique_elements, counts = np.unique(patches_preds_categorial, return_counts=True)


        # total_urchin = 0
        # if 5 in unique_elements:
        #     # index 5 is for urchin, should not be considered for the percentage computation
        #     index = np.where(unique_elements == 5)[0][0]
        #     total_urchin = counts[index]
        #     unique_elements = np.delete(unique_elements, index)
        #     counts = np.delete(counts, index)
        # total_count = len(patches_preds_categorial)
        # percentages = (counts / total_count) * 100    
        # patches_pred_ratios_dict = {'Carpophyllum': 0.0, 'Ecklonia': 0.0, 'Foliose Algae': 0.0, 'Other canopy': 0.0, 'Unconsolidated': 0.0, 'Grazed rock': 0.0}
        # for i, e in enumerate(unique_elements):
        #     patches_pred_ratios_dict[self.i_to_c_patch[e]] = percentages[i]

        total_urchin = 0
        low_confidence_mask = np.max(patches_probs, axis=1) < self.patch_conf_threshold
        total_unscorable = np.sum(low_confidence_mask)
        
        filtered_patches_preds_categorial = patches_preds_categorial[~low_confidence_mask]
        unique_elements, counts = np.unique(filtered_patches_preds_categorial, return_counts=True)
    
        if 5 in unique_elements:
            # index 5 is for urchin, should not be considered for the percentage computation
            index = np.where(unique_elements == 5)[0][0]
            total_urchin = counts[index]
            unique_elements = np.delete(unique_elements, index)
            counts = np.delete(counts, index)

        total_count = len(filtered_patches_preds_categorial) - total_urchin
    
        percentages = (counts / total_count) * 100 if total_count > 0 else np.zeros_like(counts)
       
        patches_pred_ratios_dict = {'Carpophyllum': 0.0, 'Ecklonia': 0.0, 'Foliose Algae': 0.0,'Other canopy': 0.0, 'Unconsolidated': 0.0, 'Grazed rock': 0.0, 'Unscorable': 0.0}
        for i, e in enumerate(unique_elements):
            patches_pred_ratios_dict[self.i_to_c_patch[e]] = percentages[i]
  
        total_valid_patches = len(patches_preds_categorial)  # Original number of patches
        patches_pred_ratios_dict['Unscorable'] = (total_unscorable / total_valid_patches) * 100 if total_valid_patches > 0 else 0.


        
        return img,patches,boxes,centers, patches_pred_ratios_dict, patches_preds_categorial,patches_probs

    def draw_results(self,image,patches_boxes,patches_centers,patches_pred_ratios_dict,patches_preds_categorial,patches_probs,frame_probs,urchin_boxes):
        import matplotlib
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        initial_frame_level_prediction = self.i_to_c_frame[np.argmax(frame_probs)]
        cmap = matplotlib.colormaps.get_cmap('tab10')
        # cmap = get_cmap('tab10') 
        colors = [cmap(i) for i in range(len(self.i_to_c_patch))]
        colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in [cmap(i) for i in range(len(self.i_to_c_patch))]]
        colors.append((255,255,255))
        for i in range(len(patches_boxes)):
            probs = patches_probs[i]
            c_color =  colors[patches_preds_categorial[i]]
            if np.max(probs)< self.patch_conf_threshold:
                c_color = colors[7]
            cv2.circle(image, (patches_centers[i]), 20,c_color,-1)
        
        for i in range(len(patches_boxes)):
            cv2.rectangle(image, (patches_boxes[i][0], patches_boxes[i][1]), (patches_boxes[i][2], patches_boxes[i][3]), (255,255,255),1)
        
        for i in range(len(urchin_boxes)):
            x1 = int(urchin_boxes[i][0])-(int(urchin_boxes[i][2]))//2
            y1 = int(urchin_boxes[i][1]) - (int(urchin_boxes[i][3]))//2
            x2 = x1+int(urchin_boxes[i][2])
            y2 = y1+ int(urchin_boxes[i][3])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,0),5)

        padding_height = 200 
        legend_background = np.zeros((padding_height, image.shape[1], 3), dtype=np.uint8) 
        rect_size = 50  # Size of legend color boxes
        text_offset = 5  # Adjust text position
        x_offset = 50  # Starting X position for legend
        spacing = 450  # Space between legend items (adjust as needed)
        for i, class_name in self.i_to_c_patch.items():
            color = colors[i]
            cv2.rectangle(legend_background, (x_offset + i * spacing, 10), (x_offset + i * spacing + rect_size, 10 + rect_size), color, -1)
            if class_name !="Urchin":
                cv2.putText(legend_background, class_name+f" ({np.round(patches_pred_ratios_dict[class_name],1)}%)", (x_offset + i * spacing + rect_size + 10, 10 + rect_size - text_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(legend_background, class_name+" (0%)", (x_offset + i * spacing + rect_size + 10, 10 + rect_size - text_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if i==6:
                i+=1
                color = colors[i]
                cv2.rectangle(legend_background, (x_offset + i * spacing, 10), (x_offset + i * spacing + rect_size, 10 + rect_size), color, -1)
                cv2.putText(legend_background, "Unscorable"+f" ({np.round(patches_pred_ratios_dict['Unscorable'],1)}%)", (x_offset + i * spacing + rect_size + 10, 10 + rect_size - text_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(legend_background, f"Initial Frame Prediction: {initial_frame_level_prediction}", (50,x_offset +100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            
        recommendation = "None"
        # Initial label conversion
        if initial_frame_level_prediction == "Reef-Grazed":
            if len(urchin_boxes)>0:
                initial_frame_level_prediction = "Reef-Urchin-Barren"
        elif initial_frame_level_prediction == "Reef-Urchin-Barren":
            if len(urchin_boxes)==0:
                initial_frame_level_prediction = "Reef-Grazed"

        # rules-based label conversion
        if np.max(frame_probs)<=0.6:
            if len(urchin_boxes)==0:
                recommendation = "Reef-Partial-Grazed (Review)"
            else:
                recommendation = "Reef-Partial-Urchin-Barren (Review)"
        elif initial_frame_level_prediction == "Reef-Kelp":
            if patches_pred_ratios_dict['Grazed rock']>25: # OR/AND Urchin present (both OR/AND does not matter so I did not include this condition)
                recommendation = "Reef-Partial-Urchin-Barren (Review)"
        elif initial_frame_level_prediction == "Reef-Vegetated":
            if patches_pred_ratios_dict['Grazed rock']>25:  # OR/AND Urchin present (both OR/AND does not matter so I did not include this condition)
                recommendation = "Reef-Partial-Urchin-Barren (Review)"
        elif initial_frame_level_prediction == "Reef-Urchin-Barren":
            if patches_pred_ratios_dict['Carpophyllum']+patches_pred_ratios_dict['Ecklonia']+patches_pred_ratios_dict['Other canopy']+patches_pred_ratios_dict['Foliose Algae']>25:
                recommendation = "Reef-Partial-Urchin-Barren (Review)"
        elif initial_frame_level_prediction == "Reef-Grazed":
            if patches_pred_ratios_dict['Carpophyllum']+patches_pred_ratios_dict['Ecklonia']+patches_pred_ratios_dict['Other canopy']+patches_pred_ratios_dict['Foliose Algae']>25:
                recommendation = "Reef-Partial-Grazed (Review)"

        if recommendation=="None":
            recommendation = initial_frame_level_prediction
            
        cv2.putText(legend_background, f"Recommendation: {recommendation}", (1700,x_offset +100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        image = np.vstack((image, legend_background))

        return image


    def get_all_models_predictions(self,image):
        img,patches,patches_boxes,patches_centers, patches_pred_ratios_dict, patches_preds_categorial,patches_probs = self.get_patch_level_prediction_ratios(image)
        urchin_boxes = self.get_urchin_boxes(img)
        frame_probs = self.get_frame_level_prediction_probs(img)
        return img,patches_boxes,patches_centers,patches_pred_ratios_dict,patches_preds_categorial,patches_probs,frame_probs,urchin_boxes
        
    


        


        
        

if __name__ == '__main__':

    HClassifier = HabitatClassifier(frame_classifier_path = os.path.join("trained_classifiers", "frame_classifier"),
                                    patch_classifier_path = os.path.join("trained_classifiers","patch_classifier"),
                                    crop_ratio= 0.2)
    # image_path = os.path.join("dataset","frame7_dataset_cleaned","train", "reef_barren", "5423297_NSW49WCB5MAragunnuSouthernBay071114 (17).JPG")
    start_time = time.time()
    cap = cv2.VideoCapture("33.MP4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_id = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
    out = cv2.VideoWriter("outtt.mp4", fourcc, fps, (width, height),  isColor=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break  
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img,patches_boxes,patches_centers,patches_pred_ratios_dict,patches_preds_categorial,patches_probs,frame_probs,urchin_boxes = HClassifier.get_all_models_predictions(frame)
        processed_frame = HClassifier.draw_results(img,patches_boxes,patches_centers,patches_pred_ratios_dict,patches_preds_categorial,patches_probs,frame_probs,urchin_boxes)
        processed_frame = cv2.resize(processed_frame, (width,height))
        out.write(processed_frame)
    
    out.release()
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
    print("All GOOD")