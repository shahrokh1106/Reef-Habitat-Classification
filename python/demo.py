import os
import glob
import numpy as np
import random
import cv2
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm 
from get_yolo_model import GetYoloModel
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

SEED = 2
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class Habibot():
    def __init__(self,frame_classifier_path,patch_classifier_path,crop_ratio):
        self.frame_classifier_path = frame_classifier_path
        self.patch_classifier_path = patch_classifier_path
        self.frame_classifier = self.load_classifier(self.frame_classifier_path)
        self.frame_input_size = self.frame_classifier.input.shape[1]
        self.patch_classifier = self.load_classifier(self.patch_classifier_path)
        self.patch_input_size = self.patch_classifier.input.shape[1]
        self.urch_detector = self.load_urchin_detector()
        self.i_to_c_frame = {0: "Reef-Urchin-Barren", 1: "Reef-BrLfa",2: "Reef-Kelp", 3: "Reef-FnEc",4: "Unconsolidated"}
        self.c_to_i_frame = {value:key for key,value in self.i_to_c_frame.items()}
        self.i_to_c_patch = {0: 'Carpophyllum',1: 'Ecklonia',2: 'Foliose algae',3: 'Other canopy',4: 'Unconsolidated',5: 'Urchin',6: 'BrLfa'}
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
       
        patches_pred_ratios_dict = {'Carpophyllum': 0.0, 'Ecklonia': 0.0, 'Foliose algae': 0.0,'Other canopy': 0.0, 'Unconsolidated': 0.0, 'BrLfa': 0.0, 'Unscorable': 0.0}
        for i, e in enumerate(unique_elements):
            patches_pred_ratios_dict[self.i_to_c_patch[e]] = percentages[i]
   
        
        total_valid_patches = len(patches_preds_categorial)  # Original number of patches
        patches_pred_ratios_dict['Unscorable'] = (total_unscorable / total_valid_patches) * 100 if total_valid_patches > 0 else 0.

        
        return img,patches,boxes,centers, patches_pred_ratios_dict, patches_preds_categorial,patches_probs


    def get_all_models_predictions(self,image):
        img,patches,patches_boxes,patches_centers, patches_pred_ratios_dict, patches_preds_categorial,patches_probs = self.get_patch_level_prediction_ratios(image)
        urchin_boxes = self.get_urchin_boxes(img)
        frame_probs = self.get_frame_level_prediction_probs(img)
        return img,patches_boxes,patches_centers,patches_pred_ratios_dict,patches_preds_categorial,patches_probs,frame_probs,urchin_boxes

    def get_recommendation(self,image_path,output_name):
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (3648, 2736))
        i_to_c_frame = {0: "Reef-Urchin-Barren", 1: "Reef-Grazed",2: "Reef-Kelp", 3: "Reef-Vegetated",4: "Unconsolidated"}
        i_to_c_frame = {0: "Reef-Urchin-Barren", 1: "Reef-BrLfa",2: "Reef-Kelp", 3: "Reef-FnEc",4: "Unconsolidated"} # renaming to the updated class names
        patch_conf_threshold = 0.7
        img,patches_boxes,patches_centers,patches_pred_ratios_dict,patches_preds_categorial,patches_probs,frame_probs,urchin_boxes = self.get_all_models_predictions(frame)

        patch_classes = [f'Carpophyllum ({np.round(patches_pred_ratios_dict["Carpophyllum"],1)}%)',
                        f'Ecklonia ({np.round(patches_pred_ratios_dict["Ecklonia"],1)}%)',
                        f'Foliose algae ({np.round(patches_pred_ratios_dict["Foliose algae"],1)}%)',
                        f'Other canopy ({np.round(patches_pred_ratios_dict["Other canopy"],1)}%)',
                        f'Unconsolidated ({np.round(patches_pred_ratios_dict["Unconsolidated"],1)}%)',
                        f'Urchin',
                        f'BrLfa ({np.round(patches_pred_ratios_dict["BrLfa"],1)}%)',
                        ]
        initial_frame_level_prediction = i_to_c_frame[np.argmax(frame_probs)]
        cmap = matplotlib.colormaps.get_cmap('tab10')
        # cmap = get_cmap('tab10') 
        colors = [cmap(i) for i in range(len(patch_classes))]
        colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in [cmap(i) for i in range(len(patch_classes))]]
        colors = colors[::-1]
        patch_classes.append(f'Unscorable ({np.round(patches_pred_ratios_dict["Unscorable"],1)}%)')
        colors.append((255,255,255))

        for i in range(len(patches_boxes)):
            probs = patches_probs[i]
            c_color =  colors[patches_preds_categorial[i]]
            if np.max(probs)< patch_conf_threshold:
                c_color = colors[7]
            cv2.circle(img, (patches_centers[i]), 22,c_color,-1)
            
        for i in range(len(patches_boxes)):
            cv2.rectangle(img, (patches_boxes[i][0], patches_boxes[i][1]), (patches_boxes[i][2], patches_boxes[i][3]), (255,255,255),2)
        
        for i in range(len(urchin_boxes)):
            x1 = int(urchin_boxes[i][0])-(int(urchin_boxes[i][2]))//2
            y1 = int(urchin_boxes[i][1]) - (int(urchin_boxes[i][3]))//2
            x2 = x1+int(urchin_boxes[i][2])
            y2 = y1+ int(urchin_boxes[i][3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0),5)
        rule_numbers = []

        recommendation = "None"
        # Initial label conversion
        if initial_frame_level_prediction == "Reef-BrLfa":
            if len(urchin_boxes)>0:
                initial_frame_level_prediction = "Reef-Urchin-Barren"
                rule_numbers.append("#1")
        elif initial_frame_level_prediction == "Reef-Urchin-Barren":
            if len(urchin_boxes)==0:
                initial_frame_level_prediction = "Reef-BrLfa"
                rule_numbers.append("#2")

        # rules-based label conversion
        if np.max(frame_probs)<=0.6:
            if len(urchin_boxes)==0:
                recommendation = "Reef-Partial-BrLfa"
                rule_numbers.append("#3")
            else:
                recommendation = "Reef-Partial-Urchin-Barren"
                rule_numbers.append("#4")
        elif initial_frame_level_prediction == "Reef-Kelp":
            if len(urchin_boxes)>0: 
                recommendation = "Reef-Partial-Urchin-Barren"
                rule_numbers.append("#5")
            elif (patches_pred_ratios_dict['BrLfa']>25 and len(urchin_boxes)==0):
                recommendation = "Reef-Partial-BrLfa"
                rule_numbers.append("#6")
        elif initial_frame_level_prediction == "Reef-FnEc":
            if len(urchin_boxes)>0:  
                recommendation = "Reef-Partial-Urchin-Barren"
                rule_numbers.append("#7")
            elif (patches_pred_ratios_dict['BrLfa']>25 and len(urchin_boxes)==0):
                    recommendation = "Reef-Partial-BrLfa"
                    rule_numbers.append("#8")
        elif initial_frame_level_prediction == "Reef-Urchin-Barren":
            if patches_pred_ratios_dict['Carpophyllum']+patches_pred_ratios_dict['Ecklonia']+patches_pred_ratios_dict['Other canopy']+patches_pred_ratios_dict['Foliose algae']>25:
                recommendation = "Reef-Partial-Urchin-Barren"
                rule_numbers.append("#9")
        elif initial_frame_level_prediction == "Reef-BrLfa":
            if patches_pred_ratios_dict['Carpophyllum']+patches_pred_ratios_dict['Ecklonia']+patches_pred_ratios_dict['Other canopy']+patches_pred_ratios_dict['Foliose algae']>25:
                recommendation = "Reef-Partial-BrLfa"
                rule_numbers.append("#10")
        elif  initial_frame_level_prediction == "Unconsolidated":
            if len(urchin_boxes)>0:
                recommendation = "Reef-Partial-Urchin-Barren"
                rule_numbers.append("#11")

        if recommendation=="None":
            recommendation = initial_frame_level_prediction
            rule_numbers.append("#12")

        fig, ax = plt.subplots(figsize=(15, 13)) 
        fig.patch.set_facecolor("black")  # Set figure background color
        ax.set_facecolor("black")  # Set axes background color
        plt.imshow(img)
        plt.axis("off")  
        legend_patches = [mpatches.Patch(color=(colors[i][0]/255,colors[i][1]/255,colors[i][2]/255), label=class_name) for i, class_name in enumerate(patch_classes)]
        legend =plt.legend(
                handles=legend_patches,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),  
                ncol=3,  #
                fontsize=22,
                frameon=False,  # Removes legend box frame
                handleheight=1,  # Increase height of legend patches
                handlelength=1,  # Increase width of legend patches
                labelspacing=0.2  # Increase space between text and legend boxes
        )
        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_edgecolor("white")  # Optional: White border for better visibility
        for text in legend.get_texts():
            text.set_color("white")  # Set legend text color to white for contrast
            text.set_fontstyle('italic')

        a = -0.20
        plt.text(
            0.5, a,  
            f"urchin-count by Urchinbot: {len(urchin_boxes)}",
            color="white", fontsize=22, ha="center", transform=ax.transAxes, fontstyle="italic"
        )
        plt.text(
            0.5, a-0.05,  
            f"initial-frame-prediction: {initial_frame_level_prediction} / conf-frame: {np.max(frame_probs):.2f}",
            color="white", fontsize=22, ha="center", transform=ax.transAxes, fontstyle="italic"
        )
        plt.text(
            0.5, a-0.1,  
            f"Recommendation: {recommendation}",
            color="white", fontsize=23, ha="center", transform=ax.transAxes, fontstyle="italic"
        )
        plt.text(
            0.5, a-0.15, 
            f"used rules: {rule_numbers}",
            color="white", fontsize=22, ha="center", transform=ax.transAxes, fontstyle="italic"
        )
        plt.tight_layout()  
        plt.savefig(output_name+".png", bbox_inches='tight')
        plt.close()
        return recommendation


if __name__ == '__main__':
    frame_classifier_path =  os.path.join("trained_classifiers_", "frame_classifier")
    patch_classifier_path = os.path.join("trained_classifiers_","patch_classifier")
    HClassifier = Habibot(frame_classifier_path =frame_classifier_path,
                                    patch_classifier_path = patch_classifier_path,
                                    crop_ratio= 0.2)
    image_path= os.path.join("fig","test_input.JPG")    
    recommendation = HClassifier.get_recommendation(image_path,output_name = os.path.join("fig","test_output.png")) 
    print("Recommendation: ", recommendation)
