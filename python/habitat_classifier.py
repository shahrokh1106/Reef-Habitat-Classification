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
import folium
from tqdm import tqdm
from matplotlib.cm import get_cmap
from branca.element import Template, MacroElement
from folium.plugins import HeatMap
from collections import defaultdict

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
                if patches_pred_ratios_dict['Grazed rock']>70:
                    recommendation = "Reef-Urchin-Barren (Review)"
                else:
                    recommendation = "Reef-Partial-Urchin-Barren (Review)"
        elif initial_frame_level_prediction == "Reef-Kelp":
            if (patches_pred_ratios_dict['Grazed rock']>25 and len(urchin_boxes)>0) or len(urchin_boxes)>0: 
                recommendation = "Reef-Partial-Urchin-Barren (Review)"
        elif initial_frame_level_prediction == "Reef-Vegetated":
            if (patches_pred_ratios_dict['Grazed rock']>25 and len(urchin_boxes)>0) or len(urchin_boxes)>0:  
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

        return image,recommendation

    def get_all_models_predictions(self,image):
        img,patches,patches_boxes,patches_centers, patches_pred_ratios_dict, patches_preds_categorial,patches_probs = self.get_patch_level_prediction_ratios(image)
        urchin_boxes = self.get_urchin_boxes(img)
        frame_probs = self.get_frame_level_prediction_probs(img)
        return img,patches_boxes,patches_centers,patches_pred_ratios_dict,patches_preds_categorial,patches_probs,frame_probs,urchin_boxes
        
def get_density_dicts(samples_path,output_path,frame_classifier_path,patch_classifier_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    def get_location_map(locations_dfs,output_file=os.path.join(output_path,"location_map.html")):
        colors = itertools.cycle(["blue", "green", "red", "purple", "orange", "darkred", "lightred", 
                               "beige", "darkblue", "darkgreen", "cadetblue", "pink", "gray"])
        map_center = [0, 0] 
        first_region = next(iter(locations_dfs.values()))
        region_colors = {}
        if not first_region.empty:
            map_center = [first_region.iloc[0]['Y'], first_region.iloc[0]['X']]
        m = folium.Map(location=map_center, zoom_start=5)
        for region_name, df in locations_dfs.items():
            color = next(colors)
            region_colors[region_name] = color
            for _, row in df.iterrows():
                folium.Marker(
                    location=[row['Y'], row['X']],
                    popup=f"Region: {region_name}<br>Label: {row['#Label']}<br>Z: {row['Z']}",
                    tooltip=row['#Label'],
                    icon=folium.Icon(color=color, icon="info-sign")
                ).add_to(m)
        legend_html = """
        {% macro html(this, kwargs) %}
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: auto; 
                    background-color: white; z-index:9999; padding: 10px;
                    font-size:14px; border-radius:5px;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
            <b>Legend</b><br>
        """
        
        for region, color in region_colors.items():
            legend_html += f"""
            <div style="display: flex; align-items: center; margin-top: 5px;">
                <div style="width: 12px; height: 12px; background:{color}; margin-right: 5px;"></div>
                {region}
            </div>
            """

        legend_html += "</div>{% endmacro %}"

        # Add the legend to the map
        legend = MacroElement()
        legend._template = Template(legend_html)
        m.get_root().add_child(legend)
        m.save(output_file)
        
    files_paths = glob.glob(os.path.join(samples_path, "*"))
    folders_paths = {}
    location_paths = {}
    for file_path in files_paths:
        if file_path.endswith(".csv"):
            location_paths.update({os.path.basename(file_path)[:-4]:file_path})
        else:
            folders_paths.update({os.path.basename(file_path):file_path})
    images_paths = {}
    locations_dfs = {}
    for location_name in folders_paths.keys():
        images_paths.update({location_name: glob.glob(os.path.join(folders_paths[location_name],"*"))})
        csv_path = location_paths[location_name]
        locations_dfs.update({location_name: pd.read_csv(csv_path)})
    get_location_map(locations_dfs)
    final_results = {}
    for location_name in locations_dfs.keys():
        if os.path.exists(os.path.join(output_path,location_name+".json")):
            with open(os.path.join(output_path,location_name+".json"), "r") as f:
                results_dict = json.load(f)
            final_results.update({location_name:results_dict})
            continue
        results_dict = {}
        HClassifier = HabitatClassifier(frame_classifier_path =frame_classifier_path,
                                    patch_classifier_path = patch_classifier_path,
                                    crop_ratio= 0.2)
        print("Getting classifiers' results for "+ location_name)
        img_paths = images_paths[location_name]
        saved_vis_path = os.path.join(output_path,location_name)
        if not os.path.exists(saved_vis_path):
            os.makedirs(saved_vis_path)
        for img_path in tqdm(img_paths):
            frame =cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img,patches_boxes,patches_centers,patches_pred_ratios_dict,patches_preds_categorial,patches_probs,frame_probs,urchin_boxes = HClassifier.get_all_models_predictions(frame)
            processed_frame,recommendation = HClassifier.draw_results(img,patches_boxes,patches_centers,patches_pred_ratios_dict,patches_preds_categorial,patches_probs,frame_probs,urchin_boxes)
            cv2.imwrite(os.path.join(saved_vis_path,os.path.basename(img_path)), processed_frame)
            results_dict.update({os.path.basename(img_path)[:-4]: [recommendation,len(urchin_boxes)]})
            
        with open(os.path.join(output_path,location_name+".json"), "w") as json_file:
            json.dump(results_dict, json_file, indent=4)
        final_results.update({location_name:results_dict})
    
    combined_dict = {}
    for key in final_results.keys():
        data_list = []
        for image_name, details in final_results[key].items():
            data_list.append([image_name, details[0], details[1]])
        data_df = pd.DataFrame(data_list, columns=["image_name", "predictions", "urchin_count"])
        loc_df = locations_dfs[key].rename(columns={"#Label": "image_name"})
        loc_df["image_name"] = loc_df["image_name"].str.slice(stop=-4)
        merged_df = pd.merge(loc_df, data_df, on="image_name", how="inner")
        combined_dict[key] = merged_df
    return combined_dict

def get_density_heat_maps(combined_dict, output_path=".", output_file="heatmap_maps.html"):
    output_file = os.path.join(output_path, output_file)
    all_points = []
    for df in combined_dict.values():
        all_points.extend(df[['Y', 'X']].values.tolist())
    map_center = [np.mean([p[0] for p in all_points]), np.mean([p[1] for p in all_points])]
    m = folium.Map(location=map_center, zoom_start=7)
    heatmap_colors = itertools.cycle(["red", "blue", "green", "purple", "orange", "pink", "brown"])
    heatmap_colors = {"Reef-Urchin-Barren": "red",
                      "Reef-Partial-Urchin-Barren": "orange",
                      "Unconsolidated": "blue",
                      "Reef-Vegetated": "#99ff99",
                      "Reef-Kelp": "green",
                      "Reef-Grazed": "purple",
                      "Reef-Partial-Grazed":"pink",
                      }

    # Urchin count heatmap (aggregating values for overlapping points)
    urchin_data = []
    for df in combined_dict.values():
        for _, row in df.iterrows():
            urchin_data.append([row['Y'], row['X'], row['urchin_count']])
    if urchin_data:
        max_urchin = max([x[2] for x in urchin_data]) if urchin_data else 1
        urchin_data = [[y, x, count / max_urchin] for y, x, count in urchin_data]
        HeatMap(urchin_data, name="Urchin Count", gradient={0.2: "#ffffff", 0.8: "black"}).add_to(m)

    # Multiclass prediction heatmaps
    class_heatmaps = defaultdict(list)
    for key, df in combined_dict.items():
        for _, row in df.iterrows():
            prediction = row['predictions'] 
            if prediction == "Reef-Urchin-Barren (Review)":
                prediction="Reef-Urchin-Barren"
            if prediction == "Reef-Partial-Urchin-Barren (Review)":
                prediction = "Reef-Partial-Urchin-Barren"
            if prediction == "Reef-Partial-Grazed (Review)":
                prediction = "Reef-Partial-Grazed"
            class_heatmaps[prediction].append([row['Y'], row['X'], 1])  # Binary heatmap (presence)


    # Add heatmaps for each prediction class
    legend_entries = []
    for class_name in heatmap_colors.keys():
        legend_entries.append((class_name, heatmap_colors[class_name]))
    for class_name, data in class_heatmaps.items():
        color = heatmap_colors[class_name]
        HeatMap(data, name=class_name, gradient={0.1: "#ffffff", 0.7: color},blur=25, radius=20).add_to(m)
            
    folium.LayerControl(collapsed=False).add_to(m)
    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px; height: auto; 
                background-color: white; z-index:9999; padding: 10px;
                font-size:14px; border-radius:5px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
        <b>Legend</b><br>
        <div><div style="width: 12px; height: 12px; background: black; display: inline-block;"></div> Urchin Count</div>
    """
    for class_name, color in legend_entries:
        legend_html += f"""
        <div><div style="width: 12px; height: 12px; background: {color}; display: inline-block;"></div> {class_name}</div>
        """
    legend_html += "</div>{% endmacro %}"
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)
    m.save(output_file)
    print(f"Heatmap saved to {output_file}")

def make_fig_example_for_paper(HClassifier,image_path,output_name):
    import matplotlib
    import matplotlib.patches as mpatches
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (3648, 2736))
    i_to_c_frame = {0: "Reef-Urchin-Barren", 1: "Reef-Grazed",2: "Reef-Kelp", 3: "Reef-Vegetated",4: "Unconsolidated"}
    patch_conf_threshold = 0.75
    img,patches_boxes,patches_centers,patches_pred_ratios_dict,patches_preds_categorial,patches_probs,frame_probs,urchin_boxes = HClassifier.get_all_models_predictions(frame)

    patch_classes = [f'Carpophyllum ({np.round(patches_pred_ratios_dict["Carpophyllum"],1)}%)',
                     f'Ecklonia ({np.round(patches_pred_ratios_dict["Ecklonia"],1)}%)',
                     f'Foliose Algae ({np.round(patches_pred_ratios_dict["Foliose Algae"],1)}%)',
                     f'Other canopy ({np.round(patches_pred_ratios_dict["Other canopy"],1)}%)',
                     f'Unconsolidated ({np.round(patches_pred_ratios_dict["Unconsolidated"],1)}%)',
                     f'Urchin',
                     f'Grazed rock ({np.round(patches_pred_ratios_dict["Grazed rock"],1)}%)',
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
    if initial_frame_level_prediction == "Reef-Grazed":
        if len(urchin_boxes)>0:
            initial_frame_level_prediction = "Reef-Urchin-Barren"
            rule_numbers.append("#1")
    elif initial_frame_level_prediction == "Reef-Urchin-Barren":
        if len(urchin_boxes)==0:
            initial_frame_level_prediction = "Reef-Grazed"
            rule_numbers.append("#2")

    # rules-based label conversion
    if np.max(frame_probs)<=0.6:
        if len(urchin_boxes)==0:
            recommendation = "Reef-Partial-Grazed"
            rule_numbers.append("#3")
        else:
            if patches_pred_ratios_dict['Grazed rock']>70:
                recommendation = "Reef-Urchin-Barren"
                rule_numbers.append("#4")
            else:
                recommendation = "Reef-Partial-Urchin-Barren"
                rule_numbers.append("#5")
    elif initial_frame_level_prediction == "Reef-Kelp":
        if (patches_pred_ratios_dict['Grazed rock']>25 and len(urchin_boxes)>0) or len(urchin_boxes)>0: 
            recommendation = "Reef-Partial-Urchin-Barren"
            rule_numbers.append("#6")
    elif initial_frame_level_prediction == "Reef-Vegetated":
        if (patches_pred_ratios_dict['Grazed rock']>25 and len(urchin_boxes)>0) or len(urchin_boxes)>0:  
            recommendation = "Reef-Partial-Urchin-Barren"
            rule_numbers.append("#7")
    elif initial_frame_level_prediction == "Reef-Urchin-Barren":
        if patches_pred_ratios_dict['Carpophyllum']+patches_pred_ratios_dict['Ecklonia']+patches_pred_ratios_dict['Other canopy']+patches_pred_ratios_dict['Foliose Algae']>25:
            recommendation = "Reef-Partial-Urchin-Barren"
            rule_numbers.append("#8")
    elif initial_frame_level_prediction == "Reef-Grazed":
        if patches_pred_ratios_dict['Carpophyllum']+patches_pred_ratios_dict['Ecklonia']+patches_pred_ratios_dict['Other canopy']+patches_pred_ratios_dict['Foliose Algae']>25:
            recommendation = "Reef-Partial-Grazed"
            rule_numbers.append("#9")
    elif  initial_frame_level_prediction == "Unconsolidated":
        if len(urchin_boxes)>0:
            recommendation = "Reef-Urchin-Barren"
            rule_numbers.append("#10")

    if recommendation=="None":
        recommendation = initial_frame_level_prediction
        rule_numbers.append("#11")

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

    a = -0.20
    plt.text(
        0.5, a,  
        f"urchin-count by Urchinbot: {len(urchin_boxes)}",
        color="white", fontsize=22, ha="center", transform=ax.transAxes
    )
    plt.text(
        0.5, a-0.05,  
        f"initial-frame-prediction: {initial_frame_level_prediction} / conf-frame: {np.max(frame_probs):.2f}",
        color="white", fontsize=22, ha="center", transform=ax.transAxes
    )
    plt.text(
        0.5, a-0.1,  
        f"recommendation: {recommendation}",
        color="red", fontsize=22, ha="center", transform=ax.transAxes
    )
    plt.text(
        0.5, a-0.15, 
        f"used rules: {rule_numbers}",
        color="white", fontsize=22, ha="center", transform=ax.transAxes
    )
    plt.tight_layout()  
    plt.savefig(output_name+".png", bbox_inches='tight')
    return recommendation



if __name__ == '__main__':
    # input_path = os.path.join("dataset", "density-samples","inputs")
    # output_path = os.path.join("dataset", "density-samples","outputs")
    # frame_classifier_path =  os.path.join("trained_classifiers", "frame_classifier")
    # patch_classifier_path = os.path.join("trained_classifiers","patch_classifier")
    # data_dict = get_density_dicts(input_path,output_path,frame_classifier_path,patch_classifier_path)
    # get_density_heat_maps(data_dict, output_path=output_path)
    



    # frame_classifier_path =  os.path.join("trained_classifiers_", "frame_classifier")
    # patch_classifier_path = os.path.join("trained_classifiers_","patch_classifier")
    # HClassifier = HabitatClassifier(frame_classifier_path =frame_classifier_path,
    #                                 patch_classifier_path = patch_classifier_path,
    #                                 crop_ratio= 0.2)
    # image_path = os.path.join("dataset","frame7_dataset_cleaned","train", "reef_barren", "5423297_NSW49WCB5MAragunnuSouthernBay071114 (17).JPG")    
    # image_path = os.path.join("dataset","frame7_dataset_cleaned","train", "reef_grazed", "frames_17_01_20257965250_SYD38_JT7m300422MiddleHeadSth  (20).JPG")
    # image_path = os.path.join("dataset","frame7_dataset_cleaned","train", "reef_partial_barren", "frames_17_01_20258650396_GOPR5584.jpg")
    # image_path = os.path.join("dataset","frame7_dataset_cleaned","train", "reef_partial_grazed", "frames_17_01_20258638975_GOPR0243.jpg")
    # make_fig_example_for_paper(HClassifier,image_path,output_name = "1")





    frame_classifier_path =  os.path.join("trained_classifiers_", "frame_classifier")
    patch_classifier_path = os.path.join("trained_classifiers_","patch_classifier")
    HClassifier = HabitatClassifier(frame_classifier_path =frame_classifier_path,
                                    patch_classifier_path = patch_classifier_path,
                                    crop_ratio= 0.2)
    final_dataset_path = os.path.join("dataset", "final_dataset")
    image_paths = glob.glob(os.path.join(final_dataset_path, "*"))
    output_path = "predictions-final-dataset"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    recommendation_dict = {}
    for image_path in tqdm(image_paths):
        image_name = os.path.basename(image_path)
        recommendation = make_fig_example_for_paper(HClassifier,image_path,output_name = os.path.join(output_path, image_name))
        recommendation_dict.update({image_name:recommendation})
    df = pd.DataFrame(list(recommendation_dict.items()), columns=["Image Name", "AI Prediction"])
    df.to_csv(os.path.join(output_path,"recommendations.csv"), index=False)




    


    
    # start_time = time.time()
    # cap = cv2.VideoCapture("33.MP4")
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_id = 0
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
    # out = cv2.VideoWriter("outtt.mp4", fourcc, fps, (width, height),  isColor=True)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         cap.release()
    #         break  
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     img,patches_boxes,patches_centers,patches_pred_ratios_dict,patches_preds_categorial,patches_probs,frame_probs,urchin_boxes = HClassifier.get_all_models_predictions(frame)
    #     processed_frame,recommendation = HClassifier.draw_results(img,patches_boxes,patches_centers,patches_pred_ratios_dict,patches_preds_categorial,patches_probs,frame_probs,urchin_boxes)
    #     processed_frame = cv2.resize(processed_frame, (width,height))
    #     out.write(processed_frame)
    
    # out.release()
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(execution_time)
    print("All GOOD")