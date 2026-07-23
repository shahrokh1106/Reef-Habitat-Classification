import pandas as pd 
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from habitat_classifier import *
import cv2
from tqdm import tqdm
import json
import random

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
    # plt.title('Classification Report on the Test Set')
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
    plt.figure(figsize=(50,25))
    ax =sns.heatmap(cm_df, annot = True, fmt='d', cmap='Reds',annot_kws={"size": 45},cbar=True,linewidths=0.0029, linecolor="black")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=45) 
    plt.title('Confusion Matrix on the Test Set', fontsize=40,pad=20)
    plt.ylabel('Actual Values', fontsize=40)
    plt.xlabel('Predicted Values', fontsize=40)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=45)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=45)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "confusion_matrix_"+model_name+".png"))
    plt.close()

def get_prediction(HClassifier, image_path):
    """Get prediction for an image without saving plots"""
    recommendation = make_fig_example_for_paper(HClassifier,image_path,output_name = "4")
    return recommendation

def apply_strong_augmentation(image):
    """
    Apply strong augmentations to an image to test model robustness.
    Combines multiple augmentation techniques.
    """
    # Convert to float for processing
    img = image.astype(np.float32)
    h, w = img.shape[:2]
    
    # 1. Random brightness adjustment
    brightness_delta = np.random.uniform(-40, 40)
    img = np.clip(img + brightness_delta, 0, 255)
    
    # 2. Random contrast adjustment
    contrast_factor = np.random.uniform(0.6, 1.4)
    img = np.clip((img - 128) * contrast_factor + 128, 0, 255)
    
    # 3. Random color channel adjustments (simulating color jitter)
    img[:, :, 0] = np.clip(img[:, :, 0] * np.random.uniform(0.7, 1.3), 0, 255)  # Red
    img[:, :, 1] = np.clip(img[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)  # Green
    img[:, :, 2] = np.clip(img[:, :, 2] * np.random.uniform(0.7, 1.3), 0, 255)  # Blue
    
    # 4. Random rotation
    angle = np.random.uniform(-15, 15)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 5. Random horizontal flip (50% chance)
    if np.random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    # 6. Random vertical flip (50% chance)
    if np.random.random() > 0.5:
        img = cv2.flip(img, 0)
    
    # 7. Random Gaussian blur
    if np.random.random() > 0.5:
        kernel_size = np.random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # 8. Random noise
    if np.random.random() > 0.5:
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 255)
    
    # 9. Random crop and resize (zoom effect)
    if np.random.random() > 0.5:
        crop_factor = np.random.uniform(0.85, 1.0)
        new_h, new_w = int(h * crop_factor), int(w * crop_factor)
        start_y = np.random.randint(0, h - new_h + 1)
        start_x = np.random.randint(0, w - new_w + 1)
        img = img[start_y:start_y+new_h, start_x:start_x+new_w]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 10. Random saturation adjustment
    img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation_factor = np.random.uniform(0.5, 1.5)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation_factor, 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return img.astype(np.uint8)

def apply_water_effect(image, level):
    """
    Apply realistic underwater water effect to image based on level (1-5, where 1 is lowest and 5 is most intense)
    Effects include: blue-green tint, haze/opacity, red channel reduction, contrast reduction, blur
    """
    # Define parameters for each level with more pronounced underwater effects
    if level == 1:
        # Light underwater effect
        blur_kernel = 3
        contrast = 0.85
        brightness = -8
        red_reduction = 0.70  # Reduce red channel (red light doesn't penetrate water)
        blue_boost = 1.15     # Boost blue channel
        green_boost = 1.10    # Boost green channel
        overlay_intensity = 0.15  # Blue-green overlay opacity
    elif level == 2:
        # Moderate underwater effect
        blur_kernel = 5
        contrast = 0.75
        brightness = -12
        red_reduction = 0.55
        blue_boost = 1.25
        green_boost = 1.18
        overlay_intensity = 0.25
    elif level == 3:
        # Strong underwater effect
        blur_kernel = 7
        contrast = 0.65
        brightness = -16
        red_reduction = 0.45
        blue_boost = 1.35
        green_boost = 1.25
        overlay_intensity = 0.35
    elif level == 4:
        # Very strong underwater effect
        blur_kernel = 9
        contrast = 0.55
        brightness = -20
        red_reduction = 0.35
        blue_boost = 1.45
        green_boost = 1.32
        overlay_intensity = 0.45
    else:  # level == 5
        # Extreme underwater effect
        blur_kernel = 11
        contrast = 0.45
        brightness = -25
        red_reduction = 0.25
        blue_boost = 1.55
        green_boost = 1.40
        overlay_intensity = 0.55
    
    # Convert to float for processing
    image = image.astype(np.float32)
    
    # Step 1: Apply blur for haze effect (underwater particles scatter light)
    if blur_kernel > 1:
        image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    
    # Step 2: Apply underwater color shift - red light doesn't penetrate water well
    # Note: image is in RGB format, so channel 0=Red, 1=Green, 2=Blue
    image[:, :, 0] = image[:, :, 0] * red_reduction  # Red channel (strongly reduce)
    image[:, :, 2] = image[:, :, 2] * blue_boost     # Blue channel (boost)
    image[:, :, 1] = image[:, :, 1] * green_boost    # Green channel (boost)
    
    # Clip values to valid range before further processing
    image = np.clip(image, 0, 255)
    
    # Step 3: Create blue-green overlay for underwater tint (opaque/haze effect)
    h, w = image.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    # Create a cyan-blue tint (underwater color)
    overlay[:, :, 0] = 15 + level * 8   # Very little red
    overlay[:, :, 1] = 100 + level * 20 # Green
    overlay[:, :, 2] = 150 + level * 25 # Strong blue
    
    # Blend overlay with image for opacity/haze effect
    image = (1 - overlay_intensity) * image + overlay_intensity * overlay
    image = np.clip(image, 0, 255)
    
    # Step 4: Apply contrast and brightness adjustment (affects overall visibility)
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    image = image.astype(np.float32)  # Convert back to float for final haze
    
    # Step 5: Additional light scattering haze (simulates depth and particles)
    # Create a uniform blue-white haze that increases with depth
    haze_r = 160 + level * 8   # Red component of haze
    haze_g = 180 + level * 10  # Green component of haze  
    haze_b = 200 + level * 12  # Blue component of haze
    haze = np.full((h, w, 3), [haze_r, haze_g, haze_b], dtype=np.float32)
    haze_intensity = 0.08 + level * 0.05  # Increasing haze with level
    image = (1 - haze_intensity) * image + haze_intensity * haze
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image


if __name__ == '__main__':
    results = dict()
    csv_agreements_path = os.path.join("dataset", "dataset_csv_files", "User_Agreement_2.xlsx")
    df = pd.read_excel(csv_agreements_path)
    df = df.iloc[:329]  
    
    print("Total Number of Images including unscorable: ", len(df))

    
    unscorable = df ["Unnamed: 3"].to_numpy()
    unscorable_mask = unscorable=='*'
    df = df[~unscorable_mask]
    image_names = df["Image Name"].to_numpy()
    preds = df["Recommendation"].to_numpy()

    
    print("Number of unscorable images taged with NA: ", unscorable_mask.sum())
    print("Total numner of images excluding unscorable images: ", len(df))

    labels = []
    new_preds = []
    overal_agreements= df["Correct Recommendation"].to_numpy()
    filter_groups =df["Incorrect Recommendation Condition"].to_numpy()
    true_labels_by_experts = df['Unnamed: 13'].to_numpy()
    overal_no_agreements_num = 0
    count1,count2,count3, count4 = 0,0,0,0
    incorect_predictions_num = 0
    corect_predictions_num = 0
    
    image_names_valid = []
    for index,l in enumerate(overal_agreements):
        if l=="No agreement":
            overal_no_agreements_num+=1
        else:
            
            if filter_groups[index]==1:
                count1+=1
            if filter_groups[index]==2:
                count2+=1
            if filter_groups[index]==3:
                count3+=1
            if filter_groups[index]==4:
                count4+=1
                continue
            image_names_valid.append(image_names[index])
            new_preds.append(preds[index])
            if l==True:
                labels.append(preds[index])
                corect_predictions_num+=1
            else:
                labels.append(true_labels_by_experts[index])
                incorect_predictions_num+=1
            
    print("Total number of images with condition-1: ", count1)
    print("Total number of images with condition-2: ", count2)
    print("Total number of images with condition-3: ", count3)
    print("Total number of images with condition-4: ", count4)
    print("total number of images with no agreements: ", overal_no_agreements_num)
    print("Total number of images excluding those images with consition-4 and no agreements: ", len(labels))
    print("Total number of images under evaluation: ", len(new_preds))
    print("Total number of incorect predictions determined by the experts: ", incorect_predictions_num)
    print("Total number of corect predictions determined by the experts: ", corect_predictions_num)

    new_preds = ["Reef-BrLfa" if item == "Reef-Grazed" else item for item in new_preds]
    new_preds = ["Reef-Partial-BrLfa" if item == "Reef-Partial-Grazed" else item for item in new_preds]
    new_preds = ["Reef-FnEc" if item == "Reef-Vegetated" else item for item in new_preds]

    labels = ["Reef-BrLfa" if item == "Reef-Grazed" else item for item in labels]
    labels = ["Reef-Partial-BrLfa" if item == "Reef-Partial-Grazed" else item for item in labels]
    labels = ["Reef-FnEc" if item == "Reef-Vegetated" else item for item in labels]

    print(len(labels), len(new_preds), len(image_names_valid))

    # Setup paths
    robust_results_path = "robust_results"
    if not os.path.exists(robust_results_path):
        os.makedirs(robust_results_path)
    
    summary_json_path = os.path.join(robust_results_path, "summary_results.json")
    detailed_json_path = os.path.join(robust_results_path, "detailed_predictions.json")
    
    # Check if results already exist
    if os.path.exists(summary_json_path) and os.path.exists(detailed_json_path):
        print(f"\n{'='*60}")
        print("Found existing results! Loading from JSON files...")
        print(f"{'='*60}\n")
        
        # Load existing results
        with open(summary_json_path, 'r') as f:
            summary_results = json.load(f)
        with open(detailed_json_path, 'r') as f:
            predictions_data = json.load(f)
        
        # Extract data from loaded results
        base_accuracy = summary_results['base']['accuracy']
        base_report = summary_results['base']['report']
        all_level_results = []
        
        for level_data in summary_results['levels']:
            all_level_results.append({
                'level': level_data['level'],
                'accuracy': level_data['accuracy'],
                'report': level_data['report']
            })
        
        # Reconstruct labels and predictions from detailed data
        labels = predictions_data['ground_truth']
        new_predictions = predictions_data['base_predictions']
        selected_image_names = predictions_data.get('image_names', [])
        selected_base_predictions = predictions_data.get('base_predictions', [])
        dataset_path = os.path.join("dataset", "final_dataset")
        
        # Get base_labels for visualization (filter out None predictions)
        valid_indices_base = [i for i, (l, p) in enumerate(zip(labels, new_predictions)) 
                             if p is not None and l is not None]
        base_labels = [labels[i] for i in valid_indices_base] if valid_indices_base else []
        
        # Reconstruct level predictions for change analysis
        for level_result in all_level_results:
            level_num = level_result['level']
            if f'level_{level_num}' in predictions_data['level_predictions']:
                level_result['predictions'] = predictions_data['level_predictions'][f'level_{level_num}']
            else:
                level_result['predictions'] = []
        
        print("Results loaded successfully!")
        print(f"Base Accuracy: {base_accuracy:.4f}")
        print(f"Number of levels: {len(all_level_results)}")
        
    else:
        print(f"\n{'='*60}")
        print("Starting Robustness Test with Water Effects")
        print(f"{'='*60}\n")
        
        # Initialize classifier
        frame_classifier_path = os.path.join("trained_classifiers_", "frame_classifier")
        patch_classifier_path = os.path.join("trained_classifiers_", "patch_classifier")
        HClassifier = HabitatClassifier(frame_classifier_path=frame_classifier_path,
                                        patch_classifier_path=patch_classifier_path,
                                        crop_ratio=0.2)
        
        dataset_path = os.path.join("dataset", "final_dataset")
        
        # Step 1: Get predictions from model on all original images
        print("Getting predictions from model on original images...")
        model_predictions = []
        for img_name in tqdm(image_names_valid, desc="Model predictions"):
            img_path = os.path.join(dataset_path, img_name)
            pred = get_prediction(HClassifier, img_path)
            model_predictions.append(pred)
        
        # Normalize prediction labels to match ground truth format
        model_predictions_normalized = []
        for pred in model_predictions:
            if pred is None:
                model_predictions_normalized.append(None)
            else:
                pred_norm = pred
                pred_norm = "Reef-BrLfa" if pred_norm == "Reef-Grazed" else pred_norm
                pred_norm = "Reef-Partial-BrLfa" if pred_norm == "Reef-Partial-Grazed" else pred_norm
                pred_norm = "Reef-FnEc" if pred_norm == "Reef-Vegetated" else pred_norm
                model_predictions_normalized.append(pred_norm)
        
        # Step 2: Find correct predictions grouped by class
        correct_indices_by_class = {}
        for idx, (label, pred) in enumerate(zip(labels, model_predictions_normalized)):
            if pred is not None and label is not None and pred == label:
                if label not in correct_indices_by_class:
                    correct_indices_by_class[label] = []
                correct_indices_by_class[label].append(idx)
        
        # Print statistics per class
        print(f"\nCorrect predictions by class:")
        total_correct = 0
        for cls, indices in sorted(correct_indices_by_class.items()):
            print(f"  {cls}: {len(indices)} correct predictions")
            total_correct += len(indices)
        print(f"Total correct predictions: {total_correct}/{len(labels)}")
        
        # Step 3: Randomly select 10 images per class from correct predictions (balanced selection)
        random.seed(42)  # Set seed for reproducibility
        images_per_class = 10
        target_total = 100
        selected_indices = []
        
        # First, select balanced samples (10 per class)
        for cls, indices in sorted(correct_indices_by_class.items()):
            if len(indices) < images_per_class:
                print(f"Warning: Class '{cls}' has only {len(indices)} correct predictions, using all of them")
                selected_indices.extend(indices)
            else:
                selected_indices.extend(random.sample(indices, images_per_class))
        
        print(f"\nSelected {len(selected_indices)} images from balanced selection ({images_per_class} per class)")
        
        # If total is less than target_total, randomly select additional images from remaining correct predictions
        if len(selected_indices) < target_total:
            # Get all correct indices
            all_correct_indices = []
            for indices in correct_indices_by_class.values():
                all_correct_indices.extend(indices)
            
            # Find remaining correct predictions (not yet selected)
            remaining_indices = [idx for idx in all_correct_indices if idx not in selected_indices]
            
            if len(remaining_indices) > 0:
                needed = target_total - len(selected_indices)
                if len(remaining_indices) >= needed:
                    additional_indices = random.sample(remaining_indices, needed)
                    selected_indices.extend(additional_indices)
                    print(f"Added {needed} additional images from remaining correct predictions to reach {target_total} total")
                else:
                    selected_indices.extend(remaining_indices)
                    print(f"Warning: Only {len(remaining_indices)} additional correct predictions available. Total: {len(selected_indices)} (target: {target_total})")
            else:
                print(f"Warning: No remaining correct predictions available. Total: {len(selected_indices)} (target: {target_total})")
        
        selected_indices.sort()  # Sort for consistency
        print(f"\nFinal selection: {len(selected_indices)} images for robustness testing")
        
        # Extract selected data
        selected_image_names = [image_names_valid[i] for i in selected_indices]
        selected_labels = [labels[i] for i in selected_indices]
        selected_base_predictions = [model_predictions_normalized[i] for i in selected_indices]
        
        # Step 4: Create level_0 folder and save original images
        level_0_folder = os.path.join(robust_results_path, "level_0")
        if not os.path.exists(level_0_folder):
            os.makedirs(level_0_folder)
        
        print(f"\nSaving {len(selected_image_names)} original images to level_0...")
        for img_name in tqdm(selected_image_names, desc="Saving level_0"):
            original_img_path = os.path.join(dataset_path, img_name)
            output_img_path = os.path.join(level_0_folder, img_name)
            # Copy original image to level_0
            image = cv2.imread(original_img_path)
            if image is not None:
                cv2.imwrite(output_img_path, image)
        
        # Step 5: Calculate base metrics and create plots
        print(f"\n{'='*60}")
        print("Calculating Base Metrics (Original Images)")
        print(f"{'='*60}")
        
        base_accuracy = accuracy_score(selected_labels, selected_base_predictions)
        base_report = classification_report(selected_labels, selected_base_predictions, zero_division=0, output_dict=True)
        
        # Save base metrics plots
        get_confusion_matrix_plot(selected_base_predictions, selected_labels, robust_results_path, "base")
        get_classification_report_plot(selected_base_predictions, selected_labels, robust_results_path, "base")
        
        print(f"Base - Accuracy: {base_accuracy:.4f}")
        print(f"Base - Total samples: {len(selected_labels)}")
        
        # Step 6: Create subfolders for each water effect level (1-5)
        level_folders = []
        for level in range(1, 6):
            level_folder = os.path.join(robust_results_path, f"level_{level}")
            if not os.path.exists(level_folder):
                os.makedirs(level_folder)
            level_folders.append(level_folder)
        
        # Step 7: Process each water effect level on selected images (balanced 10 per class, up to 100 total)
        all_level_results = []
        
        for level in range(1, 6):
            print(f"\n{'='*60}")
            print(f"Processing Water Effect Level {level}")
            print(f"{'='*60}")
            
            level_folder = level_folders[level - 1]
            level_predictions = []
            
            # Apply water effects and get predictions
            for img_name in tqdm(selected_image_names, desc=f"Level {level}"):
                # Load original image
                original_img_path = os.path.join(dataset_path, img_name)
                image = cv2.imread(original_img_path)
                
                if image is None:
                    print(f"Warning: Could not load {img_name}")
                    level_predictions.append(None)
                    continue
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply water effect
                modified_image = apply_water_effect(image.copy(), level)
                
                # Save modified image
                output_img_path = os.path.join(level_folder, img_name)
                # Convert back to BGR for saving
                modified_image_bgr = cv2.cvtColor(modified_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_img_path, modified_image_bgr)
                
                # Get prediction on modified image
                pred = get_prediction(HClassifier, output_img_path)
                if pred is not None:
                    # Normalize prediction
                    pred_norm = pred
                    pred_norm = "Reef-BrLfa" if pred_norm == "Reef-Grazed" else pred_norm
                    pred_norm = "Reef-Partial-BrLfa" if pred_norm == "Reef-Partial-Grazed" else pred_norm
                    pred_norm = "Reef-FnEc" if pred_norm == "Reef-Vegetated" else pred_norm
                    level_predictions.append(pred_norm)
                else:
                    level_predictions.append(None)
            
            # Calculate metrics for this level
            valid_indices = [i for i, (l, p) in enumerate(zip(selected_labels, level_predictions)) 
                            if p is not None and l is not None]
            
            if len(valid_indices) == 0:
                print(f"Warning: No valid predictions for level {level}")
                continue
            
            level_labels = [selected_labels[i] for i in valid_indices]
            level_preds = [level_predictions[i] for i in valid_indices]
            
            accuracy = accuracy_score(level_labels, level_preds)
            report = classification_report(level_labels, level_preds, zero_division=0, output_dict=True)
            
            level_result = {
                'level': level,
                'accuracy': accuracy,
                'report': report,
                'labels': level_labels,
                'predictions': level_predictions,  # Full list with None values
                'valid_indices': valid_indices
            }
            all_level_results.append(level_result)
            
            # Save confusion matrix and classification report for this level
            get_confusion_matrix_plot(level_preds, level_labels, robust_results_path, f"level_{level}")
            get_classification_report_plot(level_preds, level_labels, robust_results_path, f"level_{level}")
            
            print(f"Level {level} - Accuracy: {accuracy:.4f}")
            print(f"Level {level} - Total valid predictions: {len(valid_indices)}/{len(selected_labels)}")
        
        # Save summary results to JSON
        summary_results = {
            'base': {
                'accuracy': float(base_accuracy),
                'total_samples': len(selected_labels),
                'report': {k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float, np.number))} 
                         for k, v in base_report.items() if isinstance(v, dict)}
            },
            'levels': []
        }
        
        for level_result in all_level_results:
            summary_results['levels'].append({
                'level': level_result['level'],
                'accuracy': float(level_result['accuracy']),
                'total_samples': len(level_result['valid_indices']),
                'report': {k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float, np.number))} 
                          for k, v in level_result['report'].items() if isinstance(v, dict)}
            })
        
        with open(summary_json_path, 'w') as f:
            json.dump(summary_results, f, indent=4)
        
        # Save detailed predictions
        predictions_data = {
            'image_names': selected_image_names,
            'ground_truth': selected_labels,
            'base_predictions': selected_base_predictions,
            'level_predictions': {}
        }
        
        for level_idx, level_result in enumerate(all_level_results, 1):
            predictions_data['level_predictions'][f'level_{level_idx}'] = level_result['predictions']
        
        with open(detailed_json_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        # Update variables for visualization
        labels = selected_labels
        new_predictions = selected_base_predictions
        selected_image_names = selected_image_names
        selected_base_predictions = selected_base_predictions
        dataset_path = os.path.join("dataset", "final_dataset")
    
    # ===== AUGMENTATION EXPERIMENT =====
    # Check if augmentation results already exist
    aug_json_path = os.path.join(robust_results_path, "augmentation_results.json")
    
    if os.path.exists(aug_json_path):
        print(f"\n{'='*60}")
        print("Found existing augmentation results! Loading from JSON file...")
        print(f"{'='*60}\n")
        
        with open(aug_json_path, 'r') as f:
            aug_results = json.load(f)
        
        # Load augmented predictions from detailed predictions if available
        if os.path.exists(detailed_json_path):
            with open(detailed_json_path, 'r') as f:
                predictions_data = json.load(f)
            if 'augmented_predictions' in predictions_data:
                aug_predictions = predictions_data['augmented_predictions']
            else:
                aug_predictions = aug_results.get('augmented_predictions', [])
        else:
            aug_predictions = aug_results.get('augmented_predictions', [])
        
        # Get base predictions for visualization
        if 'selected_base_predictions' not in locals():
            if os.path.exists(detailed_json_path):
                with open(detailed_json_path, 'r') as f:
                    predictions_data = json.load(f)
                selected_base_predictions = predictions_data.get('base_predictions', [])
            else:
                selected_base_predictions = []
        
        print("Augmentation results loaded successfully!")
        print(f"Overall Consistency: {aug_results.get('overall_consistency', 0.0):.4f}")
        
        # Create visualizations from loaded results
        if 'consistency_by_class' in aug_results or aug_results.get('consistency_by_class'):
            consistency_by_class = aug_results.get('consistency_by_class', {})
            if consistency_by_class:
                print("\nCreating consistency visualizations from loaded results...")
                
                # 1. Consistency rate by class (line plot)
                classes = sorted(consistency_by_class.keys())
                consistency_rates = [consistency_by_class[cls]['consistency_rate'] for cls in classes]
                
                plt.figure(figsize=(12, 6))
                plt.plot(classes, consistency_rates, marker='o', linewidth=2, markersize=8, color='steelblue')
                plt.xlabel('Class', fontsize=12)
                plt.ylabel('Consistency Rate', fontsize=12)
                plt.title('Prediction Consistency Under Strong Augmentation by Class', fontsize=14, fontweight='bold')
                plt.ylim([0, 1])
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                
                # Add value labels
                for i, (cls, rate) in enumerate(zip(classes, consistency_rates)):
                    plt.text(i, rate + 0.02, f'{rate:.3f}', ha='center', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(os.path.join(robust_results_path, "augmentation_consistency_by_class.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. Change rate by class (line plot)
                change_rates = [consistency_by_class[cls]['change_rate'] for cls in classes]
                
                plt.figure(figsize=(12, 6))
                plt.plot(classes, change_rates, marker='o', linewidth=2, markersize=8, color='coral')
                plt.xlabel('Class', fontsize=12)
                plt.ylabel('Change Rate', fontsize=12)
                plt.title('Prediction Change Rate Under Strong Augmentation by Class', fontsize=14, fontweight='bold')
                plt.ylim([0, 1])
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                
                # Add value labels
                for i, (cls, rate) in enumerate(zip(classes, change_rates)):
                    plt.text(i, rate + 0.02, f'{rate:.3f}', ha='center', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(os.path.join(robust_results_path, "augmentation_change_rate_by_class.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. Consistency heatmap (base vs augmented predictions)
                if len(selected_base_predictions) > 0 and len(aug_predictions) > 0:
                    valid_aug_indices = [i for i, (base_pred, aug_pred) in enumerate(zip(selected_base_predictions, aug_predictions))
                                        if base_pred is not None and aug_pred is not None]
                    if len(valid_aug_indices) > 0:
                        from sklearn.metrics import confusion_matrix
                        base_preds_valid = [selected_base_predictions[i] for i in valid_aug_indices]
                        aug_preds_valid = [aug_predictions[i] for i in valid_aug_indices]
                        
                        cm = confusion_matrix(base_preds_valid, aug_preds_valid)
                        cm_df = pd.DataFrame(cm, index=sorted(set(base_preds_valid)), columns=sorted(set(aug_preds_valid)))
                        
                        plt.figure(figsize=(12, 10))
                        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
                        plt.title('Prediction Consistency Matrix: Base vs Augmented', fontsize=14, fontweight='bold')
                        plt.ylabel('Base Predictions', fontsize=12)
                        plt.xlabel('Augmented Predictions', fontsize=12)
                        plt.tight_layout()
                        plt.savefig(os.path.join(robust_results_path, "augmentation_consistency_matrix.png"), dpi=300, bbox_inches='tight')
                        plt.close()
    else:
        print(f"\n{'='*60}")
        print("Starting Augmentation Consistency Experiment")
        print(f"{'='*60}\n")
        
        # Initialize classifier if not already initialized
        if 'HClassifier' not in locals():
            frame_classifier_path = os.path.join("trained_classifiers_", "frame_classifier")
            patch_classifier_path = os.path.join("trained_classifiers_", "patch_classifier")
            HClassifier = HabitatClassifier(frame_classifier_path=frame_classifier_path,
                                            patch_classifier_path=patch_classifier_path,
                                            crop_ratio=0.2)
        
        # Get image names and base predictions from loaded data if needed
        if 'selected_image_names' not in locals():
            if os.path.exists(detailed_json_path):
                with open(detailed_json_path, 'r') as f:
                    predictions_data = json.load(f)
                selected_image_names = predictions_data.get('image_names', [])
                selected_base_predictions = predictions_data.get('base_predictions', [])
            else:
                print("Error: Cannot find image names and predictions for augmentation experiment")
                selected_image_names = []
                selected_base_predictions = []
        
        if 'dataset_path' not in locals():
            dataset_path = os.path.join("dataset", "final_dataset")
        
        # Create aug folder
        aug_folder = os.path.join(robust_results_path, "aug")
        if not os.path.exists(aug_folder):
            os.makedirs(aug_folder)
        
        print("Applying strong augmentations to original images...")
        aug_predictions = []
        
        for img_name in tqdm(selected_image_names, desc="Augmenting images"):
            # Load original image
            original_img_path = os.path.join(dataset_path, img_name)
            image = cv2.imread(original_img_path)
            
            if image is None:
                print(f"Warning: Could not load {img_name}")
                aug_predictions.append(None)
                continue
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply strong augmentation
            augmented_image = apply_strong_augmentation(image.copy())
            
            # Save augmented image
            output_img_path = os.path.join(aug_folder, img_name)
            # Convert back to BGR for saving
            augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_img_path, augmented_image_bgr)
            
            # Get prediction on augmented image
            pred = get_prediction(HClassifier, output_img_path)
            if pred is not None:
                # Normalize prediction
                pred_norm = pred
                pred_norm = "Reef-BrLfa" if pred_norm == "Reef-Grazed" else pred_norm
                pred_norm = "Reef-Partial-BrLfa" if pred_norm == "Reef-Partial-Grazed" else pred_norm
                pred_norm = "Reef-FnEc" if pred_norm == "Reef-Vegetated" else pred_norm
                aug_predictions.append(pred_norm)
            else:
                aug_predictions.append(None)
        
        # Calculate consistency metrics per class
        print(f"\n{'='*60}")
        print("Calculating Consistency Metrics")
        print(f"{'='*60}")
        
        # Filter valid predictions
        valid_aug_indices = [i for i, (base_pred, aug_pred) in enumerate(zip(selected_base_predictions, aug_predictions))
                            if base_pred is not None and aug_pred is not None]
        
        if len(valid_aug_indices) == 0:
            print("Warning: No valid predictions for augmentation experiment")
            aug_results = {}
        else:
            # Group by class and calculate consistency
            consistency_by_class = {}
            all_classes = set(selected_base_predictions[i] for i in valid_aug_indices)
            
            for cls in all_classes:
                class_indices = [i for i in valid_aug_indices if selected_base_predictions[i] == cls]
                if len(class_indices) == 0:
                    continue
                
                # Count how many predictions remained the same
                consistent = sum(1 for i in class_indices 
                              if selected_base_predictions[i] == aug_predictions[i])
                total = len(class_indices)
                consistency_rate = consistent / total if total > 0 else 0.0
                
                # Count prediction changes
                changes = total - consistent
                change_rate = changes / total if total > 0 else 0.0
                
                consistency_by_class[cls] = {
                    'total': total,
                    'consistent': consistent,
                    'changed': changes,
                    'consistency_rate': consistency_rate,
                    'change_rate': change_rate
                }
                
                print(f"\n{cls}:")
                print(f"  Total samples: {total}")
                print(f"  Consistent predictions: {consistent} ({consistency_rate*100:.2f}%)")
                print(f"  Changed predictions: {changes} ({change_rate*100:.2f}%)")
            
            # Overall consistency
            overall_consistent = sum(1 for i in valid_aug_indices 
                                   if selected_base_predictions[i] == aug_predictions[i])
            overall_total = len(valid_aug_indices)
            overall_consistency = overall_consistent / overall_total if overall_total > 0 else 0.0
            
            print(f"\n{'='*60}")
            print(f"Overall Consistency: {overall_consistent}/{overall_total} ({overall_consistency*100:.2f}%)")
            print(f"{'='*60}")
            
            # Create consistency visualization
            print("\nCreating consistency visualizations...")
            
            # 1. Consistency rate by class (line plot)
            classes = sorted(consistency_by_class.keys())
            consistency_rates = [consistency_by_class[cls]['consistency_rate'] for cls in classes]
            
            plt.figure(figsize=(12, 6))
            plt.plot(classes, consistency_rates, marker='o', linewidth=2, markersize=8, color='steelblue')
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Consistency Rate', fontsize=12)
            plt.title('Prediction Consistency Under Strong Augmentation by Class', fontsize=14, fontweight='bold')
            plt.ylim([0, 1])
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (cls, rate) in enumerate(zip(classes, consistency_rates)):
                plt.text(i, rate + 0.02, f'{rate:.3f}', ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(robust_results_path, "augmentation_consistency_by_class.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Change rate by class (line plot)
            change_rates = [consistency_by_class[cls]['change_rate'] for cls in classes]
            
            plt.figure(figsize=(12, 6))
            plt.plot(classes, change_rates, marker='o', linewidth=2, markersize=8, color='coral')
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Change Rate', fontsize=12)
            plt.title('Prediction Change Rate Under Strong Augmentation by Class', fontsize=14, fontweight='bold')
            plt.ylim([0, 1])
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (cls, rate) in enumerate(zip(classes, change_rates)):
                plt.text(i, rate + 0.02, f'{rate:.3f}', ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(robust_results_path, "augmentation_change_rate_by_class.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Consistency heatmap (base vs augmented predictions)
            from sklearn.metrics import confusion_matrix
            base_preds_valid = [selected_base_predictions[i] for i in valid_aug_indices]
            aug_preds_valid = [aug_predictions[i] for i in valid_aug_indices]
            
            cm = confusion_matrix(base_preds_valid, aug_preds_valid)
            cm_df = pd.DataFrame(cm, index=sorted(set(base_preds_valid)), columns=sorted(set(aug_preds_valid)))
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
            plt.title('Prediction Consistency Matrix: Base vs Augmented', fontsize=14, fontweight='bold')
            plt.ylabel('Base Predictions', fontsize=12)
            plt.xlabel('Augmented Predictions', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(robust_results_path, "augmentation_consistency_matrix.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save augmentation results to JSON
            aug_results = {
                'overall_consistency': float(overall_consistency),
                'overall_consistent': int(overall_consistent),
                'overall_total': int(overall_total),
                'consistency_by_class': {cls: {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                             for k, v in metrics.items()} 
                                        for cls, metrics in consistency_by_class.items()},
                'augmented_predictions': aug_predictions
            }
            
            with open(aug_json_path, 'w') as f:
                json.dump(aug_results, f, indent=2)
            
            # Also update detailed predictions to include augmented predictions
            if os.path.exists(detailed_json_path):
                with open(detailed_json_path, 'r') as f:
                    predictions_data = json.load(f)
                predictions_data['augmented_predictions'] = aug_predictions
                with open(detailed_json_path, 'w') as f:
                    json.dump(predictions_data, f, indent=2)
            
            print(f"\nAugmentation results saved to: {aug_json_path}")
    # ===== END AUGMENTATION EXPERIMENT =====
    
    # Create comparison plots
    print(f"\n{'='*60}")
    print("Creating Comparison Plots")
    print(f"{'='*60}")
    
    # Accuracy comparison across levels
    accuracies = [base_accuracy] + [r['accuracy'] for r in all_level_results]
    levels = ['Base'] + [f'Level {i+1}' for i in range(len(all_level_results))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(levels, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Water Effect Level', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy Across Water Effect Levels', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(robust_results_path, "accuracy_comparison.png"), dpi=300)
    plt.close()
    
    # Per-class metrics comparison
    # Get all classes from reports (not just labels) to handle JSON loading
    all_classes = set()
    if base_report:
        all_classes.update([k for k in base_report.keys() if k not in ['macro avg', 'weighted avg', 'accuracy']])
    for level_result in all_level_results:
        if level_result['report']:
            all_classes.update([k for k in level_result['report'].keys() if k not in ['macro avg', 'weighted avg', 'accuracy']])
    all_classes = sorted(list(all_classes))
    
    # Extract F1 scores for each class across levels
    f1_scores_by_class = {cls: [] for cls in all_classes}
    
    # Base F1 scores
    for cls in all_classes:
        if cls in base_report:
            f1_scores_by_class[cls].append(base_report[cls]['f1-score'])
        else:
            f1_scores_by_class[cls].append(0.0)
    
    # Level F1 scores
    for level_result in all_level_results:
        report = level_result['report']
        for cls in all_classes:
            if cls in report:
                f1_scores_by_class[cls].append(report[cls]['f1-score'])
            else:
                f1_scores_by_class[cls].append(0.0)
    
    # Plot F1 scores by class
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(levels))
    width = 0.8 / len(all_classes)
    
    for idx, cls in enumerate(all_classes):
        offset = (idx - len(all_classes)/2) * width + width/2
        ax.bar(x + offset, f1_scores_by_class[cls], width, label=cls, alpha=0.8)
    
    ax.set_xlabel('Water Effect Level', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score by Class Across Water Effect Levels', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(robust_results_path, "f1_score_by_class.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prediction change analysis
    prediction_changes = []
    if 'predictions' in all_level_results[0] if all_level_results else False:
        for level_idx, level_result in enumerate(all_level_results, 1):
            changes = 0
            total = 0
            # Compare base predictions with level predictions for same images
            if 'predictions' in level_result:
                for i in range(len(new_predictions)):
                    if i < len(new_predictions) and i < len(level_result['predictions']):
                        if new_predictions[i] is not None and level_result['predictions'][i] is not None:
                            if new_predictions[i] != level_result['predictions'][i]:
                                changes += 1
                            total += 1
            if total > 0:
                change_rate = changes / total
                prediction_changes.append(change_rate)
            else:
                prediction_changes.append(0.0)
    else:
        # If predictions not available, use empty list
        prediction_changes = [0.0] * len(all_level_results)
    
    if prediction_changes:
        plt.figure(figsize=(10, 6))
        plt.plot([f'Level {i+1}' for i in range(len(prediction_changes))], prediction_changes, marker='o', linewidth=2, markersize=8, color='red')
        plt.xlabel('Water Effect Level', fontsize=12)
        plt.ylabel('Prediction Change Rate', fontsize=12)
        plt.title('Prediction Change Rate from Base Predictions', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        for i, rate in enumerate(prediction_changes):
            plt.text(i, rate + 0.02, f'{rate:.3f}', ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(robust_results_path, "prediction_change_rate.png"), dpi=300)
        plt.close()
    
    # ===== NEW VISUALIZATIONS =====
    print(f"\n{'='*60}")
    print("Creating Advanced Visualizations")
    print(f"{'='*60}")
    
    # all_classes is already defined above from reports
    
    # 1. F1 Score Drop Plot (line plot showing drop from base)
    print("Creating F1 Score Drop Plot...")
    f1_drops_by_class = {cls: [] for cls in all_classes}
    
    # Get base F1 scores
    base_f1_scores = {}
    for cls in all_classes:
        if cls in base_report and 'f1-score' in base_report[cls]:
            base_f1_scores[cls] = base_report[cls]['f1-score']
        else:
            base_f1_scores[cls] = 0.0
    
    # Calculate F1 score drops for each level
    level_names = [f'Level {i+1}' for i in range(len(all_level_results))]
    for level_result in all_level_results:
        report = level_result['report']
        for cls in all_classes:
            if cls in report and 'f1-score' in report[cls]:
                f1_score = report[cls]['f1-score']
                drop = base_f1_scores[cls] - f1_score
                f1_drops_by_class[cls].append(drop)
            else:
                f1_drops_by_class[cls].append(base_f1_scores[cls])  # Full drop if class not found
    
    # Plot F1 score drops
    plt.figure(figsize=(12, 7))
    for cls in all_classes:
        plt.plot(level_names, f1_drops_by_class[cls], marker='o', linewidth=2, markersize=6, label=cls)
    plt.xlabel('Water Effect Level', fontsize=12)
    plt.ylabel('F1 Score Drop (from Base)', fontsize=12)
    plt.title('F1 Score Performance Drop Across Water Effect Levels', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(robust_results_path, "f1_score_drop.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heat Maps for Metrics and Levels
    print("Creating Heat Maps for Metrics...")
    
    # Prepare data for heat map: metrics (precision, recall, f1-score) x levels (base + 5 levels)
    metrics = ['precision', 'recall', 'f1-score']
    heat_levels = ['Base'] + level_names
    
    # Create heat map for each class (metrics as rows, levels as columns)
    for cls in all_classes:
        heat_data = {}
        
        # Collect metrics for each level
        for metric in metrics:
            metric_values = []
            # Base metric
            if cls in base_report and metric in base_report[cls]:
                metric_values.append(base_report[cls][metric])
            else:
                metric_values.append(0.0)
            # Level metrics
            for level_result in all_level_results:
                report = level_result['report']
                if cls in report and metric in report[cls]:
                    metric_values.append(report[cls][metric])
                else:
                    metric_values.append(0.0)
            heat_data[metric] = metric_values
        
        # Create heat map (metrics as rows, levels as columns)
        heat_df = pd.DataFrame(heat_data, index=heat_levels)
        heat_df = heat_df.T  # Transpose so metrics are rows, levels are columns
        plt.figure(figsize=(10, 4))
        sns.heatmap(heat_df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Score'}, vmin=0, vmax=1, 
                    xticklabels=heat_levels, yticklabels=metrics)
        plt.title(f'{cls} - Metrics Across Water Effect Levels', fontsize=14, fontweight='bold')
        plt.ylabel('Metric', fontsize=12)
        plt.xlabel('Water Effect Level', fontsize=12)
        plt.tight_layout()
        # Clean class name for filename
        cls_filename = cls.replace(' ', '_').replace('/', '_')
        plt.savefig(os.path.join(robust_results_path, f"heatmap_{cls_filename}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create overall heat map for all classes (F1 scores)
    print("Creating Overall F1 Score Heat Map...")
    overall_heat_data = []
    for cls in all_classes:
        row = [base_f1_scores[cls]]
        for level_result in all_level_results:
            report = level_result['report']
            if cls in report and 'f1-score' in report[cls]:
                row.append(report[cls]['f1-score'])
            else:
                row.append(0.0)
        overall_heat_data.append(row)
    
    overall_heat_df = pd.DataFrame(overall_heat_data, index=all_classes, columns=heat_levels)
    plt.figure(figsize=(10, max(6, len(all_classes) * 0.5)))
    sns.heatmap(overall_heat_df, annot=True, fmt='.3f', cmap='RdYlGn', cbar_kws={'label': 'F1 Score'}, vmin=0, vmax=1)
    plt.title('F1 Scores by Class Across Water Effect Levels', fontsize=14, fontweight='bold')
    plt.ylabel('Class', fontsize=12)
    plt.xlabel('Water Effect Level', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(robust_results_path, "heatmap_f1_all_classes.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Advanced visualizations created successfully!")
    
    # Print summary
    print(f"\n{'='*60}")
    print("ROBUSTNESS TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Base Accuracy: {base_accuracy:.4f}")
    print("\nAccuracy by Level:")
    for level_result in all_level_results:
        print(f"  Level {level_result['level']}: {level_result['accuracy']:.4f}")
    print(f"\nAccuracy Drop:")
    for level_result in all_level_results:
        drop = base_accuracy - level_result['accuracy']
        if base_accuracy > 0:
            print(f"  Level {level_result['level']}: {drop:.4f} ({drop/base_accuracy*100:.2f}%)")
        else:
            print(f"  Level {level_result['level']}: {drop:.4f}")
    if prediction_changes:
        print(f"\nPrediction Change Rate:")
        for i, rate in enumerate(prediction_changes, 1):
            print(f"  Level {i}: {rate:.4f} ({rate*100:.2f}%)")
    print(f"\n{'='*60}")
    print(f"All results saved to: {robust_results_path}")
    print(f"{'='*60}\n")
    print("DONE")



