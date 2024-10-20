import matplotlib.pyplot as plt 
import os
import glob
from multiclassifier import *

def plot_original_dataset():
    actual_class_names = {"reef_barren": 'Reef-Barren',
                      "reef_kelp": "Reef-kelp",
                      "reef_partial_barren": "Reef-Partial-Barren", 
                      "reef_vegetated": "Reef-Vegetated",
                      "unconsolidated": 'Unconsolidated'}
    frame_dataset_path = os.path.join(".", "dataset", "grouped_frames_with_whole_annotations")
    if not os.path.exists(frame_dataset_path):
       print("Plase first download frames_with_whole_annotations using download_frames_with_whole_annotations.py")
       print("Next, run grouped_frames_with_whole_annotations.py")
    else:
        class_paths = glob.glob(os.path.join(frame_dataset_path, "*"))
        frame_dataset_frequency = dict()
        for class_path in class_paths:
            class_name = os.path.basename(class_path)
            frame_dataset_frequency.update({actual_class_names[class_name]:len(glob.glob(os.path.join(class_path, "*")))})
        fsize = 20
        plt.style.use('default') 
        plt.rcParams['font.family'] = 'serif'  
        plt.rcParams['font.serif'] = ['Times New Roman'] 
        # Plotting the dictionary with horizontal bars
        plt.figure(figsize=(15, 8))
        bars = plt.barh(list(frame_dataset_frequency.keys()), list(frame_dataset_frequency.values()))
        plt.ylabel('Class names',fontsize=fsize)
        plt.xlabel('Frequency',fontsize=fsize)
        plt.title('Class frequency in the original dataset',fontsize=fsize)
        plt.xticks(fontsize=fsize-3)
        plt.yticks(fontsize=fsize-3)
        for bar in bars:
            plt.text(bar.get_width() + 0.1, 
                    bar.get_y() + bar.get_height()/2,  
                    f'{bar.get_width()}',  
                    va='center',  
                    fontsize=fsize-5)  
        plt.title('Label frequency in the the prepared whole frame dataset', fontsize=fsize)
        plt.tight_layout()
        plt.savefig(os.path.join("dataset", "figs_dataset_analysis", "label_frequency_whole_frame.png"))
        plt.close()


def generate_random_points(image_shape, num_points, border_margin, min_distance):
    """
    Generate random points inside an image making sure they are not close to the borders
    and have a minimum distance between each other.

    Parameters:
    image_shape (tuple): Shape of the image (height, width).
    num_points (int): Number of random points to generate.
    border_margin (int): Minimum distance from the border.
    min_distance (float): Minimum distance between any two points.

    Returns:
    numpy array of shape (num_points, 2) containing the (x, y) coordinates of the random points.
    """
    def generate_single_point():
        random_x = np.random.uniform(low=valid_x_range[0], high=valid_x_range[1])
        random_y = np.random.uniform(low=valid_y_range[0], high=valid_y_range[1])
        return np.stack([random_x, random_y])
    def is_valid_point(point, points, min_distance):
        if len(points) == 0:
            return True
        distances = np.linalg.norm(np.array(points) - np.array(point), axis=1)
        return np.all(distances >= min_distance)
        
    height, width=image_shape[0],image_shape[1]
    valid_x_range = (border_margin, width - (border_margin))
    valid_y_range = (border_margin, height - (border_margin))
    points = []
    while len(points) < num_points:
        point = generate_single_point()
        if is_valid_point(point, points, min_distance):
            points.append([int(point[0]),int(point[1])])
    return np.asarray(points)  

def generate_grid_with_total_points(image_shape, total_points, margin_x, margin_y):
    """
    Generates a grid of points on an image with a specified total number of points and margins around the grid.

    Parameters:
    - image_shape: A tuple (height, width) of the image.
    - total_points: The total number of points in the grid.
    - margin_x: Margin from the left and right edges of the image (in pixels).
    - margin_y: Margin from the top and bottom edges of the image (in pixels).

    Returns:
    - A numpy array of shape (total_points, 2), where each row is (x, y) coordinate of a grid point.
    """
    height, width = image_shape

    # Calculate the usable area (inside the margins)
    usable_width = width - 2 * margin_x
    usable_height = height - 2 * margin_y

    # Calculate aspect ratio of the usable area
    aspect_ratio = usable_width / usable_height

    # Estimate the number of points along each axis that best fits the total_points
    num_points_y = int(np.sqrt(total_points / aspect_ratio))
    num_points_x = total_points // num_points_y

    # If the product of num_points_x and num_points_y is less than total_points, adjust num_points_x
    if num_points_x * num_points_y < total_points:
        num_points_x += 1

    # Create evenly spaced coordinates along x and y within the margins
    x_coords = np.linspace(margin_x, width - margin_x, num_points_x)
    y_coords = np.linspace(margin_y, height - margin_y, num_points_y)

    # Create the grid of points
    grid_points = np.array([[int(x), int(y)] for x in x_coords for y in y_coords])

    # If we have more points than needed, trim the extra points
    if grid_points.shape[0] > total_points:
        grid_points = grid_points[:total_points]

    return grid_points

def get_patches(points, image,crop_size,model_image_size):
    patches = []
    for i in range(len(points)):
        center = points[i]
        patch = image[center[1]-(crop_size//2):center[1]+(crop_size//2),center[0]-(crop_size//2):center[0]+(crop_size//2)]
        patch = cv2.resize(patch, (model_image_size,model_image_size))
        patches.append(patch)
    return patches

def preprocess_patches (patches,model_image_size):
    patches = tf.convert_to_tensor(patches)
    patches = tf.image.resize(patches, [model_image_size, model_image_size])
    patches = preprocess_input_inception(patches)  
    return patches

def get_model_predictions (model,image_list, num_points, crop_percentage=0.2, point_generation = "random",batch_size = None,verbose = 1):
    model_image_size = model.input.shape[1]
    patches_for_all_images = []
    for image in image_list:
        if isinstance(image, str):
            image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop_size = int(((image.shape[1]+image.shape[0])/2)*crop_percentage)
        image_shape = (image.shape[0], image.shape[1])
        border_margin = crop_size//2
        min_distance = np.sqrt(image_shape[0] * image_shape[1] / (num_points*10))
        if point_generation == "grid":
            points = generate_grid_with_total_points(image_shape, num_points, border_margin, border_margin)
        else:
            points = generate_random_points(image_shape=image_shape, num_points=num_points, border_margin=border_margin, min_distance=min_distance)
        patches = get_patches(points, image,crop_size,model_image_size)
        patches_array = np.stack(patches, axis=0)
        patches_for_all_images.append(patches_array)
    patches_for_all_images = np.concatenate(patches_for_all_images)
    patches_for_all_images_tensors = preprocess_patches (patches_for_all_images,model_image_size)
    
    patches_for_all_images_logits = model.predict(patches_for_all_images_tensors,batch_size=batch_size,verbose=verbose)
    return patches_for_all_images_logits

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model



def majvote_on_img_predictions(patches_for_all_images_logits,num_images,num_points):
    class_names ={0: 'Carpophyllum spp',1: 'Ecklonia',2: 'Foliose Algae',3: 'Other canopy forming macroalgae',4: 'Unconsolidated',5: 'Urchin',6: 'Grazed rock'}
    label_pred_list = []
    for i in range(num_images):
        patches_predictions_logits = patches_for_all_images_logits[i*num_points:(i*num_points)+num_points]
        patches_predictions_binary = np.zeros_like(patches_predictions_logits)
        patches_predictions_binary[np.arange(len(patches_predictions_logits)), np.argmax(patches_predictions_logits, axis=1)] = 1
        patches_predictions_categorial = np.argmax(patches_predictions_binary == 1, axis=1)
        
        unique_elements, counts = np.unique(patches_predictions_categorial, return_counts=True)
        total_urchin = 0
        if 5 in unique_elements:
            # index 5 is for urchin, should not be considered for the percentage computation
            index = np.where(unique_elements == 5)[0][0]
            total_urchin = counts[index]
            unique_elements = np.delete(unique_elements, index)
            counts = np.delete(counts, index)

        total_count = len(patches_predictions_categorial)-total_urchin
        percentages = (counts / total_count) * 100
        prediction_results = dict()
        for i, e in enumerate(unique_elements):
            prediction_results.update({class_names[e]: percentages[i]})
        whole_frame_label = None
        if 'Unconsolidated' in prediction_results.keys():
            if prediction_results['Unconsolidated']>=50:
                if whole_frame_label==None:
                    whole_frame_label = 'Unconsolidated'
        if 'Ecklonia' in prediction_results.keys():
            if prediction_results['Ecklonia']>=50:
                if whole_frame_label==None:
                    if ('Grazed rock' in prediction_results.keys() and prediction_results['Grazed rock']>25) or total_urchin!=0: 
                        ########################################################################################################
                        whole_frame_label = None
                    else:
                        whole_frame_label = "Reef-kelp"

        if 'Grazed rock' in prediction_results.keys():
            if prediction_results['Grazed rock']>=50 and total_urchin!=0:
                if whole_frame_label==None:
                    whole_frame_label = 'Reef-Barren'
            if prediction_results['Grazed rock']>25 and prediction_results['Grazed rock']<50 and total_urchin!=0:
                if whole_frame_label==None:
                    whole_frame_label = "Reef-Partial-Barren"
        
        Others = ['Carpophyllum spp','Ecklonia','Foliose Algae','Other canopy forming macroalgae']
        Others_percentages = 0
        for label in prediction_results.keys():
            if label in Others:
                Others_percentages+=prediction_results[label]
                
        if Others_percentages >50: 
            if total_urchin!=0:
                if whole_frame_label==None:
                    whole_frame_label = "Reef-Vegetated"
            elif 'Grazed rock' in prediction_results.keys():
                max_other_percentage = 0
                for label in prediction_results.keys():
                    if label in Others:
                        if prediction_results[label]>=max_other_percentage:
                            max_other_percentage = prediction_results[label]
                if prediction_results['Grazed rock']>max_other_percentage:
                    if whole_frame_label==None:
                        #################################################################################
                        whole_frame_label = None
    
        Others = ['Carpophyllum spp','Ecklonia','Foliose Algae','Other canopy forming macroalgae', 'Grazed rock']
        Others_percentages = 0
        for label in prediction_results.keys():
            if label in Others:
                Others_percentages+=prediction_results[label]
        
        if Others_percentages >50 and total_urchin==0:
            if whole_frame_label==None:
                whole_frame_label = "Reef-Vegetated"
        elif whole_frame_label==None:
            #################################################################################
            whole_frame_label = None
        label_pred_list.append(whole_frame_label)
    return label_pred_list

def get_classification_report(label_pred_list, label_truth_list, image_list):
    y_preds, y_truths,imgs = zip(*[(a, b,c) for a, b,c in zip(label_pred_list, label_truth_list, image_list) if a is not None])
    count_none = 0
    for c in label_pred_list:
        if c==None:
            count_none+=1
    print(count_none)
    y_preds = np.asarray(list(y_preds))
    y_truths = np.asarray(list(y_truths))
    imgs = np.asarray(list(imgs))
    report = classification_report(y_truths, y_preds, zero_division=0, output_dict=True)
    report_data = {cls: {'precision': report[cls]['precision'], 
                        'recall': report[cls]['recall'], 
                        'f1-score': report[cls]['f1-score']} for cls in list(np.unique(y_truths))}
    fsize=14
    report_df = pd.DataFrame(report_data).transpose()
    plt.figure(figsize=(10, 5))
    sns.heatmap(report_df, annot=True, cmap="PuRd", fmt='.2f', cbar=True, annot_kws={"size": fsize-2})
    plt.title('Classification Report on the whole-frame dataset', fontsize =fsize)
    plt.xlabel('Metrics', fontsize =fsize)
    plt.ylabel('Classes', fontsize =fsize)
    plt.xticks(fontsize=fsize-3)
    plt.yticks(fontsize=fsize-3)
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis", "classification_report_whole_frame.png"))
    plt.close()

def get_confusion_matrix_plot (label_pred_list, label_truth_list, image_list):
    y_preds, y_truths,imgs = zip(*[(a, b,c) for a, b,c in zip(label_pred_list, label_truth_list, image_list) if a is not None])
    count_none = 0
    for c in label_pred_list:
        if c==None:
            count_none+=1
    print(count_none)

    y_preds = np.asarray(list(y_preds))
    y_truths = np.asarray(list(y_truths))
    imgs = np.asarray(list(imgs))
    cm = confusion_matrix(y_truths, y_preds)
    cm_df = pd.DataFrame(cm, index = list(np.unique(y_truths)), columns = list(np.unique(y_truths)))
    fsize = 14
    plt.figure(figsize=(10,8))
    ax =sns.heatmap(cm_df, annot = True, fmt='d', cmap='Purples',annot_kws={"size": fsize, "weight": "light"})

    plt.title('Confusion Matrix for the whole-frame dataset', fontsize=fsize)

    plt.ylabel('Actual Values', fontsize=fsize)
    plt.xlabel('Predicted Values', fontsize=fsize)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fsize)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis", "confusion_matrix_plot_whole_frame.png"))
    plt.close()

    
if __name__ == '__main__':
    actual_class_names = {"reef_barren": 'Reef-Barren',
                      "reef_kelp": "Reef-kelp",
                      "reef_partial_barren": "Reef-Partial-Barren", 
                      "reef_vegetated": "Reef-Vegetated",
                      "unconsolidated": 'Unconsolidated'}
    plot_original_dataset()
    model = load_model(os.path.join(".","saved_models","final_softmax", "20240827_020806" ))
    image_list = []
    label_truth_list = []
    frame_dataset_path = os.path.join(".", "dataset", "grouped_frames_with_whole_annotations")
    if not os.path.exists(frame_dataset_path):
       print("Plase first download frames_with_whole_annotations using download_frames_with_whole_annotations.py")
       print("Next, run grouped_frames_with_whole_annotations.py")
    else:
        class_paths = glob.glob(os.path.join(frame_dataset_path, "*"))
        for class_path in class_paths:
            class_name = os.path.basename(class_path)
            for path in glob.glob(os.path.join(class_path, "*")):
                image_list.append(path)
                label_truth_list.append(actual_class_names[class_name])
        label_pred_list = []
        num_points = 25
        for i in tqdm(range(len(image_list)),desc="Processing"):
            image=[image_list[i]]
            patches_for_all_images_logits = get_model_predictions (model,image, num_points, crop_percentage=0.2, point_generation = "random",batch_size=num_points, verbose = 0)
            label_list = majvote_on_img_predictions(patches_for_all_images_logits,num_images = len(image),num_points= num_points)
            label_pred_list.append(label_list[0])

        get_classification_report(label_pred_list, label_truth_list, image_list)






