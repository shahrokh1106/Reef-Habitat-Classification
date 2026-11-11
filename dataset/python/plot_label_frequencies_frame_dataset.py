import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
from utilities import ACTUAL_CLASS_NAMES
def get_frequency_dict (dataset_path):
    frequency_dict = {}
    classes_paths = glob.glob(os.path.join(dataset_path, "*"))
    for path in classes_paths:
        class_name = os.path.basename(path)
        if class_name =="reef_partial_barren" or class_name=="reef_partial_grazed":
            continue

        count= 0
        for path in glob.glob(os.path.join(path, "*")):
            if "augment" in path:
                continue
            else:
                count+=1
        frequency_dict.update({ACTUAL_CLASS_NAMES[class_name]:count})
    return frequency_dict


def plot_original_dataset(dataset_path,fontsize=20, fig_name = "frequency"):
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")
    train_dict = get_frequency_dict (train_path)
    val_dict = get_frequency_dict (val_path)
    test_dict = get_frequency_dict (test_path)
    dataset_dict = {}
    for key in train_dict.keys():
        dataset_dict.update({key:train_dict[key]+val_dict[key]+test_dict[key]})

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = sns.color_palette('deep', 8)
    colors = [colors[1], colors[0],colors[5],colors[6], colors[2]]

    plt.style.use('default') 
    plt.rcParams['font.family'] = 'serif'  
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    # Plotting the dictionary with horizontal bars
    plt.figure(figsize=(17, 15))
    bars = plt.barh(list(dataset_dict.keys()), list(dataset_dict.values()), color = colors,edgecolor='black', linewidth=0.5)
    plt.ylabel('Class names',fontsize=fontsize)
    plt.gca().set_yticklabels(
        plt.gca().get_yticklabels(),
        fontstyle='italic'
        )
    plt.xlabel('Frequency',fontsize=fontsize)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize+3)
    plt.title('Class frequency in the frame dataset', fontsize=fontsize,pad=25)
    for bar in bars:
        plt.text(40, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center',  
                fontsize=fontsize)  
    # total = sum(dataset_dict.values())
    # sizes = [value / total * 100 for value in dataset_dict.values()]
    # inset_axes = plt.axes([0.71, 0.35, 0.25, 0.4])  #[left, bottom, width, height] 
    # # Plot the pie chart
    # inset_axes.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=100,pctdistance=1.25, textprops={'fontsize': fontsize-5})
    # inset_axes.axis('equal') 
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis",fig_name+".png"))
    plt.close()





def plot_traning_dataset_combined(dataset_path,fontsize=20,fig_name="frequency"):
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")
    train_dict = get_frequency_dict (train_path)
    val_dict = get_frequency_dict (val_path)
    test_dict = get_frequency_dict (test_path)
    keys = list(train_dict.keys())
    values1 = list(train_dict.values())
    values2 = list(val_dict.values())
    values3 = list(test_dict.values())
    positions = np.arange(len(keys))
    bar_height = 0.25
    positions = np.arange(len(keys))
    plt.figure(figsize=(17, 15))

    bars1 = plt.barh(positions - bar_height, values1, height=bar_height, label='Train set',edgecolor='black', linewidth=0.5)
    bars2 = plt.barh(positions, values2, height=bar_height, label='Valid set',edgecolor='black', linewidth=0.5)
    bars3 = plt.barh(positions + bar_height, values3, height=bar_height, label='Test set',edgecolor='black', linewidth=0.5)
    plt.ylabel('Classes', fontsize=fontsize)
    plt.xlabel('Frequency', fontsize=fontsize)
    plt.title('Class frequency in the train, valid, and test sets (for the frame dataset)', fontsize=fontsize,pad=25)
    plt.yticks(positions, keys, fontsize=fontsize-2)  
    plt.xticks(fontsize=fontsize-2)  
    plt.legend(fontsize =fontsize)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = sns.color_palette('deep', 8)
    colors = [colors[1], colors[0],colors[5],colors[6], colors[2]]
    plt.yticks(positions, [''] * len(keys))
    for y, color in zip(positions, colors):
        plt.scatter(-max(values1) * 0.04, y, color=color, s=3000, marker='s', edgecolor='black', linewidth=0.5)  

    for bar in bars1:
        plt.text(bar.get_width() + 10, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center',  
                fontsize=fontsize)  
    for bar in bars2:
        plt.text(bar.get_width() + 10, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center',  
                fontsize=fontsize)  
    for bar in bars3:
        plt.text(bar.get_width() + 10, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center', 
                fontsize=fontsize)  
    plt.xlim(-150, 2100)
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis", fig_name+".png"))
    plt.close()


def save_figures_horizontally(fig_names, fig_name="frequency_side_by_side", gap=50):
    from PIL import Image

    figs_dir = os.path.join("dataset", "figs_dataset_analysis")
    images = [Image.open(os.path.join(figs_dir, name+".png")).convert("RGB") for name in fig_names]
    total_width = sum(img.width for img in images) + gap * (len(images) - 1)
    max_height = max(img.height for img in images)
    canvas = Image.new("RGB", (total_width, max_height), "white")
    x_offset = 0
    for img in images:
        canvas.paste(img, (x_offset, (max_height - img.height) // 2))
        x_offset += img.width + gap
    canvas.save(os.path.join(figs_dir, fig_name+".png"))


if __name__ == '__main__':
    fontsize = 40
    framem_dataset_path = os.path.join("dataset","frame7_dataset_cleaned")
    if not os.path.exists(framem_dataset_path):
        print("The frames for the frame classifier have not been downloaded. Please download the images first")
        raise Exception ("Frames not found")
    plot_original_dataset(framem_dataset_path,fontsize=fontsize,fig_name="frame_dataset_label_frequency")
    plot_traning_dataset_combined(framem_dataset_path,fontsize=fontsize,fig_name="frame_dataset_label_frequency_splits")
    save_figures_horizontally(
        ["frame_dataset_label_frequency", "frame_dataset_label_frequency_splits"],
        fig_name="frame_dataset_label_frequency_and_splits")
    