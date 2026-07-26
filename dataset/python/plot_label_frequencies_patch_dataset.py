import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

def get_frequency_dict (dataset_path):
    frequency_dict = {}
    classes_paths = glob.glob(os.path.join(dataset_path, "*"))
    for path in classes_paths:
        class_name = os.path.basename(path)
        count=  len(glob.glob(os.path.join(path, "*")))
        frequency_dict.update({class_name:count})
    return frequency_dict

def combine_classes(sample_dict):
    # Class combination based on experts' suggestions, same combinations are used for training
    new_dict = dict()
    new_dict.update({
        'BrLfa':sample_dict['Bare rock']+sample_dict['Turf']+sample_dict['Encrusting algae']+sample_dict['Filamentous algae'],
        'Urchin':sample_dict['Urchin'],
        'Unconsolidated':sample_dict['Unconsolidated'],
        'Other canopy forming macroalgae':sample_dict['Other canopy forming macroalgae'],
        'Foliose algae':sample_dict['Foliose Algae'],
        'Ecklonia radiata':sample_dict['Ecklonia radiata'],
        'Carpophyllum spp':sample_dict['Carpophyllum spp']
    })
    return  new_dict


def plot_original_dataset(dataset_path,fontsize=20, fig_name = "frequency"):
    fontsize = 30
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "valid")
    test_path = os.path.join(dataset_path, "test")
    train_dict = get_frequency_dict (train_path)
    val_dict = get_frequency_dict (val_path)
    test_dict = get_frequency_dict (test_path)
    dataset_dict = {}
    for key in train_dict.keys():
        dataset_dict.update({key:train_dict[key]+val_dict[key]+test_dict[key]})

    plt.style.use('default') 
    plt.rcParams['font.family'] = 'serif'  
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    # Plotting the dictionary with horizontal bars
    plt.figure(figsize=(20, 8))
    bars = plt.barh(list(dataset_dict.keys()), list(dataset_dict.values()))
    plt.ylabel('Class names',fontsize=fontsize)
    plt.gca().set_yticklabels(
        plt.gca().get_yticklabels(),
        fontstyle='italic'
        )
    plt.xlabel('Frequency',fontsize=fontsize)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    for bar in bars:
        plt.text(bar.get_width() + 0.1, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center',  
                fontsize=fontsize-10)  
    plt.title('Class frequency in the original patch dataset', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis",fig_name+".png"))
    plt.close()
    
def plot_original_dataset_combined(dataset_path, fontsize = 20, fig_name = "frequency"):
    
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "valid")
    test_path = os.path.join(dataset_path, "test")
    train_dict = get_frequency_dict (train_path)
    val_dict = get_frequency_dict (val_path)
    test_dict = get_frequency_dict (test_path)
    dataset_dict = {}
    for key in train_dict.keys():
        dataset_dict.update({key:train_dict[key]+val_dict[key]+test_dict[key]})

    dataset_dict_ = combine_classes(dataset_dict)
    dataset_dict_new = {}
    for key,value in dataset_dict_.items():
        if key=='Other canopy forming macroalgae':
            key = 'Other canopy'
        if key=="Foliose Algae":
            key = "Foliose algae"
        dataset_dict_new.update({key:value})
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = sns.color_palette('deep', 8)
    
    plt.style.use('default') 
    # sns.set_style("white") 
    plt.rcParams['font.family'] = 'serif'  # You can change 'serif' to 'sans-serif', 'monospace', etc.
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    # Plotting the dictionary with horizontal bars
    plt.figure(figsize=(17, 15))
    bars = plt.barh(list(dataset_dict_new.keys()), list(dataset_dict_new.values()),color=colors,edgecolor='black', linewidth=0.5)
    plt.ylabel('Class names',fontsize=fontsize)
    # make y tick labels italic
    plt.gca().set_yticklabels(
        plt.gca().get_yticklabels(),
        fontstyle='italic'
        )
    plt.xlabel('Frequency',fontsize=fontsize)
    plt.title('Class frequency in the refined patch dataset', fontsize=fontsize,pad=20)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    for bar in bars:
        plt.text(400, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center',  
                fontsize=fontsize)  

    # total = sum(dataset_dict_new.values())
    # sizes = [value / total * 100 for value in dataset_dict_new.values()]
    # inset_axes = plt.axes([0.71, 0.35, 0.25, 0.4])  #[left, bottom, width, height] 
    # # Plot the pie chart
    # inset_axes.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=100,pctdistance=1.25, textprops={'fontsize': fontsize-5})
    # inset_axes.axis('equal')  
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis", fig_name+".png"))
    plt.close()


def plot_traning_dataset_combined(dataset_path,fontsize=20,fig_name="frequency"):
    from matplotlib.patches import Patch

    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "valid")
    test_path = os.path.join(dataset_path, "test")
    train_dict = get_frequency_dict (train_path)
    valid_dict = get_frequency_dict (val_path)
    test_dict = get_frequency_dict (test_path)
    train_dict_new = combine_classes(train_dict)
    valid_dict_new = combine_classes(valid_dict)
    test_dict_new = combine_classes(test_dict)
    keys = list(train_dict_new.keys())
    values1 = list(train_dict_new.values())
    values2 = list(valid_dict_new.values())
    values3 = list(test_dict_new.values())
    positions = np.arange(len(keys))
    
    bar_height = 0.25
    positions = np.arange(len(keys))
    plt.figure(figsize=(17, 15))
    bars1 = plt.barh(positions - bar_height, values1, height=bar_height, label='Train set',edgecolor='black', linewidth=0.5)
    bars2 = plt.barh(positions, values2, height=bar_height, label='Valid set',edgecolor='black', linewidth=0.5)
    bars3 = plt.barh(positions + bar_height, values3, height=bar_height, label='Test set',edgecolor='black', linewidth=0.5)
    plt.ylabel('Classes', fontsize=fontsize)
    plt.xlabel('Frequency', fontsize=fontsize)
    plt.title('Class frequency in the train, valid, and test sets (for the patch dataset)', fontsize=fontsize, pad = 25)
    plt.yticks(positions, keys, fontsize=fontsize-2)  
    plt.xticks(fontsize=fontsize-2)  
    plt.legend(fontsize =fontsize)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = sns.color_palette('deep', 8)
    plt.yticks(positions, [''] * len(keys))
    
    for y, color in zip(positions, colors):
        plt.scatter(-max(values1) * 0.04, y, color=color, s=3000, marker='s', edgecolor='black', linewidth=0.5)  

    for bar in bars1:
        plt.text(bar.get_width() + 100, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center',  
                fontsize=fontsize)  
    for bar in bars2:
        plt.text(bar.get_width() + 100, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center',  
                fontsize=fontsize)  
    for bar in bars3:
        plt.text(bar.get_width() + 100, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center', 
                fontsize=fontsize)  
    plt.xlim(-3000, 50000)
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis", fig_name+".png"))
    plt.close()


def plot_test_extra(dataset_path,fontsize =20, fig_name = "frequency"):
    test_ex_path = os.path.join(dataset_path, "test_extra")
    test_ex_dict = get_frequency_dict (test_ex_path)
    test_ex_dict_combined = combine_classes(test_ex_dict)

    plt.style.use('default') 
    plt.rcParams['font.family'] = 'serif'  
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    # Plotting the dictionary with horizontal bars
    plt.figure(figsize=(15, 8))
    bars = plt.barh(list(test_ex_dict.keys()), list(test_ex_dict.values()))
    plt.ylabel('Class names',fontsize=fontsize)
    plt.xlabel('Frequency',fontsize=fontsize)
    plt.title('Class frequency in the original dataset',fontsize=fontsize)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    for bar in bars:
        plt.text(bar.get_width() + 0.1, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center',  
                fontsize=fontsize-5)  
    plt.title('Class frequency in the test_extra dataset', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis",  fig_name+"_conmbined"+".png"))
    plt.close()


    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = sns.color_palette('deep', 8)
    plt.style.use('default') 
    # sns.set_style("white") 
    plt.rcParams['font.family'] = 'serif'  # You can change 'serif' to 'sans-serif', 'monospace', etc.
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    # Plotting the dictionary with horizontal bars
    plt.figure(figsize=(16, 8))
    bars = plt.barh(list(test_ex_dict_combined.keys()), list(test_ex_dict_combined.values()),color=colors)
    plt.ylabel('Class names',fontsize=fontsize)
    plt.xlabel('Frequency',fontsize=fontsize)
    plt.title('Class frequency in the original dataset (combined classes)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    for bar in bars:
        plt.text(bar.get_width() + 0.1, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center',  
                fontsize=fontsize-3)  

    total = sum(test_ex_dict_combined.values())
    sizes = [value / total * 100 for value in test_ex_dict_combined.values()]
    inset_axes = plt.axes([0.71, 0.35, 0.25, 0.4])  #[left, bottom, width, height] 
    # Plot the pie chart
    inset_axes.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=100,pctdistance=1.2, textprops={'fontsize': fontsize-4})
    inset_axes.axis('equal')  
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis", fig_name+"_conmbined"+".png"))
    plt.close()



if __name__ == '__main__':
    fontsize = 40
    patch_dataset_path = os.path.join("dataset","patches")
    if not os.path.exists(patch_dataset_path):
        print("The image patches have not been created yet. Please run get_patches.py and make sure you first download the frames")
        raise Exception ("patches not found")
    # Plotting for patch dataset
    plot_original_dataset(patch_dataset_path,fontsize=fontsize,fig_name="patch_dataset_label_frequency_org_all")
    plot_original_dataset_combined(patch_dataset_path,fontsize=fontsize,fig_name="patch_dataset_label_frequency_org_all_combined")
    plot_traning_dataset_combined(patch_dataset_path,fontsize=fontsize,fig_name="patch_dataset_label_frequency_splits_combined")
    plot_test_extra(patch_dataset_path,fontsize=fontsize, fig_name = "patch_dataset_label_frequency_test_extra")