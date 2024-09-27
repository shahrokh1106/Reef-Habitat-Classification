import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_original_dataset(train_df,valid_df,test_df):
    all_data = pd.concat([train_df, valid_df, test_df])
    dataset_dict = all_data['label.translated.name'].value_counts().to_dict()
    fsize = 20
    plt.style.use('default') 
    plt.rcParams['font.family'] = 'serif'  
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    # Plotting the dictionary with horizontal bars
    plt.figure(figsize=(15, 8))
    bars = plt.barh(list(dataset_dict.keys()), list(dataset_dict.values()))
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
    plt.title('Class frequency in the the original dataset (excluding test_extra)', fontsize=fsize)
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis", "label_frequency_org_all.png"))
    plt.close()

def combine_classes(sample_dict):
    # Class combination based on experts' suggestions, same combinations are used for training
    new_dict = dict()
    new_dict.update({
        'Grazed Rock':sample_dict['Bare rock']+sample_dict['Turf']+sample_dict['Encrusting algae']+sample_dict['Filamentous algae'],
        'Urchin':sample_dict['Urchin'],
        'Unconsolidated':sample_dict['Unconsolidated'],
        'Other canopy forming macroalgae':sample_dict['Other canopy forming macroalgae'],
        'Foliose Algae':sample_dict['Foliose Algae'],
        'Ecklonia radiata':sample_dict['Ecklonia radiata'],
        'Carpophyllum spp':sample_dict['Carpophyllum spp.']
    })
    return  new_dict

def plot_original_dataset_combined(train_df,valid_df,test_df):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = sns.color_palette('deep', 8)
    all_data = pd.concat([train_df, valid_df, test_df])
    dataset_dict = all_data['label.translated.name'].value_counts().to_dict()
    dataset_dict_new = combine_classes(dataset_dict)
    fsize = 18
    plt.style.use('default') 
    # sns.set_style("white") 
    plt.rcParams['font.family'] = 'serif'  # You can change 'serif' to 'sans-serif', 'monospace', etc.
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    # Plotting the dictionary with horizontal bars
    plt.figure(figsize=(16, 8))
    bars = plt.barh(list(dataset_dict_new.keys()), list(dataset_dict_new.values()),color=colors)
    plt.ylabel('Class names',fontsize=fsize)
    plt.xlabel('Frequency',fontsize=fsize)
    plt.title('Class frequency in the original dataset (combined classes and excluding test_extra)', fontsize=fsize)
    plt.xticks(fontsize=fsize-3)
    plt.yticks(fontsize=fsize-3)
    for bar in bars:
        plt.text(bar.get_width() + 0.1, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center',  
                fontsize=fsize-3)  

    total = sum(dataset_dict_new.values())
    sizes = [value / total * 100 for value in dataset_dict_new.values()]
    inset_axes = plt.axes([0.71, 0.35, 0.25, 0.4])  #[left, bottom, width, height] 
    # Plot the pie chart
    inset_axes.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=100,pctdistance=1.2, textprops={'fontsize': fsize-4})
    inset_axes.axis('equal')  
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis", "label_frequency_org_all_combined.png"))
    plt.close()



def plot_traning_dataset_combined(train_df,valid_df,test_df):
    fsize = 18
    train_dict = train_df['label.translated.name'].value_counts().to_dict()
    valid_dict = valid_df['label.translated.name'].value_counts().to_dict()
    test_dict = test_df['label.translated.name'].value_counts().to_dict()
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
    plt.figure(figsize=(14, 6))
    bars1 = plt.barh(positions - bar_height, values1, height=bar_height, label='Train set')
    bars2 = plt.barh(positions, values2, height=bar_height, label='Valid set')
    bars3 = plt.barh(positions + bar_height, values3, height=bar_height, label='Test set')
    plt.ylabel('Class names', fontsize=fsize)
    plt.xlabel('Frequency', fontsize=fsize)
    plt.title('Class frequency in the train, valid, and test sets (excluding test_extra)', fontsize=fsize)
    plt.yticks(positions, keys, fontsize=fsize-2)  
    plt.xticks(fontsize=fsize-2)  
    plt.legend()

    for bar in bars1:
        plt.text(bar.get_width() + 0.1, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center',  
                color=bar.get_facecolor(),
                fontsize=fsize-2)  
    color_bar_1 = bar.get_facecolor()
    for bar in bars2:
        plt.text(bar.get_width() + 0.1, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center',  
                color=bar.get_facecolor(),
                fontsize=fsize-2)  
    color_bar_2 = bar.get_facecolor()
    for bar in bars3:
        plt.text(bar.get_width() + 0.1, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center', 
                color=bar.get_facecolor(),
                fontsize=fsize-2)  
    color_bar_3 = bar.get_facecolor()
    inset_axes = plt.axes([0.78, 0.3, 0.2, 0.36])  # [left, bottom, width, height] in normalized units
    sizes = [70, 15, 15]
    colors = [color_bar_1, color_bar_2, color_bar_3]
    inset_axes.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=56)
    inset_axes.axis('equal') 
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis", "label_frequency_traning_dataset_combined.png"))
    plt.close()

def plot_test_extra(test_ex_df):
    test_ex_dict = test_ex_df['label.translated.name'].value_counts().to_dict()
    test_ex_dict_new = combine_classes(test_ex_dict)
    fsize = 20
    plt.style.use('default') 
    plt.rcParams['font.family'] = 'serif'  
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    # Plotting the dictionary with horizontal bars
    plt.figure(figsize=(15, 8))
    bars = plt.barh(list(test_ex_dict.keys()), list(test_ex_dict.values()))
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
    plt.title('Class frequency in the test_extra dataset', fontsize=fsize)
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis", "label_frequency_test_extra.png"))
    plt.close()


    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = sns.color_palette('deep', 8)
    fsize = 18
    plt.style.use('default') 
    # sns.set_style("white") 
    plt.rcParams['font.family'] = 'serif'  # You can change 'serif' to 'sans-serif', 'monospace', etc.
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    # Plotting the dictionary with horizontal bars
    plt.figure(figsize=(16, 8))
    bars = plt.barh(list(test_ex_dict_new.keys()), list(test_ex_dict_new.values()),color=colors)
    plt.ylabel('Class names',fontsize=fsize)
    plt.xlabel('Frequency',fontsize=fsize)
    plt.title('Class frequency in the original dataset (combined classes and excluding test_extra)', fontsize=fsize)
    plt.xticks(fontsize=fsize-3)
    plt.yticks(fontsize=fsize-3)
    for bar in bars:
        plt.text(bar.get_width() + 0.1, 
                bar.get_y() + bar.get_height()/2,  
                f'{bar.get_width()}',  
                va='center',  
                fontsize=fsize-3)  

    total = sum(test_ex_dict_new.values())
    sizes = [value / total * 100 for value in test_ex_dict_new.values()]
    inset_axes = plt.axes([0.71, 0.35, 0.25, 0.4])  #[left, bottom, width, height] 
    # Plot the pie chart
    inset_axes.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=100,pctdistance=1.2, textprops={'fontsize': fsize-4})
    inset_axes.axis('equal')  
    plt.tight_layout()
    plt.savefig(os.path.join("dataset", "figs_dataset_analysis", "label_frequency_test_extra_combined_combined.png"))
    plt.close()


if __name__ == '__main__':
    if os.path.exists(os.path.join("dataset","dataset_csv_files","training_datasets")):
        train_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","train_df.csv"),low_memory=False)
        valid_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","valid_df.csv"),low_memory=False)
        test_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","test_df.csv"),low_memory=False)
        test_ex_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","test_ex_df.csv"),low_memory=False)
        plot_original_dataset(train_df,valid_df,test_df)
        plot_original_dataset_combined(train_df,valid_df,test_df)
        plot_traning_dataset_combined(train_df,valid_df,test_df)
        plot_test_extra(test_ex_df)
        print("all saved in dataset/figs_dataset_analysis/")
        print("The frequencies are based on the csv files and information, not downloadable and useable patches")
    else:
        print("please first run split_dataset_csv to get the training csv files")
