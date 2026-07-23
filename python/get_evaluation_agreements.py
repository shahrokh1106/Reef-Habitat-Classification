
import pandas as pd 
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter


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

    # initi_agreements = df['Correct Model Initial Prediction'].to_numpy()
    # initial_recommendations_by_experts = df['Unnamed: 7'].to_numpy()

    # nan_mask = pd.isna(initi_agreements)
    # initial_preds = []
    # initial_no_agreements_num = 0
    # for index, l in enumerate(nan_mask):
    #     if not l:
    #         if initi_agreements[index]=="No agreement":
    #             initial_no_agreements_num+=1
    #         else:
    #             if initi_agreements[index]:
    #                 initial_preds.append(1)
    #             else:
    #                 initial_preds.append(0)
    #     else:
    #         initial_preds.append(1)
    # print(len(initial_preds))
    # print(initial_no_agreements_num)
    # print(len(preds))


   
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

    
    print(Counter(labels))
    cls_report = get_classification_report_plot(new_preds, labels,"", model_name = "final_dataset")
    print("accuracy: ", cls_report["accuracy"])
    print("macro avg: ", cls_report["macro avg"])
    print('weighted avg: ', cls_report['weighted avg'])
    get_confusion_matrix_plot (new_preds, labels, "", model_name = "final_dataset")
    label_counts = Counter(labels)

    # Plot
    plt.figure(figsize=(18, 12))
    bars = plt.barh(list(label_counts.keys()), list(label_counts.values()))
    plt.ylabel('Class Names', fontsize=25)
    plt.xlabel('Frequency', fontsize=20)
    plt.title('Label Frequency', fontsize=20)

    plt.tick_params(axis='y', labelsize=20)  # y-axis = class names
    plt.tick_params(axis='x', labelsize=12)  # x-axis = frequencies

    # Add numbers to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, str(width),
                va='center', fontsize=20)

    plt.tight_layout()
    plt.savefig("frequency.png")








    