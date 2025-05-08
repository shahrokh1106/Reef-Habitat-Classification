import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from python.load_csvs import *
from sklearn.model_selection import train_test_split

def main():
    dataset_final_org = load_final_dataset()
    # Getting completely unseen subdatasets for evaluation purposes
    extra_test_dataset_names = [
        "RLS_Elizabeth and Middleton Reefs_2018",
        "RLS_Rocky Cape_2019", 
        "RLS_Victoria (other)_2014",
        "2019-CoffsHarbour",
        "RLS_New South Wales (Other)_201",
        "RLS_Sydney_2008",
        "RLS_Port Stephens_2018",
        "RLS_Port Stephens_2021",
        "202210_Nordic"]
    test_ex_df = dataset_final_org[dataset_final_org['point.media.deployment.campaign.key'].isin(extra_test_dataset_names)]
    dataset_final_filtered = dataset_final_org[~dataset_final_org['point.media.deployment.campaign.key'].isin(extra_test_dataset_names)]
    train_df, valid_df = train_test_split(dataset_final_filtered, test_size=0.3, random_state=123, stratify=dataset_final_filtered['point.media.deployment.campaign.key'])
    valid_df,test_df = train_test_split(valid_df, test_size=0.5, random_state=123, stratify=valid_df['label.translated.name'])

    out_path = os.path.join("dataset","dataset_csv_files","training_datasets")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    train_df.to_csv(os.path.join(out_path,'train_df.csv'), index=False)
    valid_df.to_csv(os.path.join(out_path,'valid_df.csv'), index=False)
    test_df.to_csv(os.path.join(out_path,'test_df.csv'), index=False)
    test_ex_df.to_csv(os.path.join(out_path,'test_ex_df.csv'), index=False)

if __name__ == '__main__':
    main()
    print("saved in dataset/figs_dataset_analysis")
    print("The shown frequency is based on the csv files not the downloadable patches")