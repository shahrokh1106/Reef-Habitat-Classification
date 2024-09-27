
import urllib.error
import urllib.request
import pandas as pd
import os
from time import time
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

def download_image(url, filename):
    try:
        urllib.request.urlretrieve(url, filename)
        return 0
    except:
        return os.path.basename(filename).split('.')[0]

if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","train_df.csv"),low_memory=False)
    valid_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","valid_df.csv"),low_memory=False)
    test_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","test_df.csv"),low_memory=False)
    test_ex_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","test_ex_df.csv"),low_memory=False)
    dataset_final = pd.concat([train_df, valid_df, test_df, test_ex_df])
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    if not os.path.exists(os.path.join("dataset","frames")):
        os.makedirs(os.path.join("dataset","frames"))
    image_dir = os.path.join("dataset","frames/")
    df_download = dataset_final.drop_duplicates(subset="point.media.path_best", keep="first")
    image_urls = df_download["point.media.path_best"]
    df_download['point.media.key'] = df_download['point.media.key'].apply(lambda x: os.path.basename(x))
    image_names = df_download["point.media.key"]
    file_names= image_dir+df_download["point.media.id"].astype(str)+"_"+image_names
    file_names[~file_names.str.endswith(valid_extensions)]= file_names[~file_names.str.endswith(valid_extensions)]+".JPG"
    start = time()
    print("\nDownloading...")
    num_cores = multiprocessing.cpu_count()
    ko_list = Parallel(n_jobs=num_cores)(delayed(download_image)(u, f) for f, u in tqdm(zip(file_names, image_urls)))
    print("\nDownload in parallel mode took %d seconds." %(time()-start))
    print("Success:", len([i for i in ko_list if i==0]))
    print("Errors:", len([i for i in ko_list if i!=0]))