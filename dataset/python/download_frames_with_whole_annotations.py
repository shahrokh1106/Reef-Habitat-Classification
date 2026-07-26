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
    frame_dataset_1= pd.read_csv(os.path.join("dataset","dataset_csv_files","csvs_whole_frame_annotations","annotations-u1683-East_Coast_Tasmania_Habitat_Classification_-East_Coast_Tasmania_Habitat_-_whole_frame-15897-828424b4218478114966-dataframe.csv"),low_memory=False)
    frame_dataset_2= pd.read_csv(os.path.join("dataset","dataset_csv_files","csvs_whole_frame_annotations","annotations-u1576-Habitat_Classification_UoA-Habitat_classfication-15881-828424b4218478114966-dataframe.csv"),low_memory=False)
    frame_dataset_3= pd.read_csv(os.path.join("dataset","dataset_csv_files","csvs_whole_frame_annotations","annotations-u1576-RLS_Habitat_Classification-RLS_Habitat_Classification-15896-828424b4218478114966-dataframe.csv"),low_memory=False)
    dataset_final = pd.concat([frame_dataset_1, frame_dataset_2, frame_dataset_3], ignore_index=True)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    image_dir = os.path.join("dataset","frames_with_whole_annotations")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_dir = os.path.join("dataset","frames_with_whole_annotations/")
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
    