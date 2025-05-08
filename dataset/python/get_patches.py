import pandas as pd
import os
import cv2

def crop_around_point (img,x,y,crop_size):
    xx, yy = crop_size//2,crop_size//2
    if x-(crop_size//2)<0 or y-(crop_size//2)<0:
        if x-(crop_size//2):
            xx = min(x,y)
        if y-(crop_size//2)<0:
            yy = min(x,y)
    if x+(crop_size//2)>=img.shape[1] or y+(crop_size//2)>=img.shape[0]: 
        if x+(crop_size//2)>=img.shape[1]:
            xx = min(img.shape[1]-x,img.shape[0]-y)
        if y+(crop_size//2)>=img.shape[0]: 
            yy = min(img.shape[1]-x,img.shape[0]-y)
        
    xx,yy = min(xx,yy), min(xx,yy)
    x1 = x - xx
    y1 = y - yy
    x2 = x + xx
    y2 = y + yy
    cropped_image = img[y1:y2, x1:x2].copy()
    # cropped_image = cv2.resize(cropped_image,(crop_size,crop_size))
    return cropped_image

if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","train_df.csv"),low_memory=False)
    valid_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","valid_df.csv"),low_memory=False)
    test_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","test_df.csv"),low_memory=False)
    test_ex_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","test_ex_df.csv"),low_memory=False)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    dataset_names = ["train", "valid", "test"]
    images_path = os.path.join("dataset","frames")
    crop_percentage = 0.20

    train_df['id'] = train_df.index
    valid_df['id'] = valid_df.index
    test_df['id'] = test_df.index
    test_ex_df['id'] = test_ex_df.index

    dataset_names = ["test_extra"]
    for dataset_name in dataset_names:
        save_patches_path = os.path.join("dataset", "patches", dataset_name)
        if not os.path.exists(save_patches_path):
            os.makedirs(save_patches_path)
        if dataset_name == "train":
            df = train_df.copy()
        elif dataset_name == "valid":
            df = valid_df.copy()
        elif dataset_name == "test":
            df = test_df.copy()  
        elif dataset_name == "test_extra":
            df = test_ex_df.copy()     
        df['point.media.key'] = df['point.media.key'].apply(lambda x: os.path.basename(x))
        image_names = df["point.media.key"]
        image_names[~image_names.str.endswith(valid_extensions)]= image_names[~image_names.str.endswith(valid_extensions)]+".JPG"
        image_names = list(image_names)
        ids = list(df["id"])
        point_xs = list(df["point.x"])
        point_ys = list(df["point.y"])
        media_ids = df["point.media.id"].astype(str)
        classes = df["label.translated.name"].astype(str)
        non_existing_images = []
        for id, media_id, image_name, x, y, c in zip(ids,media_ids, image_names,point_xs,point_ys, classes):
            try: 
                img = cv2.imread(os.path.join(images_path, media_id+'_'+image_name))
                w,h = img.shape[1], img.shape[0]
                x = int(x*w)
                y = int(y*h)
                crop_size = int(((w+h)/2)*crop_percentage)
                cropped_img = crop_around_point (img,x,y,crop_size)
                if cropped_img.shape[0]>=crop_size//4 and cropped_img.shape[1]>=crop_size//4:
                    cropped_img = cv2.resize(cropped_img,(crop_size,crop_size))
                    class_folder = os.path.join(save_patches_path,c)
                    if not os.path.exists(class_folder):
                        os.mkdir(class_folder)
                    cv2.imwrite(os.path.join(class_folder,str(id)+'_'+media_id+'_'+image_name),cropped_img)
            except:
                non_existing_images.append(id)
        print(dataset_name)
        print("not found: ",len(non_existing_images))
