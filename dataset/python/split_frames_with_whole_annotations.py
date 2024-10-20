import glob
import os
import json
import shutil


if __name__ == '__main__':
    with open(os.path.join("dataset","dataset_csv_files","csvs_whole_frame_annotations","frame_annotation.json"), "r") as f:
        class_dict = json.load(f)  
    class_names = list(class_dict.keys())
    source_path = os.path.join("dataset","frames_with_whole_annotations")
    if not os.path.exists(source_path):
        print("Plase first download frames_with_whole_annotations using download_frames_with_whole_annotations.py")
    else:
        output_path = os.path.join("dataset","grouped_frames_with_whole_annotations")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        source_image_names = [os.path.basename(x) for x in glob.glob(os.path.join(source_path, "*"))]
        for class_name in class_names:
            output_class_path = os.path.join(output_path, class_name)
            if not os.path.exists(output_class_path):
                os.makedirs(output_class_path)
            for image_name in class_dict[class_name]:
                if image_name in source_image_names:
                    shutil.copy(os.path.join(source_path, image_name), os.path.join(output_class_path, image_name))
            