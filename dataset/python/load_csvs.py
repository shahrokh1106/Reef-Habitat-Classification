import pandas as pd

def filter_reef_dataset (dataset):

    mask_1 = dataset["label.name"] == 'Unscorable'
    dataset.loc[mask_1, "label.translated.name"] = dataset.loc[mask_1, 'label.name']
    
    mask_2 = dataset["label.name"] == 'Barnacles'
    dataset.loc[mask_2, "label.translated.name"] = "Sessile invertebrate community"
    
    mask_3 = dataset["label.name"] == 'Colpomenia spp.'
    dataset.loc[mask_3, "label.translated.name"] = "Turf"
    
    mask_4 = dataset["label.name"] == 'Diver (Diver Equipment)'
    dataset.loc[mask_4, "label.translated.name"] = "Unscorable"
    
    mask_5 = dataset["label.name"] == 'Drop Camera Pole'
    dataset.loc[mask_5, "label.translated.name"] = "Unscorable"
    
    mask_6 = dataset["label.name"] == 'Feather Star'
    dataset.loc[mask_6, "label.translated.name"] = "Sessile invertebrate community"
    
    mask_7 = dataset["label.name"] == 'Globose/Saccate'
    dataset.loc[mask_7, "label.translated.name"] = "Turf"
    
    mask_8 = dataset["label.name"] == 'Heliocidaris erythrogramma'
    dataset.loc[mask_8, "label.translated.name"] = "Mobile invertebrate"
    
    mask_9 = dataset["label.name"] == 'Mobile gastropods'
    dataset.loc[mask_9, "label.translated.name"] = "Mobile invertebrate"
    
    mask_10 = dataset["label.name"] == 'Pleuroploca australasia'
    dataset.loc[mask_10, "label.translated.name"] = "Mobile invertebrate"
    
    mask_11 = dataset["label.name"] == 'Polychaete'
    dataset.loc[mask_11, "label.translated.name"] = "Mobile invertebrate"
    
    mask_12 = dataset["label.name"] == 'Sessile bivalves'
    dataset.loc[mask_12, "label.translated.name"] = "Sessile invertebrate community"
    
    mask_13 = dataset["label.name"] == 'Sessile gastropods'
    dataset.loc[mask_13, "label.translated.name"] = "Sessile invertebrate community"
    
    mask_14 = dataset["label.name"] == 'Urchins'
    dataset.loc[mask_14, "label.translated.name"] = "Mobile invertebrate"
    
    dataset = dataset.dropna(subset=['label.translated.name'])
    
    dataset=dataset.drop(columns='Unnamed: 0')

    return dataset

def load_final_dataset():
    dataset_from_groups = pd.read_csv("dataset/dataset_csv_files/annotations_from_group_datastes.csv",low_memory=False)
    subdataset_nz = dataset_from_groups[dataset_from_groups['point.media.deployment.campaign.key'] == "RLS_Cape Rodney - Okakari Point Marine Reserve_2012"]
    dataset_from_groups = dataset_from_groups[~dataset_from_groups['point.media.deployment.campaign.key'].isin(["RLS_Cape Rodney - Okakari Point Marine Reserve_2012"])]
    dataset_eastcoast = pd.read_csv("dataset/dataset_csv_files/annotations-u1683-East_Coast_Tasmania_Habitat_Classification_-East_Coast_Tasmania_Habitat_Classification_-_Random_Points-14012-2e62fbdd9d24106ae84f-dataframe.csv")
    dataset_nz = pd.read_csv("dataset/dataset_csv_files/annotations-u1576-Habitat_Classification_UoA_Drop_Camera_Full-UoA_Drop_Camera_Habitat_Classification_25_RANDOM_POINTS-14010-c9a6f0d63a8564b31084-dataframe.csv")
    dataset_au = pd.concat([dataset_from_groups, dataset_eastcoast], ignore_index=True)
    dataset_nz = pd.concat([dataset_nz, subdataset_nz], ignore_index=True)
    dataset_final = pd.concat([dataset_au, dataset_nz], ignore_index=True)

    dataset_urchin1 = pd.read_csv("dataset/dataset_csv_files/annotations-u115-NSW_DPI_Urchins_-NSW_DPI_urchins_imported_dataset-13277-f0727cabf5a4a185eadd-dataframe.csv")
    dataset_urchin2 = pd.read_csv("dataset/dataset_csv_files/annotations-u1576-UoA_Sea_Urchin-UoA_Sea_Urchin_Classification-13235-74f5548ccc83f12ed2ed-dataframe.csv")
    dataset_urchin3 = pd.read_csv("dataset/dataset_csv_files/annotations-u1683-Urchins_-_Eastern_Tasmania-Import-13088-f0727cabf5a4a185eadd-dataframe.csv")
    dataset_urchin = pd.concat([dataset_urchin1, dataset_urchin2, dataset_urchin3], ignore_index=True)
    dataset_urchin["label.translated.name"]= "Urchin"

    # recently added ####################
    dataset_additional =  pd.read_csv("dataset/dataset_csv_files/annotations-u1576-Habitat_Classification_UoA_Drop_Camera_Full-Additional_labels-15839-af9f09d055f9078423e4-dataframe.csv")
    dataset_additional.loc[dataset_additional["label.name"]=="Drop Camera Pole","label.translated.name"]="Unscorable"
    dataset_additional.loc[dataset_additional["label.name"]=="Diver (Diver Equipment)","label.translated.name"]="Unscorable"
    dataset_additional.loc[dataset_additional["label.name"]=="Barnacles","label.translated.name"]="Sessile invertebrate community"
    dataset_additional.loc[dataset_additional["label.name"]=="Paper oyster","label.translated.name"]="Sessile invertebrate community"
    dataset_additional.loc[dataset_additional["label.name"]=="Other canopy forming macroalgae > Carpophyllum spp.","label.translated.name"]="Carpophyllum spp."
    dataset_additional = dataset_additional.loc[~dataset_additional['label.name'].isin(["Mobile gastropods"])]

    dataset_final = pd.concat([dataset_final, dataset_urchin, dataset_additional], ignore_index=True)
    dataset_final = filter_reef_dataset (dataset_final)
    return dataset_final

   