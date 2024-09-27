import folium
from folium.plugins import HeatMap
from folium.plugins import MiniMap
import pandas as pd
import os

def aggregated_df_for_gps_visualization(dataset):
    aggregated_count = dataset[['point.media.deployment.campaign.key', 'label.translated.name']].groupby('point.media.deployment.campaign.key').agg(lambda x: x.value_counts().to_dict()).reset_index()
    aggregated_dataset_names = dataset[['point.media.deployment.campaign.key', 'label.translated.name']].groupby('point.media.deployment.campaign.key').agg(lambda x: x.value_counts().to_dict()).reset_index()
    aggregated_lon = dataset[['point.media.deployment.campaign.key',"point.pose.lon"]].groupby('point.media.deployment.campaign.key').agg(lambda x: x.mean()).reset_index()
    aggregated_lat= dataset[['point.media.deployment.campaign.key',"point.pose.lat"]].groupby('point.media.deployment.campaign.key').agg(lambda x: x.mean()).reset_index()
    aggregated_df = pd.concat([pd.DataFrame(aggregated_dataset_names["point.media.deployment.campaign.key"]), pd.DataFrame(aggregated_lon["point.pose.lon"]), pd.DataFrame(aggregated_lat["point.pose.lat"]),pd.DataFrame(aggregated_count["label.translated.name"]) ],axis= 1,ignore_index=True)
    aggregated_df.columns = ['point.media.deployment.campaign.key', 'point.pose.lon', 'point.pose.lat','label.translated.name']
    return aggregated_df


def save_htmls(dataset_final,test_ex_df):
    aggregated_df = aggregated_df_for_gps_visualization(dataset_final)
    map_center = [aggregated_df['point.pose.lat'].mean(), aggregated_df['point.pose.lon'].mean()]
    my_map = folium.Map(location=map_center, zoom_start=14)
    for i in range(0,len(aggregated_df)):
        folium.Marker(location=[aggregated_df.iloc[i]['point.pose.lat'], aggregated_df.iloc[i]['point.pose.lon']],
                    popup=f"Dataset Name: {aggregated_df.iloc[i]['point.media.deployment.campaign.key']}",
                    icon=folium.Icon(color="blue"),
                    ).add_to(my_map)

    # Add a heatmap
    heat_data = [[row['point.pose.lat'], row['point.pose.lon']] for index, row in aggregated_df.iterrows()]
    HeatMap(heat_data).add_to(my_map)

    # Add a minimap
    minimap = MiniMap(toggle_display=True)
    my_map.add_child(minimap)
    folium.TileLayer('CartoDB positron').add_to(my_map)
    my_map.save(os.path.join("dataset","figs_dataset_analysis","gps_data_all.html"))


    #####################################################################################

    aggregated_df = aggregated_df_for_gps_visualization(test_ex_df)
    for i in range(0,len(aggregated_df)):
        folium.Marker(location=[aggregated_df.iloc[i]['point.pose.lat'], aggregated_df.iloc[i]['point.pose.lon']],
                    popup=f"Dataset Name: {aggregated_df.iloc[i]['point.media.deployment.campaign.key']}",
                    icon=folium.Icon(color="red"),
                    ).add_to(my_map)

    # Add a heatmap
    heat_data = [[row['point.pose.lat'], row['point.pose.lon']] for index, row in aggregated_df.iterrows()]
    HeatMap(heat_data).add_to(my_map)

    # Add a minimap
    minimap = MiniMap(toggle_display=True)
    my_map.add_child(minimap)
    folium.TileLayer('CartoDB positron').add_to(my_map)
    my_map.save(os.path.join("dataset","figs_dataset_analysis","gps_data_all_plus_text_extra.html"))



if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","train_df.csv"),low_memory=False)
    valid_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","valid_df.csv"),low_memory=False)
    test_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","test_df.csv"),low_memory=False)
    test_ex_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","test_ex_df.csv"),low_memory=False)
    dataset_final = pd.concat([train_df, valid_df, test_df])
    save_htmls(dataset_final,test_ex_df)