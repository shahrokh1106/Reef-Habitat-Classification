import folium
from folium.plugins import HeatMap
from folium.plugins import MiniMap
import pandas as pd
import os
import random 
import numpy as np
SEED = 2
random.seed(SEED)
np.random.seed(SEED)


def get_all_locations(df):
    return df[['point.pose.lat', 'point.pose.lon']].dropna().values.tolist()

def sample_locations(locations, max_markers=50):
    if len(locations) <= max_markers:
        return locations 
    step = max(1, len(locations) // max_markers)
    sampled_locations = [random.choice(locations[i:i+step]) for i in range(0, len(locations), step)]
    return sampled_locations[:max_markers] 

def save_htmls_for_locations(dataset_final,test_ex_df=None):
    locations = get_all_locations(dataset_final)
    sampled_locations = sample_locations(locations, max_markers=1000)
    my_map = folium.Map(location=[-35, 150], zoom_start=5)
    HeatMap(locations, radius=20, blur=20, max_zoom=12).add_to(my_map)
    # Add randomly selected markers
    for loc in sampled_locations:
        lat = loc[0]
        lon = loc[1]
        folium.Marker(
            location=[lat, lon],
            popup=f"Sampled Location",
            icon=folium.Icon(color="red"),
        ).add_to(my_map)
    if isinstance(test_ex_df,pd.DataFrame):
        locations_ex= get_all_locations(dataset_final)
        sampled_locations = sample_locations(locations_ex, max_markers=500)
        for loc in sampled_locations:
            lat = loc[0]
            lon = loc[1]
            folium.Marker(
                location=[lat, lon],
                popup=f"Sampled Location",
                icon=folium.Icon(color="blue"),
            ).add_to(my_map)
    minimap = MiniMap(toggle_display=True)
    my_map.add_child(minimap)
    folium.TileLayer('CartoDB positron').add_to(my_map)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="© Google Maps",
        name="Google Satellite",
        overlay=False,
        control=True
    ).add_to(my_map)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="© Google Hybrid",
        name="Google Hybrid",
        overlay=False,
        control=True
    ).add_to(my_map)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="© Esri",
        name="ESRI Satellite",
        overlay=False,
        control=True
    ).add_to(my_map)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="© Esri",
        name="Labels & Boundaries",
        overlay=True,
        control=True
    ).add_to(my_map)
    folium.LayerControl().add_to(my_map)
    my_map.save(os.path.join("dataset","figs_dataset_analysis","gps_data_all.html"))

if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","train_df.csv"),low_memory=False)
    valid_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","valid_df.csv"),low_memory=False)
    test_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","test_df.csv"),low_memory=False)
    test_ex_df = pd.read_csv(os.path.join("dataset","dataset_csv_files","training_datasets","test_ex_df.csv"),low_memory=False)
    dataset_final = pd.concat([train_df, valid_df, test_df])

    # save_htmls_for_locations(dataset_final,test_ex_df)
    save_htmls_for_locations(dataset_final,None)