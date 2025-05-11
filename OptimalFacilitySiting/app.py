import os
import zipfile
import shutil
import geopandas as gpd
import pandas as pd
import numpy as np
import streamlit as st

from shapely.geometry import Point

# === Streamlit Setup ===
st.set_page_config(page_title="Optimal Facility Siting", layout="wide")
st.title("📍 Optimal Facility Siting App")
st.markdown("Upload a **.zip shapefile** of your target area to analyze suitability based on proximity to health, police, and roads.")

# === Constants ===
UPLOAD_FOLDER = "uploads"
EXTRACT_FOLDER = "uploads/extracted"
os.makedirs(EXTRACT_FOLDER, exist_ok=True)

INFRASTRUCTURE_FILES = {
    'health': 'dataset/Health Facilities.shp',
    'police': 'dataset/Police Stations.shp',
    'roads': 'dataset/Trunk Roads N13.shp'
}

# === Helper Functions ===

def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(extract_to)

def load_infrastructure():
    infra = {}
    for key, path in INFRASTRUCTURE_FILES.items():
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        gdf = gdf.to_crs("EPSG:32733")
        infra[key] = gdf
    return infra

def compute_nearest(gdf, infra):
    for key, gdf_infra in infra.items():
        if gdf_infra.empty:
            gdf[f'dist_{key}'] = np.nan
            continue
        gdf[f'dist_{key}'] = gdf.geometry.apply(
            lambda geom: gdf_infra.distance(geom.centroid).min()
        )
    return gdf

def scoring_logic(gdf):
    if 'dens_sqkm' not in gdf.columns:
        gdf['dens_sqkm'] = 1
    else:
        gdf['dens_sqkm'] = pd.to_numeric(gdf['dens_sqkm'], errors='coerce').fillna(0)

    gdf['dens_score'] = (gdf['dens_sqkm'] - gdf['dens_sqkm'].min()) / \
                        (gdf['dens_sqkm'].max() - gdf['dens_sqkm'].min() + 1e-9)

    for key in ['health', 'police', 'roads']:
        dist_col = f'dist_{key}'
        if dist_col in gdf.columns:
            dist = gdf[dist_col]
            gdf[f'{key}_score'] = 1 - ((dist - dist.min()) / (dist.max() - dist.min() + 1e-9))
        else:
            gdf[f'{key}_score'] = 0

    gdf['score'] = (
        0.4 * gdf['dens_score'] +
        0.2 * gdf['health_score'] +
        0.2 * gdf['police_score'] +
        0.2 * gdf['roads_score']
    )

    return gdf.sort_values(by='score', ascending=False).copy()

# === Upload UI ===

uploaded_file = st.file_uploader("Upload a zipped shapefile (.zip)", type="zip")

if uploaded_file:
    try:
        # Clear previous files
        if os.path.exists(EXTRACT_FOLDER):
            shutil.rmtree(EXTRACT_FOLDER)
        os.makedirs(EXTRACT_FOLDER)

        # Save uploaded ZIP
        zip_path = os.path.join(UPLOAD_FOLDER, "shapefile.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract ZIP
        extract_zip(zip_path, EXTRACT_FOLDER)

        # Find shapefile
        shp_file = None
        for file in os.listdir(EXTRACT_FOLDER):
            if file.endswith('.shp'):
                shp_file = os.path.join(EXTRACT_FOLDER, file)
                break

        if not shp_file:
            st.error("No .shp file found in the uploaded ZIP.")
        else:
            gdf = gpd.read_file(shp_file)

            if gdf.crs is None:
                st.warning("CRS undefined. Assuming EPSG:4326.")
                gdf.set_crs(epsg=4326, inplace=True)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)

            gdf = gdf.to_crs(epsg=32733)

            # Load and process infrastructure
            infra = load_infrastructure()
            gdf = compute_nearest(gdf, infra)
            gdf_scored = scoring_logic(gdf)
            top5 = gdf_scored.head(5).to_crs("EPSG:4326")

            st.success("✅ Processing complete! Showing top 5 ranked locations.")
            st.map(top5)

            # Downloadable GeoJSON
            geojson_path = os.path.join(UPLOAD_FOLDER, "top5.geojson")
            top5.to_file(geojson_path, driver="GeoJSON")
            with open(geojson_path, "rb") as f:
                st.download_button("📥 Download Top 5 as GeoJSON", f, file_name="top5.geojson", mime="application/json")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")




