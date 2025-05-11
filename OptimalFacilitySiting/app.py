import os
import zipfile
import geopandas as gpd
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
from shapely.geometry import Point
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'zip'}

INFRASTRUCTURE_FILES = {
    'health': 'dataset/Health Facilities.shp',
    'police': 'dataset/Police Stations.shp',
    'roads': 'dataset/Trunk Roads N13.shp'
}

# === Helper Functions ===

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

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

# === Routes ===

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'Error', 'error': 'No file part in the request'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'Error', 'error': 'No selected file'})

        if not allowed_file(file.filename):
            return jsonify({'message': 'Error', 'error': 'Only .zip shapefiles are supported'})

        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'shapefile.zip')
        file.save(zip_path)

        extract_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'shapefile_extracted')
        os.makedirs(extract_folder, exist_ok=True)
        extract_zip(zip_path, extract_folder)

        shp_file = None
        for filename in os.listdir(extract_folder):
            if filename.endswith('.shp'):
                shp_file = os.path.join(extract_folder, filename)
                break

        if not shp_file:
            return jsonify({'message': 'Error', 'error': 'No .shp file found in ZIP'})

        gdf = gpd.read_file(shp_file)

        # Log CRS for debug
        print(f"Original CRS: {gdf.crs}")

        # If no CRS, force to EPSG:4326
        if gdf.crs is None:
            print("CRS undefined. Forcing to EPSG:4326.")
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            print(f"CRS is {gdf.crs}. Reprojecting to EPSG:4326 first.")
            gdf = gdf.to_crs(epsg=4326)

        # Now convert to 32733 (meters)
        gdf = gdf.to_crs(epsg=32733)

        # Load infrastructure, also in 32733
        infra = load_infrastructure()

        # Compute distances
        gdf = compute_nearest(gdf, infra)

        # Score and get top 5
        gdf_scored = scoring_logic(gdf)
        top5 = gdf_scored.head(5).to_crs("EPSG:4326")

        # Save GeoJSON
        top5_json_path = os.path.join(app.config['UPLOAD_FOLDER'], 'top5.geojson')
        top5.to_file(top5_json_path, driver='GeoJSON')

        return jsonify({'message': 'Success', 'geojson': 'uploads/top5.geojson'})

    except Exception as e:
        return jsonify({'message': 'Error', 'error': str(e)})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)



