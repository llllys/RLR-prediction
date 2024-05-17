import os
import pandas as pd
import numpy as np
from pyproj import CRS, Transformer
import geopandas as gpd
import torch


def is_in_poly_01(latitude, longitude, polys):
    """
    judge is pos (lat, lon) in polygons

    args:
        latitude 
        longitude 
        polys 

    return:
        is_in 
    """
    px, py = latitude, longitude
    
    for poly in polys:
        is_in = 0
        for i, corner in enumerate(poly):
            x1, y1 = corner
            x2, y2 = poly[(i+1) % len(poly)]
            if (x1 == px and y1 == py) or (x2 == px and y2 == py):  
                return 1
            if min(y1, y2) < py <= max(y1, y2):
                x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if (x == px):
                    return 1
                elif x > px: 
                    is_in = not is_in
        if is_in:
            return 1
        
    return 0

def transform_meters_vector(points):
    ''' 
    [latitude, longitude]
    '''
    crs_wgs84 = CRS.from_epsg(4326)
    crs_bjUTM = CRS.from_epsg(32650)
    transformer = Transformer.from_crs(crs_wgs84,crs_bjUTM)
        
    _y, _x = transformer.transform(points[:, 0], points[:, 1])
    x = _x - 4403600
    y = _y - 458000
    return np.array([x, y]).T

def move_spin_vector_matrix(vector, anchor_point, M):

    vector = vector - anchor_point.reshape(-1, 1, 2)
    vector = np.einsum('ijk, imk -> imj', M, vector)

    return vector

def get_intersection_points(intersection_ind=11, 
                            map_path="data/map/demo_yzsfq_hd_intersection_polygon.shp"):
    intersection_points_map = {
        11: 9
    }
    polygon = gpd.read_file(map_path)
    polygon_ind = intersection_points_map[intersection_ind]
    pp = polygon['geometry'][polygon_ind]
    xx, yy = pp.exterior.coords.xy
    pys = [list(zip(yy, xx))]
    pys_arr = np.array(pys[0])
    
    # if simple polygon 
    boundary_ind = [6, 7, 8, 9, 11, 13, 16, 17, 18, 19, 20, 21]
    pys_arr = pys_arr[boundary_ind, :]
    pys_arr = transform_meters_vector(pys_arr)
    return pys_arr


def get_all_files_in_folder(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names



def makedir(args):
    results_root = args.results_path
    model_name = args.model
    results_path = os.path.join(results_root, model_name)
    
    if args.results_dir:
        model_results_dir = os.path.join(results_path, 'models/', args.results_dir)
    else:
        model_results_dir = os.path.join(results_path, 'models/')
    results_results_dir = os.path.join(results_path, 'results/')

    if not os.path.exists(model_results_dir):
        os.makedirs(model_results_dir, exist_ok=True)
    if not os.path.exists(results_results_dir):
        os.makedirs(results_results_dir, exist_ok=True)
    
    model_results_path = os.path.join(model_results_dir, 'model_%04d.p')

    return model_results_path
