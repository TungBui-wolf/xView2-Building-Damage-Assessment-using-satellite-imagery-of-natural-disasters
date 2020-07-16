import os 
import cv2
from cv2 import fillPoly
from shapely import wkt
import numpy as np
from shapely.geometry import mapping
import argparse
import random
from random import seed
seed(16)
from shutil import copy2
from tqdm import tqdm
import json


def generate_localization_polygon(json_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(json_path, "r") as f:
        annotations = json.load(f)
    h = annotations["metadata"]["height"]
    w = annotations["metadata"]["width"]
    mask_img = np.zeros((h, w), np.uint8)
    out_filename = os.path.splitext(os.path.basename(json_path))[0] + ".png"
    for feat in annotations['features']['xy']:
        feat_shape = wkt.loads(feat['wkt'])
        coords = list(mapping(feat_shape)['coordinates'][0])
        cv2.fillPoly(mask_img, [np.array(coords, np.int32)], (1))
    cv2.imwrite(os.path.join(out_dir, out_filename), mask_img)

def generate_damage_polygon(json_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(json_path, "r") as f:
        annotations = json.load(f)

    h = annotations["metadata"]["height"]
    w = annotations["metadata"]["width"]
    mask_img = np.zeros((h, w), np.uint8)

    damage_dict = {
        "no-damage": 1,
        "minor-damage": 2,
        "major-damage": 3,
        "destroyed": 4,
        "un-classified": 1
    }
    out_filename = os.path.splitext(os.path.basename(json_path))[0] + ".png"
    for feat in annotations['features']['xy']:
        feat_shape = wkt.loads(feat['wkt'])
        coords = list(mapping(feat_shape)['coordinates'][0])
        fillPoly(mask_img, [np.array(coords, np.int32)], damage_dict[feat['properties']['subtype']])
    cv2.imwrite(os.path.join(out_dir, out_filename), mask_img)


if __name__=="__main__":
    parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
    arg = parser.add_argument
    arg('--path', type=str, default='./data/tier3/', help='Path to get some test data')

    args = parser.parse_args()
    assert len(os.listdir(args.path)) > 0 , f" '{args.path}' is empty"

    src_img_dir = os.path.join(args.path, 'images')
    src_json_dir = os.path.join(args.path, 'labels')

    test_dir = './data/test'
    test_img_dir = os.path.join(test_dir, 'images')
    test_msk_dir = os.path.join(test_dir, 'masks')
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_msk_dir, exist_ok=True)

    all_files = []
    for f in os.listdir(src_img_dir):
        if '_pre_disaster' in f:
            all_files.append(f)  

    print(len(all_files))

    ran = random.sample(all_files, 1000)

    for f in tqdm(ran, desc='Get image test folder'):
        src_pre_img = os.path.join(src_img_dir, f)
        test_pre_img = os.path.join(test_img_dir, f)
        copy2(src_pre_img, test_pre_img)

        src_post_img = os.path.join(src_img_dir, f.replace('_pre_', '_post_'))
        test_post_img = os.path.join(test_img_dir, f.replace('_pre_', '_post_'))
        copy2(src_post_img, test_post_img)

        src_pre_json = os.path.join(src_json_dir, f.replace('.png', '.json'))
        generate_localization_polygon(src_pre_json, test_msk_dir)

        src_post_json = os.path.join(src_json_dir, f.replace('_pre_', '_post_').replace('.png', '.json'))
        generate_damage_polygon(src_post_json, test_msk_dir)
