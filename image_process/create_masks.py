import json
import os
import argparse
from functools import partial
from multiprocessing.pool import Pool
from os import cpu_count

import cv2
from cv2 import fillPoly
from shapely import wkt
import numpy as np
from shapely.geometry import mapping
from tqdm import tqdm


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
        fillPoly(mask_img, [np.array(coords, np.int32)], (255))
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


if __name__ == '__main__':

    train_dirs =  ["data/train", "data/tier3"]
    
    for train_dir in train_dirs:
        out_dir = os.path.join(train_dir, "masks")
        in_dir = os.path.join(train_dir, "labels")
        pre_images = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if '_pre_' in f]
        post_images = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if '_post_' in f]

        pool = Pool(cpu_count())
        with tqdm(total=len(pre_images), desc='gen_pre_mask in {}'.format(train_dir)) as pbar:
            for i, v in enumerate(pool.imap_unordered(partial(generate_localization_polygon, out_dir=out_dir), pre_images)):
                pbar.update()
        with tqdm(total=len(post_images), desc='gen_post_mask in {}'.format(train_dir)) as pbar:
            for i, v in enumerate(pool.imap_unordered(partial(generate_damage_polygon, out_dir=out_dir), post_images)):
                pbar.update()
