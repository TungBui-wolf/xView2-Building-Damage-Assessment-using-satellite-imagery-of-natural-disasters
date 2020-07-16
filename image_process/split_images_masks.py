import cv2
import os
from os import path, makedirs, listdir
from tqdm import tqdm


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def split_image_for_train(img_path, out_dir, overlap=0):
    out_image_dir = path.join(out_dir, 'images')
    os.makedirs(out_image_dir, exist_ok=True)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img_h, img_w, _= img.shape

    out_mask_dir = path.join(out_dir, 'masks')
    os.makedirs(out_mask_dir, exist_ok=True)
    mask_path = img_path.replace('images', 'masks')
    msk = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    msk_h, msk_w = msk.shape
    
    img_name = img_path.split('/')[-1]
    msk_name = mask_path.split('/')[-1]

    split_width = 256
    split_height = 256

    X_points = start_points(img_w, split_width, overlap=overlap)
    Y_points = start_points(img_h, split_height, overlap=overlap)

    count = 0

    for i in Y_points:
        for j in X_points:
            split_img = img[i:i+split_height, j:j+split_width]
            # print(os.path.join(out_dir, img_name.replace('.png', '_split_{}.png'.format(count))))
            cv2.imwrite(os.path.join(out_image_dir, img_name.replace('.png', '_split_{}.png'.format(count))), split_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            split_msk = msk[i:i+split_height, j:j+split_width]
            cv2.imwrite(os.path.join(out_mask_dir, msk_name.replace('.png', '_split_{}.png'.format(count))), split_msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            count += 1  

if __name__=="__main__":
    train_dirs = ['data/train', 'data/tiger3']
    out_dir = 'data/split_data'

    all_files = []
    for train in train_dirs:
        for f in listdir(path.join(train, 'images')):
            all_files.append(path.join(train, 'images', f))

    for img_path in tqdm(all_files, desc='split_images'):
        split_image_for_train(img_path=img_path, out_dir=out_dir)         
