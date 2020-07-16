from os import path, makedirs, listdir
from multiprocessing import Pool
import numpy as np

from tqdm import tqdm
import timeit
import cv2

from skimage.morphology import square, dilation

pred_cls_folders = ['data/test/predict/pred34_cls', 'data/test/predict/pred-unet_cls', 'data/test/predict/pred50_cls']
cls_coefs = [1.0] * 3

pred_loc_folders = ['data/test/predict/pred34_loc', 'data/test/predict/pred-unet_loc', 'data/test/predict/pred50_loc']
loc_coefs = [1.0] * 3

sub_folder = 'data/test/predictions'

_thr = [0.38, 0.13, 0.14]

def post_process_image(f):
    # localization
    loc_preds = []
    _i = -1
    for d in pred_loc_folders:
        _i += 1
        msk = cv2.imread(path.join(d, f), cv2.IMREAD_UNCHANGED)
        loc_preds.append(msk * loc_coefs[_i])
    loc_preds = np.asarray(loc_preds).astype('float').sum(axis=0) / np.sum(loc_coefs) / 255

    # classification
    cls_preds = []
    _i = -1
    for d in pred_cls_folders:
        _i += 1
        msk1 = cv2.imread(path.join(d, f), cv2.IMREAD_UNCHANGED)
        msk2 = cv2.imread(path.join(d, f.replace('_part1', '_part2')), cv2.IMREAD_UNCHANGED)
        msk = np.concatenate([msk1, msk2[..., 1:]], axis=2)
        cls_preds.append(msk * cls_coefs[_i])
    cls_preds = np.asarray(cls_preds).astype('float').sum(axis=0) / np.sum(cls_coefs) / 255

    msk_dmg = cls_preds[..., 1:].argmax(axis=2) + 1
    msk_loc = (1 * ((loc_preds > _thr[0]) | ((loc_preds > _thr[1]) & (msk_dmg > 1) & (msk_dmg < 4)) | ((loc_preds > _thr[2]) & (msk_dmg > 1)))).astype('uint8')
    
    msk_dmg = msk_dmg * msk_loc
    _msk = (msk_dmg == 2)
    if _msk.sum() > 0:
        _msk = dilation(_msk, square(5))
        msk_dmg[_msk & msk_dmg == 1] = 2

    msk_dmg = msk_dmg.astype('uint8')
    cv2.imwrite(path.join(sub_folder, f.replace('_part1.png', '.png')), msk_loc, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(path.join(sub_folder, f.replace('_pre_', '_post_').replace('_part1.png', '.png')), msk_dmg, [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(sub_folder, exist_ok=True)

    all_files = []
    for f in tqdm(sorted(listdir(pred_loc_folders[0]))):
        if '_part1.png' in f:
#             all_files.append(f)
            post_process_image(f)

#     with Pool() as pool:
#         _ = pool.map(post_process_image, all_files)


    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))