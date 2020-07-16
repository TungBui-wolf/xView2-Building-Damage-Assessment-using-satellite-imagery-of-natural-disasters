from os import path, makedirs, listdir
import numpy as np

import torch
from torch import nn
from torch.backends import cudnn

from torch.autograd import Variable

from tqdm import tqdm
import timeit
import argparse
import cv2

from src.models import ResNet34_Unet_Loc, ResNet34_Unet_Double

from src.utils import *

def predict_loc():
    test_dir = 'data/test/images'
    pred_loc_folder = 'data/test/predict/pred34_loc'
    models_folder = 'weights'

    makedirs(pred_loc_folder, exist_ok=True)

    snap_to_load = '_resnet34_unet_loc_best'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet34_Unet_Loc(pretrained=True, bilinear=False).to(device)
    model = nn.DataParallel(model).to(device)

    print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print("loaded checkpoint '{}' (epoch {}, best_score {})"
            .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
    model.eval()  

    with torch.no_grad():
        for f in tqdm(sorted(listdir(test_dir))):
            if '_pre_' in f:
                fn = path.join(test_dir, f)

                image = cv2.imread(fn, cv2.IMREAD_COLOR)
                splited_imgs = split_image(img=image)

                splited_msks = []
                for img in splited_imgs:
                    img = preprocess_inputs(img)

                    inp = []
                    inp.append(img)
                    inp.append(img[::-1, ...])
                    inp.append(img[:, ::-1, ...])
                    inp.append(img[::-1, ::-1, ...])
                    inp = np.asarray(inp, dtype='float')
                    inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
                    inp = Variable(inp).to(device)

                    pred = []
                                  
                    msk = model(inp)
                    msk = torch.sigmoid(msk)
                    msk = msk.cpu().numpy()

                    pred.append(msk[0, ...])
                    pred.append(msk[1, :, ::-1, :])
                    pred.append(msk[2, :, :, ::-1])
                    pred.append(msk[3, :, ::-1, ::-1])

                    pred_full = np.asarray(pred).mean(axis=0)
                    
                    msk = pred_full * 255
                    msk = msk.astype('uint8').transpose(1, 2, 0)

                    splited_msks.append(msk)

                mask = merge_image(splited_imgs=splited_imgs)
                cv2.imwrite(path.join(pred_loc_folder, f.replace('.png', '_part1.png')), mask[..., 0], [cv2.IMWRITE_PNG_COMPRESSION, 9])

def predict_cls():
    test_dir = 'data/test/images'
    pred_cls_folder = 'data/test/predict/pred34_cls'
    models_folder = 'weights'

    makedirs(pred_cls_folder, exist_ok=True)

    snap_to_load = '_resnet34_unet_cls_best'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet34_Unet_Double(pretrained=True, bilinear=False).to(device)
    model = nn.DataParallel(model).to(device)

    print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print("loaded checkpoint '{}' (epoch {}, best_score {})"
            .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
    model.eval()   

    with torch.no_grad():
        for f in tqdm(sorted(listdir(test_dir))):
            if '_pre_' in f:
                fn = path.join(test_dir, f)

                img = cv2.imread(fn, cv2.IMREAD_COLOR)
                img2 = cv2.imread(fn.replace('_pre_', '_post_'), cv2.IMREAD_COLOR)

                image = np.concatenate([img, img2], axis=2)

                splited_imgs = split_image(img=image)

                splited_msks = []

                for img in splited_imgs:

                    img = preprocess_inputs(img)

                    inp = []
                    inp.append(img)
                    inp.append(img[::-1, ...])
                    inp.append(img[:, ::-1, ...])
                    inp.append(img[::-1, ::-1, ...])
                    inp = np.asarray(inp, dtype='float')
                    inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
                    inp = Variable(inp).to(device)

                    pred = []
                    
                    msk = model(inp)
                    msk = torch.sigmoid(msk)
                    msk = msk.cpu().numpy()

                    pred.append(msk[0, ...])
                    pred.append(msk[1, :, ::-1, :])
                    pred.append(msk[2, :, :, ::-1])
                    pred.append(msk[3, :, ::-1, ::-1])

                    pred_full = np.asarray(pred).mean(axis=0)
                    
                    msk = pred_full * 255
                    msk = msk.astype('uint8').transpose(1, 2, 0)

                    splited_msks.append(msk)

                mask = merge_image(splited_msks)

                cv2.imwrite(path.join(pred_cls_folder, f.replace('.png', '_part1.png')), mask[..., :3], [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(path.join(pred_cls_folder, f.replace('.png', '_part2.png')), mask[..., 2:], [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
    arg = parser.add_argument
    arg('--mode', type=str, default='loc', help='Localization or Classification')

    args = parser.parse_args()
    
    assert args.mode in ['loc', 'cls'], f"mode '{args.mode}' was not one of 'loc' or 'cls'"

    if args.mode == 'loc':
        t0 = timeit.default_timer()
        predict_loc()
        elapsed = timeit.default_timer() - t0
        print('Time: {:.3f} min'.format(elapsed / 60))
    if args.mode == 'cls':
        t0 = timeit.default_timer()
        predict_cls()
        elapsed = timeit.default_timer() - t0
        print('Time: {:.3f} min'.format(elapsed / 60))