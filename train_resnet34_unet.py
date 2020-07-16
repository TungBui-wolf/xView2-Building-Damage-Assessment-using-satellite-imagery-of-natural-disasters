from src.dataset import *
from src.models import ResNet34_Unet_Loc, ResNet34_Unet_Double
from src.utils import AverageMeter
from src.metrics import *
from os import path, makedirs, listdir
from sklearn.model_selection import train_test_split
from src.losses import *
import timeit
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import torch.optim as optim
import torch
from torch import nn
from tqdm import tqdm
import argparse
import gc

def train_loc_model(model, data_loaders, optimizer, scheduler, seg_loss, num_epochs, weight_dir, snapshot_name, log_dir, best_score=0):

    writer = SummaryWriter(log_dir + 'localization')
    print('Tensorboard is recording into folder: ' + log_dir + 'localization')

    torch.cuda.empty_cache()

    for epoch in range(num_epochs):
        losses = AverageMeter()

        dices = AverageMeter()
        iterator = data_loaders['train']
        iterator = tqdm(iterator)
        model.train()
        for i, sample in enumerate(iterator):
            imgs = sample["img"].cuda(non_blocking=True)
            msks = sample["msk"].cuda(non_blocking=True)
        
            out = model(imgs)

            loss = seg_loss(out, msks)

            with torch.no_grad():
                _probs = torch.sigmoid(out[:, 0, ...])
                dice_sc = 1 - dice_round(_probs, msks[:, 0, ...])

            losses.update(loss.item(), imgs.size(0))

            dices.update(dice_sc, imgs.size(0))

            iterator.set_description("Epoch {}/{}, lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                    epoch, num_epochs, scheduler.get_lr()[-1], loss=losses, dice=dices))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
            optimizer.step()

            writer.add_scalar('Train/Loss', losses.avg, epoch)
            writer.add_scalar('Train/Dice', dices.avg, epoch)
            writer.flush()
        
        if epoch % 2 == 0:
            torch.cuda.empty_cache()

            model = model.eval()
            dices0 = []

            _thr = 0.5
            iterator = data_loaders['val']
            iterator = tqdm(iterator)
            with torch.no_grad():
                for i, sample in enumerate(iterator):
                    msks = sample["msk"].numpy()
                    imgs = sample["img"].cuda(non_blocking=True)
            
                    out = model(imgs)

                    msk_pred = torch.sigmoid(out[:, 0, ...]).cpu().numpy()
            
                    for j in range(msks.shape[0]):
                        dices0.append(dice(msks[j, 0], msk_pred[j] > _thr))

            d = np.mean(dices0)

            writer.add_scalar('Val/Dice', d, epoch)
            writer.flush()

            print("Val Dice: {}".format(d))

            if d > best_score:
                best_score = d
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_score': d,
                }, path.join(weight_dir, snapshot_name + '_best'))

            print("score: {}\tscore_best: {}".format(d, best_score))

        writer.close()
            
    return best_score

def train_cls_model(model, data_loaders, optimizer, scheduler, seg_loss, ce_loss, num_epochs, weight_dir, snapshot_name, log_dir, best_score=0):
    torch.cuda.empty_cache()

    writer = SummaryWriter(log_dir + 'classification')
    print('Tensorboard is recording into folder: ' + log_dir + 'classification')

    for epoch in range(num_epochs):
        losses = AverageMeter()
        dices = AverageMeter()
        
        iterator = data_loaders['train']
        iterator = tqdm(iterator)
        model.train()
        for i, sample in enumerate(iterator):
            imgs = sample["img"].cuda(non_blocking=True)
            msks = sample["msk"].cuda(non_blocking=True)
            lbl_msk = sample["lbl_msk"].cuda(non_blocking=True)
        
            out = model(imgs)

            loss_loc = seg_loss(out[:, 0, ...], msks[:, 0, ...])
            loss1 = seg_loss(out[:, 1, ...], msks[:, 1, ...])
            loss2 = seg_loss(out[:, 2, ...], msks[:, 2, ...])
            loss3 = seg_loss(out[:, 3, ...], msks[:, 3, ...])
            loss4 = seg_loss(out[:, 4, ...], msks[:, 4, ...])

            loss5 = ce_loss(out, lbl_msk)

            loss = 0.1 * loss_loc + 0.1 * loss1 + 0.3 * loss2 + 0.3 * loss3 + 0.2 * loss4 + loss5 * 11

            with torch.no_grad():
                _probs = torch.sigmoid(out[:, 0, ...])
                dice_sc = 1 - dice_round(_probs, msks[:, 0, ...])

            losses.update(loss.item(), imgs.size(0))

            dices.update(dice_sc, imgs.size(0))

            iterator.set_description("Epoch {}/{}, lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                    epoch, num_epochs, scheduler.get_lr()[-1], loss=losses, dice=dices))
        
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
            optimizer.step()

            writer.add_scalar('Train/Loss', losses.avg, epoch)
            writer.add_scalar('Train/Dice', dices.avg, epoch)
            writer.add_scalar('Train/Loc_loss', loss_loc, epoch)
            writer.add_scalar('Train/NoDamage_loss', loss1, epoch)
            writer.add_scalar('Train/MinorDamage_loss', loss2, epoch)
            writer.add_scalar('Train/MajorDamage_loss', loss3, epoch)
            writer.add_scalar('Train/Destroyed_loss', loss4, epoch)
            writer.add_scalar('Train/Cls_loss', loss4, epoch)

            writer.flush()
        
        if epoch % 2 == 0:
            torch.cuda.empty_cache()

            model = model.eval()
            dices0 = []

            tp = np.zeros((4,))
            fp = np.zeros((4,))
            fn = np.zeros((4,))

            _thr = 0.3
            
            iterator = data_loaders['val']
            iterator = tqdm(iterator)
            with torch.no_grad():
                for i, sample in enumerate(iterator):
                    msks = sample["msk"].numpy()
                    lbl_msk = sample["lbl_msk"].numpy()
                    imgs = sample["img"].cuda(non_blocking=True)
                    out = model(imgs)
                    
                    msk_pred = torch.sigmoid(out[:, 0, ...]).cpu().numpy()
                    msk_damage_pred = torch.sigmoid(out).cpu().numpy()[:, 1:, ...]
            
                    for j in range(msks.shape[0]):
                        dices0.append(dice(msks[j, 0], msk_pred[j] > _thr))

                        targ = lbl_msk[j][msks[j, 0] > 0]
                        pred = msk_damage_pred[j].argmax(axis=0)
                        pred = pred * (msk_pred[j] > _thr)
                        pred = pred[msks[j, 0] > 0]
                        for c in range(4):
                            tp[c] += np.logical_and(pred == c, targ == c).sum()
                            fn[c] += np.logical_and(pred != c, targ == c).sum()
                            fp[c] += np.logical_and(pred == c, targ != c).sum()

            d0 = np.mean(dices0)
            f1_sc = np.zeros((4,))
            
            for c in range(4):
                f1_sc[c] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])

            f1 = 4 / np.sum(1.0 / (f1_sc + 1e-6))

            sc = 0.3 * d0 + 0.7 * f1
            print("Val Score: {}, Dice: {}, F1: {}, F1_no-damage: {}, F1_minor-damage: {}, F1_major-damage: {}, F1_destroyed: {}".format(
                sc, d0, f1, f1_sc[0], f1_sc[1], f1_sc[2], f1_sc[3]))

            writer.add_scalar('Val/Score', sc, epoch)
            writer.add_scalar('Val/Dice', d0, epoch)
            writer.add_scalar('Val/NoDamage_F1', f1, epoch)
            writer.add_scalar('Val/MinorDamage_F1', f1_sc[0], epoch)
            writer.add_scalar('Val/MajorDamage_F1', f1_sc[1], epoch)
            writer.add_scalar('Val/Destroyed_F1', f1_sc[2], epoch)
            writer.add_scalar('Val/Cls_F1', f1_sc[3], epoch)

            writer.flush()
            
            if sc > best_score:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_score': sc,
                }, path.join(weight_dir, snapshot_name + '_best'))
                best_score = sc

            print("score: {}\tscore_best: {}".format(sc, best_score))
        
        writer.close()

    return best_score

def main_loc():
    train_dirs = ['data/split_train_data']
    weight_dir = 'weights'
    makedirs(weight_dir, exist_ok=True)
    snapshot_name = '_resnet34_unet_loc'

    log_dir = 'logs/resnet34_unet/'
    makedirs(log_dir, exist_ok=True)

    all_files = []
    for train in train_dirs:
        for f in listdir(path.join(train, 'images')):
            if '_pre_disaster' in f:
                all_files.append(path.join(train, 'images', f))

    # print(len(all_files))
    # all_files = all_files[:100]
    
    cudnn.benchmark = True

    batch_size = 16
    val_batch_size = 8

    train_idxs, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.2, random_state=42)
    print('train: ', len(train_idxs), ' validation: ', len(val_idxs))

    steps_per_epoch = len(train_idxs) // batch_size
    validation_steps = len(val_idxs) // val_batch_size

    print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

    data_train = TrainDataLoc(train_idxs, all_files=all_files)
    val_train = ValDataLoc(val_idxs, all_files=all_files)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_train, batch_size=val_batch_size, num_workers=4, shuffle=False, pin_memory=False)

    data_loaders = {
        'train': train_data_loader,
        'val': val_data_loader
    }

    num_epochs = 55

    model = ResNet34_Unet_Loc(pretrained=True, bilinear=False).cuda()
    
    model = nn.DataParallel(model).cuda()

    seg_loss = ComboLoss({'dice': 1.0, 'focal': 10.0}, per_image=False).cuda()

    optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=6, gamma=0.1)
    
    try:
        snap_to_load = '_resnet34_unet_loc_best'
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(weight_dir, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        state_dict = model.state_dict()
        for k in state_dict:
            if k in loaded_dict and state_dict[k].size() == loaded_dict[k].size():
                state_dict[k] = loaded_dict[k]
        # loaded_dict = sd
        model.load_state_dict(state_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})"
                .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
        best_score = checkpoint['best_score']
        del loaded_dict
        del state_dict
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    except:
        best_score = 0

    history = train_loc_model(model=model, data_loaders=data_loaders, optimizer=optimizer_ft, scheduler=scheduler, seg_loss=seg_loss, num_epochs=num_epochs, weight_dir=weight_dir, snapshot_name=snapshot_name, log_dir=log_dir, best_score=best_score)

    return history

def main_cls():
    train_dirs = ['data/split_train_data']
    weight_dir = 'weights'
    makedirs(weight_dir, exist_ok=True)
    snapshot_name = '_resnet34_unet_cls'
    
    log_dir = 'logs/resnet34_unet/'
    makedirs(log_dir, exist_ok=True)

    all_files = []
    for train in train_dirs:
        for f in listdir(path.join(train, 'images')):
            if '_pre_disaster' in f:
                all_files.append(path.join(train, 'images', f))

    # print(len(all_files))

    file_classes = []
    for fn in all_files:
        fl = np.zeros((4,), dtype=bool)
        msk1 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)
        for c in range(1, 5):
            fl[c-1] = c in msk1
        file_classes.append(fl)
    file_classes = np.asarray(file_classes)

    cudnn.benchmark = True

    batch_size = 8
    val_batch_size = 4

    train_idxs0, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.2, random_state=42)
    print('train: ', len(train_idxs0), ' validation: ', len(val_idxs))

    train_idxs = []
    for i in train_idxs0:
        train_idxs.append(i)
        if file_classes[i, 1:].max():
            train_idxs.append(i)
        if file_classes[i, 1:3].max():
                train_idxs.append(i)
    train_idxs = np.asarray(train_idxs)

    steps_per_epoch = len(train_idxs) // batch_size
    validation_steps = len(val_idxs) // val_batch_size

    print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

    data_train = TrainDataCls(train_idxs, all_files)
    val_train = ValDataCls(val_idxs, all_files)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=6, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_train, batch_size=val_batch_size, num_workers=6, shuffle=False, pin_memory=False)

    data_loaders = {
        'train': train_data_loader,
        'val': val_data_loader
    }

    num_epochs = 25

    model = ResNet34_Unet_Double(pretrained=True, bilinear=False).cuda()
    
    model = nn.DataParallel(model).cuda()

    seg_loss = ComboLoss({'dice': 1.0, 'focal': 12.0}, per_image=False).cuda()
    ce_loss = nn.CrossEntropyLoss().cuda()

    optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=6, gamma=0.1)
    
    # Load from checkpoint
    try:
        snap_to_load = '_resnet34_unet_cls_best'
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(weight_dir, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        state_dict = model.state_dict()
        for k in state_dict:
            if k in loaded_dict and state_dict[k].size() == loaded_dict[k].size():
                state_dict[k] = loaded_dict[k]
        # loaded_dict = sd
        model.load_state_dict(state_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})"
                .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
        best_score = checkpoint['best_score']
        del loaded_dict
        del state_dict
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    except:
            # Load pretrained with localization
            snap_to_load = '_resnet34_unet_loc_best'
            print("=> loading checkpoint '{}'".format(snap_to_load))
            checkpoint = torch.load(path.join(weight_dir, snap_to_load), map_location='cpu')
            loaded_dict = checkpoint['state_dict']
            state_dict = model.state_dict()
            for k in state_dict:
                if k in loaded_dict and state_dict[k].size() == loaded_dict[k].size():
                    state_dict[k] = loaded_dict[k]
            # loaded_dict = sd
            model.load_state_dict(state_dict)
            print("loaded checkpoint '{}' (epoch {}, best_score {})"
                    .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
            best_score = 0
            del loaded_dict
            del state_dict
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()

    history = train_cls_model(model=model, data_loaders=data_loaders, optimizer=optimizer_ft, scheduler=scheduler, seg_loss=seg_loss, ce_loss=ce_loss, num_epochs=num_epochs, weight_dir=weight_dir, snapshot_name=snapshot_name, log_dir=log_dir, best_score=best_score)


if __name__=="__main__":
    parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
    arg = parser.add_argument
    arg('--mode', type=str, default='loc', help='Localization or Classification')

    args = parser.parse_args()
    
    assert args.mode in ['loc', 'cls'], f"mode '{args.mode}' was not one of 'loc' or 'cls'"

    if args.mode == 'loc':
        t0 = timeit.default_timer()
        main_loc()
        elapsed = timeit.default_timer() - t0
        print('Time: {:.3f} min'.format(elapsed / 60))
    if args.mode == 'cls':
        t0 = timeit.default_timer()
        main_cls()
        elapsed = timeit.default_timer() - t0
        print('Time: {:.3f} min'.format(elapsed / 60))
