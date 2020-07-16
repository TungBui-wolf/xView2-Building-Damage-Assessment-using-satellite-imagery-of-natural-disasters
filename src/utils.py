import numpy as np
import cv2

#### Augmentations
def shift_image(img, shift_pnt):
    M = np.float32([[1, 0, shift_pnt[0]], [0, 1, shift_pnt[1]]])
    res = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
    return res


def rotate_image(image, angle, scale, rot_pnt):
    rot_mat = cv2.getRotationMatrix2D(rot_pnt, angle, scale)
    result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) #INTER_NEAREST
    return result


def gauss_noise(img, var=30):
    row, col, ch = img.shape
    mean = var
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss = (gauss - np.min(gauss)).astype(np.uint8)
    return np.clip(img.astype(np.int32) + gauss, 0, 255).astype('uint8')


def clahe(img, clipLimit=2.0, tileGridSize=(5,5)):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2RGB)
    return img_output


def _blend(img1, img2, alpha):
    return np.clip(img1 * alpha + (1 - alpha) * img2, 0, 255).astype('uint8')


_alpha = np.asarray([0.114, 0.587, 0.299]).reshape((1, 1, 3))
def _grayscale(img):
    return np.sum(_alpha * img, axis=2, keepdims=True)


def saturation(img, alpha):
    gs = _grayscale(img)
    return _blend(img, gs, alpha)


def brightness(img, alpha):
    gs = np.zeros_like(img)
    return _blend(img, gs, alpha)


def contrast(img, alpha):
    gs = _grayscale(img)
    gs = np.repeat(gs.mean(), 3)
    return _blend(img, gs, alpha)


def change_hsv(img, h, s, v):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(int)
    hsv[:,:,0] += h
    hsv[:,:,0] = np.clip(hsv[:,:,0], 0, 255)
    hsv[:,:,1] += s
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    hsv[:,:,2] += v
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    hsv = hsv.astype('uint8')
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def shift_channels(img, b_shift, g_shift, r_shift):
    img = img.astype(int)
    img[:,:,0] += b_shift
    img[:,:,0] = np.clip(img[:,:,0], 0, 255)
    img[:,:,1] += g_shift
    img[:,:,1] = np.clip(img[:,:,1], 0, 255)
    img[:,:,2] += r_shift
    img[:,:,2] = np.clip(img[:,:,2], 0, 255)
    img = img.astype('uint8')
    return img
    
def invert(img):
    return 255 - img

def channel_shuffle(img):
    ch_arr = [0, 1, 2]
    np.random.shuffle(ch_arr)
    img = img[..., ch_arr]
    return img
    
    
####### Utils

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x


### Split image and merge again

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

def split_image(img, overlap=0):
    img_h, img_w = img.shape[0], img.shape[1]

    split_width = 256
    split_height = 256

    X_points = start_points(img_w, split_width, overlap=overlap)
    Y_points = start_points(img_h, split_height, overlap=overlap)

    count = 0
    
    splited_imgs = []
    for i in Y_points:
        for j in X_points:
            split_img = img[i:i+split_height, j:j+split_width]
            splited_imgs.append(split_img)
            count += 1
            
    return splited_imgs

def merge_image(splited_imgs, new_img_w=1024, new_img_h=1024, overlap=0):
    per_img = splited_imgs[0]
    if len(per_img.shape) > 2:
        size = [new_img_h, new_img_w, per_img.shape[2]]
    else:
        size = [new_img_h, new_img_w]
    img = np.zeros(size, dtype=np.uint8)
    
    X_points = start_points(new_img_w, per_img.shape[0], overlap=overlap)
    Y_points = start_points(new_img_h, per_img.shape[1], overlap=overlap)
    
    count = 0
    for i in Y_points:
        for j in X_points:
            img[i:i+per_img.shape[1], j:j+per_img.shape[0]] = splited_imgs[count] 
            count += 1
            
    return img
