import argparse
import json
import os
import vessl
from collections import defaultdict

from sklearn.metrics import log_loss
from torch import topk

import sys
from os import path

# from training import losses
from torch.nn.modules.loss import BCEWithLogitsLoss


class BinaryCrossentropy(BCEWithLogitsLoss):
    pass


import random
import traceback

import cv2
import numpy as np
import pandas as pd
import skimage.draw
from albumentations import ImageCompression, OneOf, GaussianBlur, Blur
from albumentations.pytorch.functional import img_to_tensor
from scipy.ndimage import binary_erosion, binary_dilation
from skimage import measure
from torch.utils.data import Dataset
import dlib


def prepare_bit_masks(mask):
    h, w = mask.shape
    mid_w = w // 2
    mid_h = w // 2
    masks = []
    ones = np.ones_like(mask)
    ones[:mid_h] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[mid_h:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, :mid_w] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, :mid_w] = 0
    ones[mid_h:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, mid_w:] = 0
    ones[mid_h:, :mid_w] = 0
    masks.append(ones)
    return masks


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("libs/shape_predictor_68_face_landmarks.dat")


def blackout_convex_hull(img):
    try:
        rect = detector(img)[0]
        sp = predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        outline = landmarks[[*range(17), *range(26, 16, -1)]]
        Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
        cropped_img = np.zeros(img.shape[:2], dtype=np.uint8)
        cropped_img[Y, X] = 1
        # if random.random() > 0.5:
        #     img[cropped_img == 0] = 0
        #     #leave only face
        #     return img

        y, x = measure.centroid(cropped_img)
        y = int(y)
        x = int(x)
        first = random.random() > 0.5
        if random.random() > 0.5:
            if first:
                cropped_img[:y, :] = 0
            else:
                cropped_img[y:, :] = 0
        else:
            if first:
                cropped_img[:, :x] = 0
            else:
                cropped_img[:, x:] = 0

        img[cropped_img > 0] = 0
    except Exception as e:
        pass


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def remove_eyes(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_nose(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_mouth(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[-2:]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_landmark(image, landmarks):
    if random.random() > 0.5:
        image = remove_eyes(image, landmarks)
    elif random.random() > 0.5:
        image = remove_mouth(image, landmarks)
    elif random.random() > 0.5:
        image = remove_nose(image, landmarks)
    return image


def change_padding(image, part=5):
    h, w = image.shape[:2]
    # original padding was done with 1/3 from each side, too much
    pad_h = int(((3 / 5) * h) / part)
    pad_w = int(((3 / 5) * w) / part)
    image = image[h // 5 - pad_h : -h // 5 + pad_h, w // 5 - pad_w : -w // 5 + pad_w]
    return image


def blackout_random(image, mask, label):
    binary_mask = mask > 0.4 * 255
    h, w = binary_mask.shape[:2]

    tries = 50
    current_try = 1
    while current_try < tries:
        first = random.random() < 0.5
        if random.random() < 0.5:
            pivot = random.randint(h // 2 - h // 5, h // 2 + h // 5)
            bitmap_msk = np.ones_like(binary_mask)
            if first:
                bitmap_msk[:pivot, :] = 0
            else:
                bitmap_msk[pivot:, :] = 0
        else:
            pivot = random.randint(w // 2 - w // 5, w // 2 + w // 5)
            bitmap_msk = np.ones_like(binary_mask)
            if first:
                bitmap_msk[:, :pivot] = 0
            else:
                bitmap_msk[:, pivot:] = 0

        if label < 0.5 and np.count_nonzero(image * np.expand_dims(bitmap_msk, axis=-1)) / 3 > (h * w) / 5 or np.count_nonzero(binary_mask * bitmap_msk) > 40:
            mask *= bitmap_msk
            image *= np.expand_dims(bitmap_msk, axis=-1)
            break
        current_try += 1
    return image


def blend_original(img):
    img = img.copy()
    h, w = img.shape[:2]
    rect = detector(img)
    if len(rect) == 0:
        return img
    else:
        rect = rect[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    outline = landmarks[[*range(17), *range(26, 16, -1)]]
    Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
    raw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    raw_mask[Y, X] = 1
    face = img * np.expand_dims(raw_mask, -1)

    # add warping
    h1 = random.randint(h - h // 2, h + h // 2)
    w1 = random.randint(w - w // 2, w + w // 2)
    while abs(h1 - h) < h // 3 and abs(w1 - w) < w // 3:
        h1 = random.randint(h - h // 2, h + h // 2)
        w1 = random.randint(w - w // 2, w + w // 2)
    face = cv2.resize(face, (w1, h1), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]))
    face = cv2.resize(face, (w, h), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]))

    raw_mask = binary_erosion(raw_mask, iterations=random.randint(4, 10))
    img[raw_mask, :] = face[raw_mask, :]
    if random.random() < 0.2:
        img = OneOf([GaussianBlur(), Blur()], p=0.5)(image=img)["image"]
    # image compression
    if random.random() < 0.5:
        img = ImageCompression(quality_lower=40, quality_upper=95)(image=img)["image"]
    return img


class DeepFakeClassifierDataset(Dataset):
    def __init__(
        self,
        data_path="/mnt/sota/datasets/deepfake",
        fold=0,
        label_smoothing=0.01,
        padding_part=3,
        hardcore=True,
        crops_dir="crops",
        folds_csv="folds.csv",
        normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        rotation=False,
        mode="train",
        reduce_val=True,
        oversample_real=True,
        transforms=None,
    ):
        super().__init__()
        self.data_root = data_path
        self.fold = fold
        self.folds_csv = folds_csv
        self.mode = mode
        self.rotation = rotation
        self.padding_part = padding_part
        self.hardcore = hardcore
        self.crops_dir = crops_dir
        self.label_smoothing = label_smoothing
        self.normalize = normalize
        self.transforms = transforms
        self.df = pd.read_csv(self.folds_csv)
        self.oversample_real = oversample_real
        self.reduce_val = reduce_val

    def __getitem__(self, index: int):

        while True:
            video, img_file, label, ori_video, frame, fold = self.data[index]
            try:
                if self.mode == "train":
                    label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)
                img_path = os.path.join(self.data_root, self.crops_dir, video, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                diff_path = os.path.join(self.data_root, "diffs", video, img_file[:-4] + "_diff.png")
                if os.path.exists(diff_path):
                    msk = cv2.imread(diff_path, cv2.IMREAD_GRAYSCALE)
                    if msk is not None:
                        mask = msk
                else:
                    # print("not found mask", diff_path)
                    pass
                if self.mode == "train" and self.hardcore and not self.rotation:
                    landmark_path = os.path.join(self.data_root, "landmarks", ori_video, img_file[:-4] + ".npy")
                    if os.path.exists(landmark_path) and random.random() < 0.7:
                        landmarks = np.load(landmark_path)
                        image = remove_landmark(image, landmarks)
                    elif random.random() < 0.2:
                        blackout_convex_hull(image)
                    elif random.random() < 0.1:
                        binary_mask = mask > 0.4 * 255
                        masks = prepare_bit_masks((binary_mask * 1).astype(np.uint8))
                        tries = 6
                        current_try = 1
                        while current_try < tries:
                            bitmap_msk = random.choice(masks)
                            if label < 0.5 or np.count_nonzero(mask * bitmap_msk) > 20:
                                mask *= bitmap_msk
                                image *= np.expand_dims(bitmap_msk, axis=-1)
                                break
                            current_try += 1
                if self.mode == "train" and self.padding_part > 3:
                    image = change_padding(image, self.padding_part)
                valid_label = np.count_nonzero(mask[mask > 20]) > 32 or label < 0.5
                valid_label = 1 if valid_label else 0
                rotation = 0
                if self.transforms:
                    data = self.transforms(image=image, mask=mask)
                    image = data["image"]
                    mask = data["mask"]
                if self.mode == "train" and self.hardcore and self.rotation:
                    # landmark_path = os.path.join(self.data_root, "landmarks", ori_video, img_file[:-4] + ".npy")
                    dropout = 0.8 if label > 0.5 else 0.6
                    if self.rotation:
                        dropout *= 0.7
                    elif random.random() < dropout:
                        blackout_random(image, mask, label)

                #
                # os.makedirs("../images", exist_ok=True)
                # cv2.imwrite(os.path.join("../images", video+ "_" + str(1 if label > 0.5 else 0) + "_"+img_file), image[...,::-1])

                def rot90(img: np.ndarray, factor: int) -> np.ndarray:
                    img = np.rot90(img, factor)
                    return np.ascontiguousarray(img)

                if self.mode == "train" and self.rotation:
                    rotation = random.randint(0, 3)
                    image = rot90(image, rotation)

                image = img_to_tensor(image, self.normalize)
                return {"image": image, "labels": np.array((label,)), "img_name": os.path.join(video, img_file), "valid": valid_label, "rotations": rotation}
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                print("Broken image", os.path.join(self.data_root, self.crops_dir, video, img_file))
                index = random.randint(0, len(self.data) - 1)

    def random_blackout_landmark(self, image, mask, landmarks):
        x, y = random.choice(landmarks)
        first = random.random() > 0.5
        #  crop half face either vertically or horizontally
        if random.random() > 0.5:
            # width
            if first:
                image[:, :x] = 0
                mask[:, :x] = 0
            else:
                image[:, x:] = 0
                mask[:, x:] = 0
        else:
            # height
            if first:
                image[:y, :] = 0
                mask[:y, :] = 0
            else:
                image[y:, :] = 0
                mask[y:, :] = 0

    def reset(self, epoch, seed):
        self.data = self._prepare_data(epoch, seed)

    def __len__(self) -> int:
        return len(self.data)

    def _prepare_data(self, epoch, seed):
        df = self.df
        if self.mode == "train":
            rows = df[df["fold"] != self.fold]
        else:
            rows = df[df["fold"] == self.fold]
        seed = (epoch + 1) * seed
        # if self.oversample_real:
        #     rows = self._oversample(rows, seed)
        if self.mode == "val" and self.reduce_val:
            # every 2nd frame, to speed up validation
            rows = rows[rows["frame"] % 20 == 0]
            # another option is to use public validation set
            # rows = rows[rows["video"].isin(PUBLIC_SET)]

        print("real {} fakes {} mode {}".format(len(rows[rows["label"] == 0]), len(rows[rows["label"] == 1]), self.mode))
        data = rows.values

        np.random.seed(seed)
        np.random.shuffle(data)
        return data

    def _oversample(self, rows: pd.DataFrame, seed):
        real = rows[rows["label"] == 0]
        fakes = rows[rows["label"] == 1]
        num_real = real["video"].count()
        if self.mode == "train":
            fakes = fakes.sample(n=num_real, replace=False, random_state=seed)
        return pd.concat([real, fakes])


from torch import nn


class WeightedLosses(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, *input: Any, **kwargs: Any):
        cum_loss = 0
        for loss, w in zip(self.losses, self.weights):
            cum_loss += w * loss.forward(*input, **kwargs)
        return cum_loss


# from training.tools.config import load_config

import json

DEFAULTS = {
    "network": "dpn",
    "encoder": "dpn92",
    "model_params": {},
    "optimizer": {
        "batch_size": 32,
        "type": "SGD",  # supported: SGD, Adam
        "momentum": 0.9,
        "weight_decay": 0,
        "clip": 1.0,
        "learning_rate": 0.1,
        "classifier_lr": -1,
        "nesterov": True,
        "schedule": {"type": "constant", "mode": "epoch", "epochs": 10, "params": {}},  # supported: constant, step, multistep, exponential, linear, poly  # supported: epoch, step
    },
    "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = json.load(fd)
    _merge(defaults, config)
    return config


from training.tools.utils import create_optimizer, AverageMeter

import cv2

# from apex.optimizers import FusedAdam, FusedSGD
# from timm.optim import AdamW
from torch import optim
from torch.optim.rmsprop import RMSprop
from torch.optim.adamw import AdamW

from training.tools.schedulers import ExponentialLRScheduler, PolyLR, LRStepScheduler

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


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


def create_optimizer(optimizer_config, lr, wd, model, master_params=None):
    """Creates optimizer and schedule from configuration

    Parameters
    ----------
    optimizer_config : dict
        Dictionary containing the configuration options for the optimizer.
    model : Model
        The network model.

    Returns
    -------
    optimizer : Optimizer
        The optimizer.
    scheduler : LRScheduler
        The learning rate scheduler.
    """
    # if optimizer_config.get("classifier_lr", -1) != -1:
    #     # Separate classifier parameters from all others
    #     net_params = []
    #     classifier_params = []
    #     for k, v in model.named_parameters():
    #         if not v.requires_grad:
    #             continue
    #         if k.find("encoder") != -1:
    #             net_params.append(v)
    #         else:
    #             classifier_params.append(v)
    #     params = [
    #         {"params": net_params},
    #         {"params": classifier_params, "lr": optimizer_config["classifier_lr"]},
    #     ]
    # else:
    if master_params:
        params = master_params
    else:
        params = model.parameters()

    mmt = 0.9
    nes = True
    if optimizer_config == "SGD":
        optimizer = optim.SGD(params, lr=lr, momentum=mmt, weight_decay=wd, nesterov=nes)
    # elif optimizer_config == "FusedSGD":
    #     optimizer = FusedSGD(params, lr=lr, momentum=mmt, weight_decay=wd, nesterov=nes)
    elif optimizer_config == "Adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
    # elif optimizer_config == "FusedAdam":
    #     optimizer = FusedAdam(params, lr=lr, weight_decay=wd)
    elif optimizer_config == "AdamW":
        optimizer = AdamW(params, lr=lr, weight_decay=wd)
    elif optimizer_config == "RmsProp":
        optimizer = RMSprop(params, lr=lr, weight_decay=wd)
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config))

    scheduler = PolyLR(optimizer, 100500)

    return optimizer, scheduler


# from training.transforms.albu import IsotropicResize

import random

import cv2
import numpy as np
from albumentations import DualTransform, ImageOnlyTransform

# from albumentations.augmentations.functional import crop
from albumentations.augmentations.crops.transforms import Crop


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down, interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


# from training.zoo import classifiers
from functools import partial

import numpy as np
import torch
from timm.models.efficientnet import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, tf_efficientnet_b6_ns, tf_efficientnet_b7_ns
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d

encoder_params = {
    "tf_efficientnet_b3_ns": {"features": 1536, "init_op": partial(tf_efficientnet_b3_ns, pretrained=True, drop_path_rate=0.2)},
    "tf_efficientnet_b2_ns": {"features": 1408, "init_op": partial(tf_efficientnet_b2_ns, pretrained=False, drop_path_rate=0.2)},
    "tf_efficientnet_b4_ns": {"features": 1792, "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.5)},
    "tf_efficientnet_b5_ns": {"features": 2048, "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.2)},
    "tf_efficientnet_b4_ns_03d": {"features": 1792, "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.3)},
    "tf_efficientnet_b5_ns_03d": {"features": 2048, "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.3)},
    "tf_efficientnet_b5_ns_04d": {"features": 2048, "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.4)},
    "tf_efficientnet_b6_ns": {"features": 2304, "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2)},
    "tf_efficientnet_b7_ns": {"features": 2560, "init_op": partial(tf_efficientnet_b7_ns, pretrained=True, drop_path_rate=0.2)},
    "tf_efficientnet_b6_ns_04d": {"features": 2304, "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.4)},
}


class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import numpy as np
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur

# from apex.parallel import DistributedDataParallel, convert_syncbn_model
from tensorboardX import SummaryWriter

# from apex import amp

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist

torch.backends.cudnn.benchmark = True


def create_train_transforms(size=300):
    return Compose(
        [
            ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            GaussNoise(p=0.1),
            GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            OneOf(
                [
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                ],
                p=1,
            ),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
    )


def create_val_transforms(size=300):
    return Compose(
        [
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ]
    )


def main():
    print(path.dirname(path.dirname(path.abspath(__file__))))

    parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
    arg = parser.add_argument
    arg("--config", metavar="CONFIG_FILE", help="path to configuration file")
    arg("--workers", type=int, default=6, help="number of cpu threads to use")
    arg("--gpu", type=str, default="0", help="List of GPUs for parallel training, e.g. 0,1,2,3")
    arg("--output-dir", type=str, default="weights/")
    arg("--resume", type=str, default="")
    arg("--fold", type=int, default=0)
    arg("--prefix", type=str, default="classifier_")
    arg("--data-dir", type=str, default="/mnt/sota/datasets/deepfake")
    arg("--folds-csv", type=str, default="folds.csv")
    arg("--crops-dir", type=str, default="crops")
    arg("--label-smoothing", type=float, default=0.01)
    arg("--logdir", type=str, default="logs")
    arg("--zero-score", action="store_true", default=False)
    arg("--from-zero", action="store_true", default=False)
    arg("--distributed", action="store_true", default=False)
    arg("--freeze-epochs", type=int, default=0)
    arg("--local_rank", default=0, type=int)
    arg("--seed", default=777, type=int)
    arg("--padding-part", default=3, type=int)
    arg("--opt-level", default="O1", type=str)
    arg("--test_every", type=int, default=1)
    arg("--no-oversample", action="store_true")
    arg("--no-hardcore", action="store_true")
    arg("--only-changed-frames", action="store_true")

    optimizer = str(os.environ.get("optimizer", "SGD"))
    batch_size = int(os.environ.get("batch_size", 1))
    lr = float(os.environ.get("learning_rate", 0.01))
    wd = float(os.environ.get("weight_decay", 0.01))
    print(f"hyper param == {optimizer} == {batch_size} == {lr} == {wd}")

    # arg("--opt", default="sgd", type=str)
    # arg("--batch_size", default=1, type=int)
    # arg("--lr", default=0.1, type=float)
    # arg("--wd", default=0.01, type=float)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.distributed:
        # torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True

    conf = load_config(args.config)
    model = DeepFakeClassifier(encoder=conf["encoder"])

    model = model.cuda()
    # if args.distributed:
    #     model = convert_syncbn_model(model)
    ohem = conf.get("ohem_samples", None)
    reduction = "mean"
    if ohem:
        reduction = "none"
    loss_fn = []
    weights = []
    for loss_name, weight in conf["losses"].items():
        loss_fn.append(BinaryCrossentropy(reduction=reduction).cuda())
        weights.append(weight)
    loss = WeightedLosses(loss_fn, weights)
    loss_functions = {"classifier_loss": loss}
    # optimizer, scheduler = create_optimizer(conf["optimizer"], model)
    optimizer, scheduler = create_optimizer(optimizer, lr, wd, model)
    bce_best = 100
    start_epoch = 0
    # batch_size = conf["optimizer"]["batch_size"]

    data_train = DeepFakeClassifierDataset(
        mode="train",
        oversample_real=not args.no_oversample,
        fold=args.fold,
        padding_part=args.padding_part,
        hardcore=not args.no_hardcore,
        crops_dir=args.crops_dir,
        data_path=args.data_dir,
        label_smoothing=args.label_smoothing,
        folds_csv=args.folds_csv,
        transforms=create_train_transforms(conf["size"]),
        normalize=conf.get("normalize", None),
    )
    data_val = DeepFakeClassifierDataset(
        mode="val",
        fold=args.fold,
        padding_part=args.padding_part,
        crops_dir=args.crops_dir,
        data_path=args.data_dir,
        folds_csv=args.folds_csv,
        transforms=create_val_transforms(conf["size"]),
        normalize=conf.get("normalize", None),
    )
    val_data_loader = DataLoader(data_val, batch_size=batch_size * 2, num_workers=args.workers, shuffle=False, pin_memory=False)
    os.makedirs(args.logdir, exist_ok=True)
    summary_writer = SummaryWriter(args.logdir + "/" + conf.get("prefix", args.prefix) + conf["encoder"] + "_" + str(args.fold))
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="gpu")
            state_dict = checkpoint["state_dict"]
            state_dict = {k[7:]: w for k, w in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            if not args.from_zero:
                start_epoch = checkpoint["epoch"]
                if not args.zero_score:
                    bce_best = checkpoint.get("bce_best", 0)
            print("=> loaded checkpoint '{}' (epoch {}, bce_best {})".format(args.resume, checkpoint["epoch"], checkpoint["bce_best"]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.from_zero:
        start_epoch = 0
    current_epoch = start_epoch

    # if conf["fp16"]:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale="dynamic")

    snapshot_name = "{}{}_{}_{}".format(conf.get("prefix", args.prefix), conf["network"], conf["encoder"], args.fold)

    # if args.distributed:
    #     model = DistributedDataParallel(model, delay_allreduce=True)
    # else:
    model = DataParallel(model).cuda()

    data_val.reset(1, args.seed)
    max_epochs = conf["optimizer"]["schedule"]["epochs"]
    # mlops init
    # if args.local_rank == 0:
    # wandb.init(project="dfdc-deepfake-detection", entity="greenteaboom")
    # wandb.config = {"annotate": "0210-dfdc-vanilla", "epochs": max_epochs, "batch_size": batch_size}
    # wandb.run.name = "0210-dfdc-vanilla"

    vessl.init(organization="greentea", project="dfdc-deepfake-detection")
    for epoch in range(start_epoch, max_epochs):
        data_train.reset(epoch, args.seed)
        train_sampler = None
        # if args.distributed:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
        #     train_sampler.set_epoch(epoch)
        if epoch < args.freeze_epochs:
            print("Freezing encoder!!!")
            model.module.encoder.eval()
            for p in model.module.encoder.parameters():
                p.requires_grad = False
        else:
            model.module.encoder.train()
            for p in model.module.encoder.parameters():
                p.requires_grad = True

        train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=args.workers, shuffle=train_sampler is None, sampler=train_sampler, pin_memory=False, drop_last=True)

        train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf, args.local_rank, args.only_changed_frames)
        model = model.eval()

        # if args.local_rank == 0:
        torch.save(
            {
                "epoch": current_epoch + 1,
                "state_dict": model.state_dict(),
                "bce_best": bce_best,
            },
            args.output_dir + "/" + snapshot_name + "_last",
        )
        torch.save(
            {
                "epoch": current_epoch + 1,
                "state_dict": model.state_dict(),
                "bce_best": bce_best,
            },
            args.output_dir + snapshot_name + "_{}".format(current_epoch),
        )
        if (epoch + 1) % args.test_every == 0:
            bce_best = evaluate_val(args, val_data_loader, bce_best, model, snapshot_name=snapshot_name, current_epoch=current_epoch, summary_writer=summary_writer)
        current_epoch += 1


def evaluate_val(args, data_val, bce_best, model, snapshot_name, current_epoch, summary_writer):
    print("Test phase")
    model = model.eval()

    bce, probs, targets = validate(model, data_loader=data_val, epoch=current_epoch, local_rank=args.local_rank)
    # if args.local_rank == 0:
    summary_writer.add_scalar("val/bce", float(bce), global_step=current_epoch)
    if bce < bce_best:
        print("Epoch {} improved from {} to {}".format(current_epoch, bce_best, bce))
        if args.output_dir is not None:
            torch.save(
                {
                    "epoch": current_epoch + 1,
                    "state_dict": model.state_dict(),
                    "bce_best": bce,
                },
                args.output_dir + snapshot_name + "_best_dice",
            )
        bce_best = bce
        with open("predictions_{}.json".format(args.fold), "w") as f:
            json.dump({"probs": probs, "targets": targets}, f)
    torch.save(
        {
            "epoch": current_epoch + 1,
            "state_dict": model.state_dict(),
            "bce_best": bce_best,
        },
        args.output_dir + snapshot_name + "_last",
    )
    print("Epoch: {} bce: {}, bce_best: {}".format(current_epoch, bce, bce_best))
    return bce_best


def validate(net, data_loader, prefix="", epoch=-1, local_rank=-1):
    probs = defaultdict(list)
    targets = defaultdict(list)

    with torch.no_grad():
        for sample in tqdm(data_loader):
            imgs = sample["image"].cuda()
            img_names = sample["img_name"]
            labels = sample["labels"].cuda().float()
            out = net(imgs)
            labels = labels.cpu().numpy()
            preds = torch.sigmoid(out).cpu().numpy()
            for i in range(out.shape[0]):
                video, img_id = img_names[i].split("/")
                probs[video].append(preds[i].tolist())
                targets[video].append(labels[i].tolist())
    data_x = []
    data_y = []
    for vid, score in probs.items():
        score = np.array(score)
        lbl = targets[vid]

        score = np.mean(score)
        lbl = np.mean(lbl)
        data_x.append(score)
        data_y.append(lbl)
    y = np.array(data_y)
    x = np.array(data_x)
    fake_idx = y > 0.1
    real_idx = y < 0.1
    prediction = x > 0.5
    valid_accuracy = np.average((prediction == y))

    fake_loss = log_loss(y[fake_idx], x[fake_idx], labels=[0, 1])
    real_loss = log_loss(y[real_idx], x[real_idx], labels=[0, 1])
    print("{}fake_loss".format(prefix), fake_loss)
    print("{}real_loss".format(prefix), real_loss)

    if local_rank == 0:
        # wandb.log({"val_fake_loss": fake_loss, "val_real_loss": real_loss, "val_loss": (fake_loss + real_loss) / 2, "val_accuracy": valid_accuracy})
        vessl.log(step=epoch, payload={"val_fake_loss": fake_loss, "val_real_loss": real_loss, "val_loss": (fake_loss + real_loss) / 2, "val_accuracy": valid_accuracy})

    return (fake_loss + real_loss) / 2, probs, targets


def train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf, local_rank, only_valid):
    losses = AverageMeter()
    fake_losses = AverageMeter()
    real_losses = AverageMeter()
    max_iters = conf["batches_per_epoch"]
    print("training epoch {}".format(current_epoch))
    model.train()
    pbar = tqdm(enumerate(train_data_loader), total=max_iters, desc="Epoch {}".format(current_epoch), ncols=0)
    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(current_epoch)
    for i, sample in pbar:
        imgs = sample["image"].cuda()
        labels = sample["labels"].cuda().float()
        out_labels = model(imgs)
        if only_valid:
            valid_idx = sample["valid"].cuda().float() > 0
            out_labels = out_labels[valid_idx]
            labels = labels[valid_idx]
            if labels.size(0) == 0:
                continue

        fake_loss = 0
        real_loss = 0
        fake_idx = labels > 0.5
        real_idx = labels <= 0.5

        ohem = conf.get("ohem_samples", None)
        if torch.sum(fake_idx * 1) > 0:
            fake_loss = loss_functions["classifier_loss"](out_labels[fake_idx], labels[fake_idx])
        if torch.sum(real_idx * 1) > 0:
            real_loss = loss_functions["classifier_loss"](out_labels[real_idx], labels[real_idx])
        if ohem:
            fake_loss = topk(fake_loss, k=min(ohem, fake_loss.size(0)), sorted=False)[0].mean()
            real_loss = topk(real_loss, k=min(ohem, real_loss.size(0)), sorted=False)[0].mean()

        loss = (fake_loss + real_loss) / 2
        losses.update(loss.item(), imgs.size(0))
        fake_losses.update(0 if fake_loss == 0 else fake_loss.item(), imgs.size(0))
        real_losses.update(0 if real_loss == 0 else real_loss.item(), imgs.size(0))

        optimizer.zero_grad()
        pbar.set_postfix({"lr": float(scheduler.get_lr()[-1]), "epoch": current_epoch, "loss": losses.avg, "fake_loss": fake_losses.avg, "real_loss": real_losses.avg})

        # if local_rank == 0:
        #     wandb.log({"fake_loss": fake_loss, "real_loss": real_loss, "loss": (fake_loss + real_loss) / 2})

        # if conf["fp16"]:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
        optimizer.step()
        torch.cuda.synchronize()
        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * max_iters)
        if i == max_iters - 1:
            break
    pbar.close()
    if local_rank == 0:
        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group["lr"]
            summary_writer.add_scalar("group{}/lr".format(idx), float(lr), global_step=current_epoch)
        summary_writer.add_scalar("train/loss", float(losses.avg), global_step=current_epoch)
        # log per epoch
        vessl.log(step=current_epoch, payload={"fake_loss": float(fake_losses.avg), "real_loss": float(real_losses.avg), "loss": float(losses.avg)})


if __name__ == "__main__":
    main()
