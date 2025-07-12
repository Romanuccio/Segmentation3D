import torch
import numpy as np
# import wandb
from models.lightning_models import Unet3D
from modules.segmentation_model import Unet, PositionalUnet
from loaders.lazy_loaders import PatchDataloader
from monai.losses.focal_loss import FocalLoss
from monai.losses.dice import DiceLoss
from monai.losses.hausdorff_loss import HausdorffDTLoss
from monai.metrics import DiceMetric, compute_hausdorff_distance
from monai.transforms import (
    Compose,
    RandShiftIntensityd,
    RandAffined,
    RandZoomd,
    RandRotated,
)

from torch.nn import BCELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import sigmoid
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.trainer import Trainer

# from losses.dice_loss import DiceLoss

channel_list = [1]
# slices = 80

train = np.load("../MAMAMIA_Challenge/train_ids_v_pojeb.npy", allow_pickle=True)
validation = np.load(
    "../MAMAMIA_Challenge/validation_ids_v_pojeb.npy", allow_pickle=True
)[:100]
sets = [train, validation]

# get zipped filepaths depending on channel count
channel_images = [None] * len(sets)
for i, set in enumerate(sets):
    set_images = [None] * len(channel_list)
    for j, channel in enumerate(channel_list):
        collected_channel_list = []
        for set_id in set:
            set_channel = next(set_id.glob(f"*000{channel}.nii.gz"))
            collected_channel_list.append(set_channel)

        set_images[j] = collected_channel_list

    channel_images[i] = set_images

# get labels
labels = [None] * len(sets)
for i, set in enumerate(sets):
    set_labels = []
    for id in set:
        seg = next(id.glob("*segmentation.nii.gz"))
        set_labels.append(seg)
    labels[i] = set_labels

# get dicts
train_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(list(zip(*channel_images[0])), labels[0])
]
val_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(list(zip(*channel_images[1])), labels[1])
]

# train_transforms = Compose([
#     RandShiftIntensityd(
#         keys=["image"],  # typically only apply to image, not label
#         offsets=0.1,
#         prob=0.5
#     ),
#     RandAffined(
#         keys=["image", "label"],
#         prob=0.5,
#         rotate_range=(0.05, 0.05, 0.05),  # small rotations in radians
#         shear_range=(0.01, 0.01, 0.01),
#         translate_range=(2, 2, 2),        # in voxels
#         scale_range=(0.05, 0.05, 0.05),
#         mode=("bilinear", "nearest")
#     ),
#     RandZoomd(
#         keys=["image", "label"],
#         min_zoom=0.95,
#         max_zoom=1.05,
#         prob=0.5,
#         mode=("trilinear", "nearest")
#     ),
#     RandRotated(
#         keys=["image", "label"],
#         range_x=0.05, range_y=0.05, range_z=0.05,  # small rotations in radians
#         prob=0.5,
#         mode=("bilinear", "nearest"),
#         align_corners=True
#     )
# ])

patch_size = (128, 128, 128)
train_dataset = PatchDataloader(
    images_labels_dict=train_dicts, repeat=1, patch_size=patch_size#, threshold=0.0001
)
validation_dataset = PatchDataloader(
    images_labels_dict=val_dicts, repeat=1, patch_size=-1
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=8)
validation_loader = DataLoader(
    validation_dataset, batch_size=1, shuffle=False, num_workers=1
)

# losses
haus = HausdorffDTLoss(sigmoid=True)
dice = DiceLoss(sigmoid=True)
bce = BCEWithLogitsLoss()
focal = FocalLoss()

combined_loss = lambda predicted, label: 0.5 * (
    haus(predicted, label) + dice(predicted, label)
)

# metrics
dice_metric = DiceMetric(include_background=False)


def tresholded_dice(predicted, label, threshold=0.5):
    predicted = sigmoid(predicted) > threshold
    return dice_metric(predicted, label)


def tresholded_haus(predicted, label, threshold=0.5):
    predicted = sigmoid(predicted) > threshold
    return compute_hausdorff_distance(predicted, label, include_background=False)

stride = (64, 64, 64)

# def safe_dice_loss(pred, target):
#     shape = target.shape
#     if target.sum() <= 0.001*shape[2]*shape[3]*shape[4]:
#         return torch.tensor(0.0, device=target.device)
#     return dice(pred, target)

dice_focal_loss = lambda predicted, label: dice(predicted, label) + focal(predicted, label)

# dice model
model = Unet3D(
    model=Unet,
    depths=[3, 3, 3, 9, 3],
    channel_multipliers=[1, 2, 4, 8, 16],
    embed_dim=64,
    patch_size=patch_size,
    strides=stride,
    padding="same",
    classes=1,
    initial_LR=1e-4,
    loss=focal,
    monai=True,
    metrics=[
        ("thresholded_dice", tresholded_dice),
        ("tresholded_haus", tresholded_haus),
    ],
    beta=1,
    # final_activation=torch.nn.Sigmoid(),
    # optimizer=AdamW
)

# bce model
# model = Unet3D(
#     model=Unet,
#     depths=[3, 3, 3, 9, 3],
#     channel_multipliers=[1, 2, 4, 8, 16],
#     embed_dim=64,
#     patch_size=patch_size,
#     strides=stride,
#     padding="same",
#     classes=1,
#     initial_LR=1e-3,
#     loss=bce,
#     monai=True,
#     metrics=[
#         ("thresholded_dice", tresholded_dice),
#         ("tresholded_haus", tresholded_haus),
#     ],
#     beta=1,
#     # final_activation=torch.nn.Sigmoid(),
#     # optimizer=AdamW
# )

checkpoint_callback = ModelCheckpoint(
    monitor="val_thresholded_dice",  # Metric to monitor
    mode="max",  # 'min' for loss, 'max' for accuracy
    save_top_k=1,  # Save only the best model
    filename="best-checkpoint-{epoch:02d}-{val_thresholded_dice:.2f}",
    verbose=True,
)

wandb_logger = WandbLogger(name='gianluca_focal1e-4', project="Gianluca_Mamamia")
trainer = Trainer(
    max_epochs=300,
    callbacks=[checkpoint_callback, EarlyStopping(monitor="val_thresholded_dice", mode="max", patience=6)],
    devices=[0],
    log_every_n_steps=0,
    logger=wandb_logger,
    check_val_every_n_epoch=6,
)
trainer.fit(
    model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader
)
