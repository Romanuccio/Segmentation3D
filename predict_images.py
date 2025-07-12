import numpy as np
import torch
import SimpleITK as sitk
import os
import scipy
from scipy import ndimage
import skimage.measure as measure
from models.lightning_models import Unet3D
from modules.segmentation_model import Unet
from losses.focal_loss import FocalLoss
from torchmetrics import JaccardIndex, F1Score
from time import time
import torchio as tio

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def large_connected_domain(label):
    cd, num = measure.label(label, return_num=True, connectivity=1)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    # print(volume_sort)
    label = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    label = ndimage.binary_fill_holes(label)
    label = label.astype(np.uint8)
    return label

def normalize_img(img):

    img = img.astype(np.float32)
    img += 525.3960714344607
    img /= 528.3803170888319

    return img

path_to_img = './AIIB23_Valid_T1/img'

path_to_model = './pt_models/best_model.ckpt'

path_to_pred = './predictions_big_model'

files = os.listdir(path_to_img)

model = Unet3D.load_from_checkpoint(
    path_to_model,
    model=Unet,
    loss=FocalLoss(num_classes=2, from_logits=False, gamma=3),
    classes=2,
    metrics={
        "Jaccard": JaccardIndex(
            task="multiclass", num_classes=2, average="weighted", ignore_index=0
        ),
        "F1": F1Score(
            task="multiclass", num_classes=2, average="weighted", ignore_index=0
        ),
    },
    final_activation=torch.nn.Softmax(dim=1),
    embed_dim=64,
    strides=(96, 96, 96),
)


model.eval()
model.to(device)

tic = time()

os.makedirs(path_to_pred, exist_ok=True)

for i, file in enumerate(files):

    if os.path.exists(os.path.join(path_to_pred, file)):
        print(f'Predicting {file} ({i+1}/{len(files)})... Already predicted')
        continue

    print(f'Predicting {file} ({i+1}/{len(files)})...')
    volume = sitk.ReadImage(os.path.join(path_to_img, file))
    img = sitk.GetArrayFromImage(volume)

    img = normalize_img(img)
    img = np.expand_dims(img, axis=0)

    img = tio.ToCanonical()(img)

    img_tensor = torch.from_numpy(img).float().to(device)

    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model.predict_step(
            img_tensor,
            unpad=True,
            patch_size=(128, 128, 128),
            strides=(64, 64, 64),
            verbose=True,
            padding='same'
        )
    
    pred = torch.argmax(pred, dim=0).cpu().numpy().astype(np.int64)

    pred = large_connected_domain(pred)

    pred = sitk.GetImageFromArray(pred)
    
    pred.SetSpacing(volume.GetSpacing())
    pred.SetOrigin(volume.GetOrigin())
    pred.SetDirection(volume.GetDirection())

    assert pred.GetSize() == volume.GetSize()
    assert pred.GetSpacing() == volume.GetSpacing()
    assert pred.GetOrigin() == volume.GetOrigin()
    assert pred.GetDirection() == volume.GetDirection()

    sitk.WriteImage(pred, os.path.join(path_to_pred, file))

toc = time()

print(f'Total prediction time: {toc-tic} seconds')