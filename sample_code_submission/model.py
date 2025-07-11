# ==================== MAMA-MIA CHALLENGE SAMPLE SUBMISSION ====================
#
# This is the official sample submission script for the **MAMA-MIA Challenge**, 
# covering both tasks:
#
#   1. Primary Tumour Segmentation (Task 1)
#   2. Treatment Response Classification (Task 2)
#
# ----------------------------- SUBMISSION FORMAT -----------------------------
# Participants must implement a class `Model` with one or two of these methods:
#
#   - `predict_segmentation(output_dir)`: required for Task 1
#       > Must output NIfTI files named `{patient_id}.nii.gz` in a folder
#       > called `pred_segmentations/`
#
#   - `predict_classification(output_dir)`: required for Task 2
#       > Must output a CSV file `predictions.csv` in `output_dir` with columns:
#           - `patient_id`: patient identifier
#           - `pcr`: binary label (1 = pCR, 0 = non-pCR)
#           - `score`: predicted probability (flaot between 0 and 1)
#
#   - `predict_classification(output_dir)`: if a single model handles both tasks
#       > Must output NIfTI files named `{patient_id}.nii.gz` in a folder
#       > called `pred_segmentations/`
#       > Must output a CSV file `predictions.csv` in `output_dir` with columns:
#           - `patient_id`: patient identifier
#           - `pcr`: binary label (1 = pCR, 0 = non-pCR)
#           - `score`: predicted probability (flaot between 0 and 1)
#
# You can submit:
#   - Only Task 1 (implement `predict_segmentation`)
#   - Only Task 2 (implement `predict_classification`)
#   - Both Tasks (implement both methods independently or define `predict_segmentation_and_classification` method)
#
# ------------------------ SANITY-CHECK PHASE ------------------------
#
# ✅ Before entering the validation or test phases, participants must pass the **Sanity-Check phase**.
#   - This phase uses **4 samples from the test set** to ensure your submission pipeline runs correctly.
#   - Submissions in this phase are **not scored**, but must complete successfully within the **20-minute timeout limit**.
#   - Use this phase to debug your pipeline and verify output formats without impacting your submission quota.
#
# 💡 This helps avoid wasted submissions on later phases due to technical errors.
#
# ------------------------ SUBMISSION LIMITATIONS ------------------------
#
# ⚠️ Submission limits are strictly enforced per team:
#   - **One submission per day**
#   - **Up to 15 submissions total on the validation set**
#   - **Only 1 final submission on the test set**
#
# Plan your development and testing accordingly to avoid exhausting submissions prematurely.
#
# ----------------------------- RUNTIME AND RESOURCES -----------------------------
#
# > ⚠️ VERY IMPORTANT: Each image has a **timeout of 5 minutes** on the compute worker.
#   - **Validation Set**: 58 patients → total budget ≈ 290 minutes
#   - **Test Set**: 516 patients → total budget ≈ 2580 minutes
#
# > The compute worker environment is based on the Docker image:
#       `lgarrucho/codabench-gpu:latest`
#
# > You can install additional dependencies via `requirements.txt`.
#   Please ensure all required packages are listed there.
#
# ----------------------------- SEGMENTATION DETAILS -----------------------------
#
# This example uses `nnUNet v2`, which is compatible with the GPU compute worker.
# Note the following nnUNet-specific constraints:
#
# ✅ `predict_from_files_sequential` MUST be used for inference.
#     - This is because nnUNet’s multiprocessing is incompatible with the compute container.
#     - In our environment, a single fold prediction using `predict_from_files_sequential` 
#       takes approximately **1 minute per patient**.
#
# ✅ The model uses **fold 0 only** to reduce runtime.
# 
# ✅ Predictions are post-processed by applying a breast bounding box mask using 
#    metadata provided in the per-patient JSON file.
#
# ----------------------------- CLASSIFICATION DETAILS -----------------------------
#
# If using predicted segmentations for Task 2 classification:
#   - Save them in `self.predicted_segmentations` inside `predict_segmentation()`
#   - You can reuse them in `predict_classification()`
#   - Or perform Task 1 and Task 2 inside `predict_segmentation_and_classification`
#
# ----------------------------- DATASET INTERFACE -----------------------------
# The provided `dataset` object is a `RestrictedDataset` instance and includes:
#
#   - `dataset.get_patient_id_list() → list[str]`  
#         Patient IDs for current split (val/test)
#
#   - `dataset.get_dce_mri_path_list(patient_id) → list[str]`  
#         Paths to all image channels (typically pre and post contrast)
#         - iamge_list[0] corresponds to the pre-contrast image path
#         - iamge_list[1] corresponds to the first post-contrast image path and so on
#
#   - `dataset.read_json_file(patient_id) → dict`  
#         Metadata dictionary per patient, including:
#         - breast bounding box (`primary_lesion.breast_coordinates`)
#         - scanner metadata (`imaging_data`), etc...
#
# Example JSON structure:
# {
#   "patient_id": "XXX_XXX_SXXXX",
#   "primary_lesion": {
#     "breast_coordinates": {
#         "x_min": 1, "x_max": 158,
#         "y_min": 6, "y_max": 276,
#         "z_min": 1, "z_max": 176
#     }
#   },
#   "imaging_data": {
#     "bilateral": true,
#     "dataset": "HOSPITAL_X",
#     "site": "HOSPITAL_X",
#     "scanner_manufacturer": "SIEMENS",
#     "scanner_model": "Aera",
#     "field_strength": 1.5,
#     "echo_time": 1.11,
#     "repetition_time": 3.35
#   }
# }
#
# ----------------------------- RECOMMENDATIONS -----------------------------
# ✅ We recommend to always test your submission first in the Sanity-Check Phase.
#    As in Codabench the phases need to be sequential and they cannot run in parallel,
#    we will open a secondary MAMA-MIA Challenge Codabench page with a permanen Sanity-Check phase.
#   That way you won't lose submission trials to the validation or even wore, the test set.
# ✅ We recommend testing your solution locally and measuring execution time per image.
# ✅ Use lightweight models or limit folds if running nnUNet.
# ✅ Keep all file paths, patient IDs, and formats **exactly** as specified.
# ✅ Ensure your output folders are created correctly (e.g. `pred_segmentations/`)
# ✅ For faster runtime, only select a single image for segmentation.
#
# ------------------------ COPYRIGHT ------------------------------------------
#
# © 2025 Lidia Garrucho. All rights reserved.
# Unauthorized use, reproduction, or distribution of any part of this competition's 
# materials is prohibited without explicit permission.
#
# ------------------------------------------------------------------------------

# === MANDATORY IMPORTS ===
import os
import pandas as pd
import shutil

# === OPTIONAL IMPORTS: only needed if you modify or extend nnUNet input/output handling ===
# You can remove unused imports above if not needed for your solution
import numpy as np
import torch
import SimpleITK as sitk
from pathlib import Path
from monai.transforms import (
    Activations,
    AsDiscrete,
    LoadImage,
    Resize,
    Compose,
    Spacing,
    EnsureChannelFirst
)
#gianluca
from models.lightning_models import Unet3D
from modules.segmentation_model import Unet

def resample_sitk(image_sitk, new_spacing=None, new_size=None,
                   interpolator=sitk.sitkLinear, tol=0.00001):
    # Get original settings
    original_size = image_sitk.GetSize()
    original_spacing = image_sitk.GetSpacing()
   
    # ITK can only do 3D images
    if len(original_size) == 2:
        original_size = original_size + (1, )
    if len(original_spacing) == 2:
        original_spacing = original_spacing + (1.0, )

    if new_size is None:
        # Compute output size
        new_size = [round(original_size[0]*(original_spacing[0] + tol) / new_spacing[0]),
                    round(original_size[1]*(original_spacing[0] + tol) / new_spacing[1]),
                    round(original_size[2]*(original_spacing[2] + tol) / new_spacing[2])]

    if new_spacing is None:
        # Compute output spacing
        tol = 0
        new_spacing = [original_size[0]*(original_spacing[0] + tol)/new_size[0],
                       original_size[1]*(original_spacing[0] + tol)/new_size[1],
                       original_size[2]*(original_spacing[2] + tol)/new_size[2]]

    # Set and execute the filter
    ResampleFilter = sitk.ResampleImageFilter()
    ResampleFilter.SetInterpolator(interpolator)
    ResampleFilter.SetOutputSpacing(new_spacing)
    ResampleFilter.SetSize(np.array(new_size, dtype='int').tolist())
    ResampleFilter.SetOutputDirection(image_sitk.GetDirection())
    ResampleFilter.SetOutputOrigin(image_sitk.GetOrigin())
    ResampleFilter.SetOutputPixelType(image_sitk.GetPixelID())
    ResampleFilter.SetTransform(sitk.Transform())
    try:
        resampled_image_sitk = ResampleFilter.Execute(image_sitk)
    except RuntimeError:
        # Assume the error is due to the direction determinant being 0
        # Solution: simply set a correct direction
        # print('Bad output direction in resampling, resetting direction.')
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        ResampleFilter.SetOutputDirection(direction)
        image_sitk.SetDirection(direction)
        resampled_image_sitk = ResampleFilter.Execute(image_sitk)

    return resampled_image_sitk


class Model:
    def __init__(self, dataset):
        """
        Initializes the model with the restricted dataset.
        
        Args:
            dataset (RestrictedDataset): Preloaded dataset instance with controlled access.
        """
        # MANDATOR
        self.dataset = dataset  # Restricted Access to Private Dataset
        self.predicted_segmentations = None  # Optional: stores path to predicted segmentations
        # Only if using nnUNetv2, you can define here any other variables
        # self.dataset_id = "105"  # Dataset ID must match your folder structure
        # self.config = "3d_fullres" # nnUNetv2 configuration
        

    def predict_segmentation(self, output_dir):
        """
        Task 1 — Predict tumor segmentation with nnUNetv2.
        You MUST define this method if participating in Task 1.

        Args:
            output_dir (str): Directory where predictions will be stored.

        Returns:
            str: Path to folder with predicted segmentation masks.
        """
        
        checkpoint_path = '/app/ingested_program/best-checkpoint-epoch=131-val_thresholded_dice=0.59.ckpt'
        device = torch.device(0)
        stride = (64, 64, 64)
        patch_size = (128, 128, 128)
        model = Unet3D.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=Unet,
            depths=[3, 3, 3, 9, 3],
            channel_multipliers=[1, 2, 4, 8, 16],
            embed_dim=64,
            patch_size=patch_size,
            strides=stride,
            loss=None,
            padding="same",
            classes=1,
            beta=1,
            initial_LR=1e-4,
            monai=True,
            # metrics=[('thresholded_dice', tresholded_dice), ('tresholded_haus', tresholded_haus)],
            # final_activation=torch.nn.Sigmoid(),
            # optimizer=AdamW
        ).to(device)
        model.eval()

        # === Participants can modify how they prepare input ===
        patient_ids = self.dataset.get_patient_id_list()
        threshold = AsDiscrete(threshold=0.5)
        sigmoid = Activations(sigmoid=True)
        
       # === Final output folder (MANDATORY name) ===
        output_dir_final = os.path.join(output_dir, 'pred_segmentations')
        os.makedirs(output_dir_final, exist_ok=True)
        
        for patient_id in patient_ids:
            data = self.dataset.read_json_file(patient_id)
            coords = data.get("primary_lesion", {}).get("breast_coordinates", {})

            required_keys = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
            if not all(key in coords for key in required_keys):
                print(f"Missing coordinates in")
                
            x_min, x_max = coords["x_min"], coords["x_max"]
            y_min, y_max = coords["y_min"], coords["y_max"]
            z_min, z_max = coords["z_min"], coords["z_max"]

            # get data
            images = self.dataset.get_dce_mri_path_list(patient_id)
            # precontrast_image = images[0]
            first_post_contrast_image = images[1]
            
            ### read raw image
            # raw_image0 = sitk.ReadImage(precontrast_image) #read the image, it's W H D
            raw_image1 = sitk.ReadImage(first_post_contrast_image) #read the image, it's W H D
            
            ### bounding box extraction
            start_index = (z_min, y_min, x_min) # bounding box is in NP/torch format D H W => have to reverse order
            size = (z_max - z_min, y_max - y_min, x_max - x_min)
            
            # box0 = sitk.Extract(raw_image0, size, start_index)
            # box0 = resample_sitk(box0, new_spacing=(1,1,1))
            
            box1 = sitk.Extract(raw_image1, size, start_index)
            box1_shape = box1.GetSize()
            box1 = resample_sitk(box1, new_spacing=(1,1,1))
            
            ### moving down to model input level
            # box_array0 = torch.tensor(sitk.GetArrayFromImage(box0)).to(device)[None, ...]
            # box_array0 = box_array0.transpose(1, 3)
            # image0 = transforms(box_array0)[None, ...]
            
            box_array1 = torch.tensor(sitk.GetArrayFromImage(box1)).to(device)[None, ...]
            box_array1 = box_array1.transpose(1, 3)
            image1 = box_array1[None, ...]
            
            ### obtain segmentation
            # input_image = torch.cat((image0, image1), dim=1)
            with torch.no_grad():
                seg = model.predict_step(
                    data,
                    patch_size=patch_size,
                    strides=stride,
                    padding="same", unpad=True, verbose=True
                )
                seg = threshold(sigmoid(seg))
                
                # seg = seg[:, 1:2, ...]
                seg = torch.round(seg).to(torch.uint8)
            
            ### resize to extracted bounding box dimensions back from model dimensions
            box_resize = Resize(box1_shape, mode='nearest')
            seg = box_resize(seg[0, ...])
            seg = seg[0, ...]
            seg = seg.int()
            seg = seg.detach().cpu()
            
            ### sitk fuckery
            imgseg = sitk.Image(raw_image1.GetSize(), raw_image1.GetPixelID())
            zero_array = sitk.GetArrayFromImage(imgseg)
            zero_array[x_min:x_max, y_min:y_max, z_min:z_max] = seg.transpose(0, 2)
            imgseg = sitk.GetImageFromArray(zero_array)
            imgseg.SetSpacing(raw_image1.GetSpacing())
            imgseg.SetOrigin(raw_image1.GetOrigin())
            imgseg.SetDirection(raw_image1.GetDirection())
            imgseg.SetSpacing(raw_image1.GetSpacing())
            
            final_seg_path = os.path.join(output_dir_final, f"{patient_id}.nii.gz")
            sitk.WriteImage(imgseg, final_seg_path)

        return output_dir_final
    
    # def predict_classification(self, output_dir):
    #     """
    #     Task 2 — Predict treatment response (pCR).
    #     You MUST define this method if participating in Task 2.

    #     Args:
    #         output_dir (str): Directory to save output predictions.

    #     Returns:
    #         pd.DataFrame: DataFrame with patient_id, pcr prediction, and score.
    #     """
    #     patient_ids = self.dataset.get_patient_id_list()
    #     predictions = []
        
    #     for patient_id in patient_ids:
    #         if self.predicted_segmentations:
    #             # === Example using segmentation-derived feature (volume) ===
    #             seg_path = os.path.join(self.predicted_segmentations, f"{patient_id}.nii.gz")
    #             if not os.path.exists(seg_path):
    #                 continue
                
    #             segmentation = sitk.ReadImage(seg_path)
    #             segmentation_array = sitk.GetArrayFromImage(segmentation)
    #             # You can use the predicted segmentation to compute features if task 1 is done
    #             # For example, compute the volume of the segmented region
    #             # ...

    #             # RANDOM CLASSIFIER AS EXAMPLE
    #             # Replace with real feature extraction + ML model
    #             probability = np.random.rand()
    #             pcr_prediction = int(probability > 0.5)

    #         else:
    #             # === Example using raw image intensity for rule-based prediction ===
    #             image_paths = self.dataset.get_dce_mri_path_list(patient_id)
    #             if not image_paths:
    #                 continue
                
    #             image = sitk.ReadImage(image_paths[1])
    #             image_array = sitk.GetArrayFromImage(image)
    #             mean_intensity = np.mean(image_array)
    #             pcr_prediction = 1 if mean_intensity > 500 else 0
    #             probability = np.random.rand() if pcr_prediction == 1 else np.random.rand() / 2
            
    #         # === MANDATORY output format ===
    #         predictions.append({
    #             "patient_id": patient_id,
    #             "pcr": pcr_prediction,
    #             "score": probability
    #         })

    #     return pd.DataFrame(predictions)

# IMPORTANT: The definition of this method will skip the execution of `predict_segmentation` and `predict_classification` if defined
    # def predict_segmentation_and_classification(self, output_dir):
    #     """
    #     Define this method if your model performs both Task 1 (segmentation) and Task 2 (classification).
    #     
    #     This naive combined implementation:
    #         - Generates segmentation masks using thresholding.
    #         - Applies a rule-based volume threshold for response classification.
    #     
    #     Args:
    #         output_dir (str): Path to the output directory.
    #     
    #     Returns:
    #         str: Path to the directory containing the predicted segmentation masks (Task 1).
    #         DataFrame: Pandas DataFrame containing predicted labels and scores (Task 2).
    #     """
    #     # Folder to store predicted segmentation masks
    #     output_dir_final = os.path.join(output_dir, 'pred_segmentations')
    #     os.makedirs(output_dir_final, exist_ok=True)

    #     predictions = []

    #     for patient_id in self.dataset.get_patient_id_list():
    #         # Load DCE-MRI series (assuming post-contrast is the second timepoint)
    #         image_paths = self.dataset.get_dce_mri_path_list(patient_id)
    #         if not image_paths or len(image_paths) < 2:
    #             continue

    #         image = sitk.ReadImage(image_paths[1])
    #         image_array = sitk.GetArrayFromImage(image)

    #         # Step 1: Naive threshold-based segmentation
    #         threshold_value = 150
    #         segmentation_array = (image_array > threshold_value).astype(np.uint8)

    #         # Step 2: Mask segmentation to breast region using provided lesion coordinates
    #         patient_info = self.dataset.read_json_file(patient_id)
    #         if not patient_info or "primary_lesion" not in patient_info:
    #             continue

    #         coords = patient_info["primary_lesion"]["breast_coordinates"]
    #         x_min, x_max = coords["x_min"], coords["x_max"]
    #         y_min, y_max = coords["y_min"], coords["y_max"]
    #         z_min, z_max = coords["z_min"], coords["z_max"]

    #         masked_segmentation = np.zeros_like(segmentation_array)
    #         masked_segmentation[x_min:x_max, y_min:y_max, z_min:z_max] = \
    #             segmentation_array[x_min:x_max, y_min:y_max, z_min:z_max]

    #         # Save predicted segmentation
    #         masked_seg_image = sitk.GetImageFromArray(masked_segmentation)
    #         masked_seg_image.CopyInformation(image)
    #         seg_path = os.path.join(output_dir_final, f"{patient_id}.nii.gz")
    #         sitk.WriteImage(masked_seg_image, seg_path)

    #         # Step 3: Classify based on tumour volume (simple rule-based)
    #         tumor_volume = np.sum(masked_segmentation > 0)
    #         pcr_prediction = 1 if tumor_volume < 1000 else 0
    #         probability = 0.5  # Example: fixed confidence

    #         predictions.append({
    #             "patient_id": patient_id,
    #             "pcr": pcr_prediction,
    #             "score": probability
    #         })

    #     return output_dir_final, pd.DataFrame(predictions)