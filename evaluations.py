import os
import json

import torch
import numpy as np
from scipy.spatial.distance import cdist
from skimage.metrics import hausdorff_distance
from scipy.ndimage import binary_erosion
# from monai.metrics import DiceMetric, HausdorffDistanceMetric
# from medpy.metric.binary import dc, hd95

def dice_coefficient_score(gt: np.ndarray, seg: np.ndarray):
    gt = gt.astype(bool)
    seg = seg.astype(bool)
    intersection = np.logical_and(seg, gt).sum()
    size_gt = gt.sum()
    size_seg = seg.sum()

    # Handle both empty case
    if size_gt + size_seg == 0:
        return 1.0  # define Dice as 1.0 if both masks are empty

    return 2.0 * intersection / (size_gt + size_seg)

def iou_score(gt: np.ndarray, seg: np.ndarray):
    gt = gt.astype(bool)
    seg = seg.astype(bool)
    intersection = np.logical_and(seg, gt).sum()
    union = np.logical_or(seg, gt).sum()

    # Handle empty case
    if union == 0:
        return 1.0

    return intersection / union

def hausdorff(gt: np.ndarray, seg: np.ndarray, voxel_spacing=None):
    gt = gt.astype(bool)
    seg = seg.astype(bool)

    if not gt.any() and not seg.any():
        return 0.0
    if gt.any() != seg.any():
        return np.inf

    if voxel_spacing is None:
        voxel_spacing = np.ones(3)

    gt_pts = np.argwhere(gt) * voxel_spacing
    seg_pts = np.argwhere(seg) * voxel_spacing

    d_gt_to_seg = cdist(gt_pts, seg_pts).min(axis=1)
    d_seg_to_gt = cdist(seg_pts, gt_pts).min(axis=1)

    return max(d_gt_to_seg.max(), d_seg_to_gt.max())

def hd95(gt: np.ndarray, seg: np.ndarray, voxel_spacing=None):
    """
    @param voxel_spacing: Pixel spacing (e.g., (1.0, 1.0)). If None, spacing = 1.
    @type voxel_spacing: tuple or None
    
    @return HD95 distance, 0.0 if both masks are empty, np.inf if one mask if empty
    @rtype: float
    """
    gt = gt.astype(bool)
    seg = seg.astype(bool)

    # Handle empty cases
    if not gt.any() and not seg.any():
        return 0.0
    if gt.any() != seg.any():
        return np.inf

    if voxel_spacing is None:
        voxel_spacing = np.ones(gt.ndim)

    # Extract surfaces
    gt_surface = gt ^ binary_erosion(gt)
    seg_surface = seg ^ binary_erosion(seg)

    gt_pts = np.argwhere(gt_surface) * voxel_spacing
    seg_pts = np.argwhere(seg_surface) * voxel_spacing

    # Compute surface-to-surface distances
    d_gt_to_seg = cdist(gt_pts, seg_pts).min(axis=1)
    d_seg_to_gt = cdist(seg_pts, gt_pts).min(axis=1)

    all_dists = np.concatenate([d_gt_to_seg, d_seg_to_gt])

    return np.percentile(all_dists, 95)

    )

