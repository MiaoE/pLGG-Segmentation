import numpy as np
# from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import hausdorff_distance
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
    size_gt = gt.sum()
    size_seg = seg.sum()

    # Handle empty case
    if size_gt + size_seg - intersection == 0:
        return 1.0  # define IOU as 1.0

    return intersection / (size_gt + size_seg - intersection)

def hausdorff_dist(gt: np.ndarray, seg: np.ndarray):
    return hausdorff_distance(gt, seg)
