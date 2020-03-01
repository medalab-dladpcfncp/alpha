"""Created on 2019/01/02

Usage: 2D patch data generator

Content:
    patch_generator
    masked_2D_sampler
    masked_3D_sampler
    adjust_patch_num
"""


import os
import glob
import ntpath

import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure
import scipy.misc
import random

from data_loader.preprocessing import (minmax_normalization, windowing,
                                       smoothing)

# from preprocessing import (minmax_normalization, windowing,
#                            smoothing)


def patch_generator(data_path, tumor_id, patch_size=50,
                    dilation=[5, 5, 5], add_mask=True,
                    stride=30, threshold=0.5,
                    max_amount=1000):
    """
    Usage: Complete process of patch generating

    Parameters
    ----------
    data_path (str): The path for the box data
    tumor_id (str): The target ID
                    Example: 'PT1'
    patch_size (int): Patch size, default to 50
    dilation (int): the dilation of non-lesion region

    Returns
    -------
    List: each content is a 2D patch
    List: each content is a label

    """

    X = []
    Y = []

    img_path = data_path + '/' + tumor_id + '/ctscan.npy'
    pan_path = data_path + '/' + tumor_id + '/pancreas.npy'

    pancreas = np.load(pan_path)
    pancreas = smoothing(pancreas)

    if tumor_id[:2] == 'PC' or tumor_id[:2] == 'PT':
        les_path = data_path + '/' + tumor_id + '/lesion.npy'
        lesion = np.load(les_path)
        lesion = smoothing(lesion)
    else:
        lesion = np.zeros(pancreas.shape)

    pancreas = morphology.dilation(pancreas, np.ones([3, 3, 1]))
    lesion = morphology.dilation(lesion, np.ones([3, 3, 1]))

    coords = masked_2D_sampler(
        lesion, pancreas, patch_size, stride, threshold,
        max_amount=max_amount)

    img = np.load(img_path)
    img = windowing(img)
    img = minmax_normalization(img)
    if add_mask:
        mask = np.zeros(pancreas.shape)
        mask[np.where(lesion == 1)] = 1
        mask[np.where(pancreas == 1)] = 1
        img = img * mask

    for coord in coords:
        mask_pancreas = img[coord[1]:coord[4], coord[2]:coord[5], coord[3]]
        X.append(mask_pancreas)
        Y.append(coord[0])

    return X, Y


def masked_2D_sampler(lesion, pancreas, patch_size, stride,
                      threshold, max_amount=1000, y_type='bin'):
    """
    Usage: generate 2D patches from mask

    Parameters
    ----------
    mask (numpy array): Mask for patch segmentation
    patch_size (int): Patch size
    stride (int): Distance of the moving window
    threshold (float): 0 < threshold < 1

    Returns
    -------
    List: each content is a 2D patch coordinates [y, min x, min y, min z,
                                                     max x, max y, max z]

    """

    coords = []
    for z in range(lesion.shape[2]):
        for row in range((lesion.shape[0] - patch_size) // stride + 1):
            for col in range((lesion.shape[1] - patch_size) // stride + 1):
                x_min = row * stride
                y_min = col * stride

                x_max = x_min + patch_size
                if x_max > lesion.shape[0]:
                    x_max = lesion.shape[0]
                    x_min = x_max - patch_size
                y_max = y_min + patch_size
                if y_max > lesion.shape[1]:
                    y_max = lesion.shape[1]
                    y_min = y_max - patch_size
                z_min = z
                z_max = z + 1
                patch_lesion = lesion[x_min:x_max, y_min:y_max, z_min:z_max]
                patch_pancreas = pancreas[x_min:x_max, y_min:y_max, z_min:z_max]
                if np.sum(patch_lesion) / (patch_size**2) > threshold:
                    if y_type == 'bin':
                        value = 1
                    elif y_type == 'reg':
                        value = np.sum(patch_lesion) / (patch_size**2)
                    coords.append([value, x_min, y_min, z_min,
                                   x_max, y_max, z_max])
                elif np.sum(patch_pancreas) / (patch_size**2) > threshold:
                    value = 0
                    coords.append([value, x_min, y_min, z_min,
                                   x_max, y_max, z_max])
                # else:
                #     value = 0
                # coords.append([value, x_min, y_min, z_min,
                #                x_max, y_max, z_max])
    while len(coords) > max_amount:
        coords = coords[::2]

    return coords


def masked_3D_sampler(mask, patch_size, stride, threshold, value=0):
    """
    Usage: generate 3D patches from mask

    Parameters
    ----------
    mask (numpy array): Mask for patch segmentation
    patch_size (int): Patch size
    stride (int): Distance of the moving window
    threshold (float): 0 < threshold < 1

    Returns
    -------
    List: each content is a 2D patch coordinates [y, min x, min y, min z,
                                                     max x, max y, max z]

    """

    coords = []
    lbl_map = measure.label(mask.astype(int))
    for region in measure.regionprops(lbl_map):
        box = region.bbox

        xlen = box[3] - box[0]
        ylen = box[4] - box[1]
        zlen = box[5] - box[2]
        for row in range(max((xlen - patch_size), 0) // stride + 1):
            for col in range(max((ylen - patch_size), 0) // stride + 1):
                for lyr in range(max((zlen - patch_size), 0) // stride + 1):
                    x_min = box[0] + row * stride
                    y_min = box[1] + col * stride
                    z_min = box[2] + lyr * stride

                    x_max = x_min + patch_size
                    if x_max > box[3]:
                        x_max = box[3]
                        x_min = x_max - patch_size
                    y_max = y_min + patch_size
                    if y_max > box[4]:
                        y_max = box[4]
                        y_min = y_max - patch_size
                    z_max = z_min + patch_size
                    if z_max > box[5]:
                        z_max = box[5]
                        z_min = z_max - patch_size

                    patch = mask[x_min:x_max, y_min:y_max, z_min:z_max]
                    if np.sum(patch) / patch_size**3 > threshold:
                        coords.append([value, x_min, y_min, z_min,
                                       x_max, y_max, z_max])
    return coords


def adjust_patch_num(mask, patch_size, stride, threshold, stride_decay=0.5,
                     threshold_decay=0.5, min_amount=10,
                     max_amount=200, train_mode=True, force=False):
    """
    Usage: adjust the number of patches in each case

    Parameters
    ----------
    mask (numpy array): Mask for patch segmentation
    patch_size (int): Original patch size
    stride (int): Original distance of the moving window
    threshold (float): 0 < threshold < 1, original threshold
    stride_decay (float): 0 < stride_decay < 1, decay amount of stride
    threshold_decay (float): 0 < threshold_decay < 1,
                               decay amount of stride
    min_amount (int): Minimal amount of patches in each case
                      Please notice that this only monitor the decay part,
                      sometimes it still cannot generate any patches,
                      unless using force
    max_amount (int): Maximal amount of patches in each case
    force (bool): when there's more than 10 pixels in a patches,
                  it will be counted.

    Returns
    -------
    List: each content is a 2D patch coordinates [y, min x, min y, min z,
                                                     max x, max y, max z]

    """

    # initial generating
    coords = []

    # loose standard if not enough patches
    while len(coords) < min_amount and stride > 0 and threshold > 0:
        coords = masked_2D_sampler(mask, patch_size, stride, threshold)
        stride = int(stride * stride_decay)
        # threshold = threshold * threshold_decay

    # # force to generate mask if the mask is "lesion"
    # if len(coords) == 0 and force:
    #     coords = masked_2D_sampler(mask, patch_size,
    #                                stride, 10/(patch_size**2))
    #     min_amount = 5

    # randomly delete patches if too much
    if len(coords) > max_amount:
        random.shuffle(coords)
        coords = coords[:max_amount]

    return coords
