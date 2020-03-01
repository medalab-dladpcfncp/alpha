"""Created on 2018/12/18

Usage: Everying about image preprocessing

Content:
    get_pixels_hu
    resample
    minmax_normalization
    windowing
    smoothing
    find_largest
"""

import os
import glob
import ntpath

import numpy as np
import nibabel as nib
from skimage import morphology, measure
import scipy


def get_pixels_hu(scans):
    """
    Usage: Transfer dicom image's data to Hounsfield units (HU)
    """
    assert len(scans) > 0, "Must not be empty array"
    # Convert to int16, should be possible as values should
    # always be low enough (<32k)
    image = np.stack([s.pixel_array for s in scans]).astype(np.int16)

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(scans)):
        intercept = scans[slice_number].RescaleIntercept
        slope = scans[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = (slope * image[slice_number]
                                   .astype(np.float64)).astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def resample(image, original_spacing, new_spacing=[1, 1, 1]):
    """
    Resample images to specific new spacing.

    Reference
    ---------
    https://stackoverflow.com/questions/13242382/resampling-a-numpy-array-representing-an-image
    http://scipy.github.io/devdocs/generated/scipy.ndimage.zoom.html#scipy.ndimage.zoom
    """

    resize_factor = original_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = original_spacing / real_resize_factor
    image = scipy.ndimage.zoom(image, real_resize_factor,
                               order=0, mode='nearest')
    return image, new_spacing


def minmax_normalization(img):
    img_min, img_max = np.min(img), np.max(img)
    img = img - img_min
    img = img / (img_max - img_min)
    return img


def windowing(img, min_val=-50, max_val=200):
    img = np.clip(img, min_val, max_val)
    return img


def smoothing(mask):
    assert len(mask.shape) == 3, "The input dimension must be 3!"
    mask = morphology.erosion(mask, np.ones([3, 3, 3]))
    mask = morphology.dilation(mask, np.ones([3, 3, 3]))
    return mask


def find_largest(label):
    lbl_map = measure.label(label.astype(int))
    volumn_count = np.delete(np.bincount(lbl_map.flat), 0)
    new_label = np.zeros(label.shape)
    new_label[np.where(lbl_map == np.argmax(volumn_count) + 1)] = 1
    return new_label


def find_mean(image, pancreas, lesion):
    healthy_pancreas = pancreas.copy()
    healthy_pancreas[np.where(lesion == 1)] = 0

    mean_image = image.copy()
    return np.mean(image[np.where(pancreas == 1)])


def pancreas_normalization(image, pancreas, lesion):
    pancreas_mean = find_mean(image, pancreas, lesion)
    image = image - pancreas_mean
    return image
