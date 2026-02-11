import os
import sys
import cv2
import json
import random
import string
import skimage
import skimage.io
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

ROOT_DIR = os.path.abspath("./tf_unet/")
sys.path.append(ROOT_DIR)

from tf_unet import unet, util, image_util

def random_string(size=20, chars=string.ascii_lowercase + string.digits):
    """Generate a random sequence of lowercase of a given size

    Args:
        - size: number of characters in string

    Returns:
        - string of `size` characters
    """

    return ''.join(random.choice(chars) for x in range(size))


def make_polygon(xList, yList):
    """Get a list of xcoordinates and
       ycoordinates (with respect to the
       image itself) and return a polygon as well as
       a closed polygon that can be used as additional
       class label for training

    Args:
        - xList: an array of integers corresponding
                 x-pixel locations (list)
        - yList: an array of integers corresponding
                 y-pixel locations (list)

    Returns:
        - polygon: a list of tuples with (x, y) (list)
        - polygon_closed: a list of tuples with (x, y) (list)
    """

    polygon=[]
    for value in range(len(xList)):
        _tuple=(xList[value], yList[value])
        polygon.append(_tuple)
        
    polygon_closed = polygon
    polygon_closed.append((xList[0], yList[0]))

    return polygon, polygon_closed


def json_to_jpg(JSON_NAME, IMG_DIR, MASK_SAVE_DIR,
                save_images=False, IMG_SAVE_DIR=None,
                label_borders=False):
    """Convert labels from json file format for a group of
       images to a JPEG mask. So far this works only for one
       label per image.

    Args:
        - JSON_NAME: path and name of the JSON file (str)
        - IMG_DIR: the directory where the original images
                   that are used to create the labels in the 
                   JSON file are located (str)
        - MASK_SAVE_DIR: path to save the generated masks (str)

    Returns:
        - creates the directory MASK_SAVE_DIR if it does not
          exist and save the JPEG masks with the corresponding names
          to the MASK_SAVE_DIR directory
    """

    if not os.path.exists(MASK_SAVE_DIR):
        os.mkdir(MASK_SAVE_DIR)

    with open(IMG_DIR + JSON_NAME) as f:
        data = json.load(f)

    for item in data.items():
        name = item[1]['filename']
        regions = item[1]['regions']
        image = Image.open(IMG_DIR + name)

        masked_image = image

        if save_images:
            if not os.path.exists(IMG_SAVE_DIR):
                os.mkdir(IMG_SAVE_DIR)
            os.system(r"cp '{}' {}".format(IMG_DIR + name, IMG_SAVE_DIR))
            print('## copied '+name)

        mask = Image.new('L', image.size, color=(0))
#        mask = Image.new('RGB', image.size, color=(0))

        for region in regions:
            x_pts = region['shape_attributes']['all_points_x']
            y_pts = region['shape_attributes']['all_points_y']
            assert len(x_pts) == len(y_pts)

            polygon, polygon_closed = make_polygon(x_pts, y_pts)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.polygon(polygon, fill=(255))
#            mask_draw.polygon(polygon, fill=(255,255,255))

            if label_borders:
                mask_draw.line(polygon_closed, fill=(255,0,0), width=5)

        # change the original image:
        # The following two lines keep the inside of the polygon
        # instance from json file and mask the rest forably to zero.
        # This is in order avoid penalizing on true positives
        # that are no included in the lazy input annotations.
        #inverse_poly = ImageOps.invert(mask)
        #masked_image.paste(mask,(0,0), mask=inverse_poly)

        random_name = random_string()

#        mask.save(MASK_SAVE_DIR+'{}_mask.png'.format(name[:-4]), "PNG")
#        mask = mask.point(lambda x: 0 if x<128 else 255, '1')

        mask.save(MASK_SAVE_DIR+'{}_mask.png'.format(random_name), "PNG")
        masked_image.save(MASK_SAVE_DIR+'{}.jpg'.format(random_name), "JPEG")
#        image.save(MASK_SAVE_DIR+'{}.jpg'.format(random_name), "JPEG")

        print('## saved {}.png '.format(random_name))


# tf_unet works only when all images have the same .shape
# `reshape_image` gets the skimage instance of an image
# and pads zeros around it to make it applicable for unet
def reshape_image(image, new_height, new_width, mode="center", dtype=np.uint32):
    """takes an image and checks its shape. If the shape does not
       have the desired shape, it pads zeros around the image and
       returns the zero padded image. If the shape is fine, it returns
       the same image intact.

    Args:
    -----
        image: the skimage.io.imread instance of an image
        new_height: the target desired height (integer - # pixels)
        new_width: the target desired width (integer - # pixels)
        mode: the mode of zero padding of the image (string).
              center will pad zeros around the old image and
              corner will keep it in the upper left corner and
              pad the rest with zeros.

    Returns:
    --------
        resized image
    """

    img_height, img_width = image.shape[:2]

    if (img_height, img_width) == (new_height, new_width):
        reshaped_image = image
        print("no reshaping is required")

    else:
        # RGBs have image.shape==3 but masks have image.shape==2
        if len(image.shape)==3:
            reshaped_image = np.zeros([new_height, new_width, 3])
        
        elif len(image.shape)==2:
            reshaped_image = np.zeros([new_height, new_width])
        
        new_height, new_width = int(new_height), int(new_width)

        if mode=="center":
            reshaped_image[new_height//2-img_height//2:new_height//2-img_height//2+img_height,
                          new_width//2-img_width//2:new_width//2-img_width//2+img_width] = image

        elif mode=="corner":
            reshaped_image[:img_height, :img_width] = image

        else:
            raise Exception("The mode '{}' you specified does not exist - can be only 'center' or 'corner'".format(mode))

    return reshaped_image.astype(dtype)


# function to distort image
def elastic_transform(image, mask, weight, alpha, sigma, alpha_affine, random_state=None):
    """Takes an image with its corresponding mask as a cv2.imread instance
       and applies the same elastic deformation to both of them. 

    Args:
    -----
        image: the cv2.imread instance of an image
        mask: the cv2.imread instance of the corresponding mask
        alpha: parameter for the Gaussian filter
        sigma: parameter for the Gaussian filter
        alpha_affine: parameter for affine transform
        random_state: a random state is randomly assigned!

    Returns:
    --------
        image_elastic: transformed original image (numpy array)
        mask_elastic: transformed binary mask (numpy array)

    Usage:
    ------
        image_elastic, mask_elastic = elastic_transform(image, mask, weight,
                                                     image.shape[1]*3,
                                                     image.shape[1]*0.05,
                                                     image.shape[1]*0.09)

    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    # Make sure x,y of the image and mask are identical
    if image.shape[:2] != mask.shape[:2]:
        raise Exception("image and mask must have the same size {}!={}".format(image.shape, mask.shape))

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    
    pts1 = np.float32([center_square + square_size,
                      [center_square[0]+square_size, center_square[1]-square_size],
                       center_square - square_size])
    
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)

    M = cv2.getAffineTransform(pts1, pts2)
    
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    mask  = cv2.warpAffine(mask,  M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    weight  = cv2.warpAffine(weight,  M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    image_elastic   = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    mask_elastic    = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)
    weight_elastic  = map_coordinates(weight, indices, order=1, mode='reflect').reshape(shape)

    # Turn the mask to binary and make it a 2D array
    mask_elastic = np.where(mask_elastic>128, 255, 0)[...,0]
    weight_elastic = weight_elastic[:, :, 0]
    
    return image_elastic, mask_elastic, weight_elastic


def unet_weight_map(y, wc=None, w0 = 15, sigma = 5):
    """Generate weight maps as specified in the U-Net paper
    for boolean mask.

    Args:
    -----
        mask: Numpy array 2D array of shape (image_height, image_width)
              representing binary mask of objects.
        wc: dict
            Dictionary of weight classes.
        w0: int
            Border weight parameter.
        sigma: int
            Border width parameter.

    Returns:
    --------
        w: numpy array of weights

    Usage:
    ------
        wc = {
              0: 1, # background
              1: 5  # objects
             }

        w = unet_weight_map(y, wc)
    """

    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels

        if wc:
            class_weights = np.zeros_like(y)
            for k, v in wc.items():
                class_weights[y == k] = v
            w = w + class_weights
    else:
        w = np.zeros_like(y)

    return w


def border_weight_map(mask, border_weight=5, non_border_weight=1):
    """Generate weight maps with object borders having a heavier
    weight for improving instance segmentation task.

    Args:
    -----
        mask: the skimage.io.imread instance of a mask
        border_weight: weight applied to border pixels (float)
        non_border_weight: weight for non-border pixels (float)

    Returns:
    --------
        w: numpy array of border weight maps
    """

#    _mask = skimage.io.imread(mask, dtype=np.float)
    boundaries = find_boundaries(mask)
    w = np.where(boundaries==True, border_weight, non_border_weight)

    return w


def load_data_unit(IMG_PATH):
    """The image_util.ImageDataProvider in image_util is useful when
       both training image and mask are available. This is a simple 
       adjustment to that script to load individual image and be able
       to predict the mask using a trained model, without having to
       load both the mask and image

    Args:
    -----
        IMG_PATH: full string path to an image, e.g.
              /path/to/image.png 

    Returns:
    --------
        x_test: an array to feed the trained net.predict()
    """

    _sample = np.array(Image.open(IMG_PATH), np.float32)

    _sample = np.clip(np.fabs(_sample), a_min=-np.inf, a_max=np.inf)

    _sample -= np.amin(_sample)

    if np.amax(_sample) != 0:
        _sample /= np.amax(_sample)

    x_test = np.zeros((1, _sample.shape[0],
                          _sample.shape[1],
                          _sample.shape[2]),
                          dtype=np.float32)
    x_test[0,...] = _sample

    return x_test


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Args:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


