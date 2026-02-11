import os
import glob
import cv2
import skimage.io
from skimage.color import rgb2gray
import numpy as np
from util_unet import (
    json_to_jpg,
    reshape_image,
    unet_weight_map,
    border_weight_map,
    elastic_transform
    )
import warnings
warnings.filterwarnings("ignore")

## some constants and paths
###########################
STEP = 4

# target image reshaping dims
N_Y, N_X = 768, 768

IMAGE_DIR = os.getcwd() + '/Sentinel-dense-train-all-original/'
AUGMENTATION_DIR = os.getcwd() + '/Sentinel-dense-train-all-augmented/'

if not os.path.exists(IMAGE_DIR):
    os.mkdir(IMAGE_DIR)
if not os.path.exists(AUGMENTATION_DIR):
    os.mkdir(AUGMENTATION_DIR)
##################################

def prep(STEP):
    ## step 1: from VGG to binary PNG labels
    ## - find original images and annotations
    ## - convert the VGG annotations to binary .PNG format
    ## - convert the original images from JPG to PNG
    ## - remove the old JPG files
    if STEP==1:
        IMG_DIRs = [
                    '/home/soroushr/projects/Planet/planet-mask-rcnn/mask-rcnn_icebergs/iceberg_dataset_Sentinel/',
#                    '/home/soroushr/projects/Planet/planet-mask-rcnn/mask-rcnn_icebergs/iceberg_dataset_dense/train/',
#                    '/home/soroushr/projects/Planet/planet-mask-rcnn/mask-rcnn_icebergs/iceberg_dataset_dense/val/',
#                    '/home/soroushr/projects/Planet/planet-mask-rcnn/mask-rcnn_icebergs/iceberg_dataset_dense2/train/',
#                    '/home/soroushr/projects/Planet/planet-mask-rcnn/mask-rcnn_icebergs/iceberg_dataset_dense2/val/',
#                    '/home/soroushr/projects/Planet/planet-mask-rcnn/mask-rcnn_icebergs/iceberg_dataset_large/val/'
                   ]

        IMG_SAVE_DIRs = [os.getcwd() + '/Sentinel-dense-train-all-original/']*len(IMG_DIRs)

        JSON_NAME     = 'iceberg-annotations.json'

        for IMG_DIR, IMG_SAVE_DIR in zip(IMG_DIRs, IMG_SAVE_DIRs):

            MASK_SAVE_DIR = IMG_SAVE_DIR

            json_to_jpg(JSON_NAME,
                        IMG_DIR,
                        MASK_SAVE_DIR,
                        save_images=False,
                        IMG_SAVE_DIR=IMG_SAVE_DIR)

            print("converting JPGs to PNGs in {}".format(IMG_SAVE_DIR))
            os.system('mogrify -format png -path {} {}*.jpg'.format(IMG_SAVE_DIR, IMG_SAVE_DIR))

            print("removing JPGs in {}".format(IMG_SAVE_DIR))
            os.system('find {} -type f -iname \*.jpg -delete'.format(IMG_SAVE_DIR))

    ##step 2
    ## - reshape the images to a desired N_Y, N_X
    elif STEP==2:
        count = 0
        for image in os.listdir(IMAGE_DIR):

            # only take PNGs
            if os.path.splitext(image)[-1].lower() != '.png':
                continue
            count += 1
            image_name = os.path.basename(image)
            print(count)

            image = skimage.io.imread(IMAGE_DIR + '/' + image)

            print("original size: ", image.shape[:2])
            reshaped_image = reshape_image(image, N_Y, N_X, mode="center")

            skimage.io.imsave(IMAGE_DIR+image_name, reshaped_image)
            print("resized and saved {}".format(image_name))

    ##step 3: generate weight map based on the masks
    ##   it can either be the unet papers weight map (unet_weight_map)
    ##   or just adding more weight to the borders (border_weight_map)
    elif STEP==3:
        masks  = glob.glob(IMAGE_DIR+'/*_mask.png')
#        wc = {0: 1, # background
#              1: 5  # objects
#                  }
        for item in masks:
            mask_name = os.path.basename(item)
            
            mask = skimage.io.imread(item, dtype=np.float)
#            weight = unet_weight_map(mask//255, wc, w0=15, sigma=5)
            border_weight = border_weight_map(mask, border_weight = 5,
                                                    non_border_weight = 1)

            WEIGHT_BORDER_DIR = IMAGE_DIR

            if not os.path.exists(WEIGHT_BORDER_DIR):
                os.mkdir(WEIGHT_BORDER_DIR)

            print("border weight maps generated for {}".format(mask_name))

            skimage.io.imsave(WEIGHT_BORDER_DIR+'{}_border.png'.format(mask_name[:-9]), border_weight.astype(np.uint8))

    ##step 4: augment everything
    ## - Apply elastic deformation similar to 
    ## the original unet paper for augmenting
    ## images, labels, and weight maps
    ## - rgb2gray for saving augmented masks
    ## ensures that the saved image is (nx, ny)
    elif STEP==4:
        masks  = glob.glob(IMAGE_DIR+'/*_mask.png') 
        images = [mask[:-9] + '.png' for mask in masks]
        weights = [mask[:-9] + '_border.png' for mask in masks]

        alphas  = np.arange(3,5)
        sigmas  = [0.04, 0.05] 
        affines = [0.1, 0.20]

        count = 0
        for image, mask, weight in zip(images, masks, weights):

            # get the image name w/o path
            img_name = os.path.basename(image)

            # load image and corresponding mask
            image = cv2.imread(image)
            mask  = cv2.imread(mask)
            weight  = cv2.imread(weight)

            image_flip  = cv2.flip(image, -1)
            mask_flip   = cv2.flip(mask, -1)
            weight_flip = cv2.flip(weight, -1)

            count += 1
            print("#number {}".format(count))
            print("fliped image {}".format(img_name) )

            skimage.io.imsave(AUGMENTATION_DIR+'{}_flip.png'.format(img_name[:-4]), image_flip)
            skimage.io.imsave(AUGMENTATION_DIR+'{}_flip_mask.png'.format(img_name[:-4]), rgb2gray(mask_flip) )
            skimage.io.imsave(AUGMENTATION_DIR+'{}_flip_border.png'.format(img_name[:-4]), weight_flip.astype(np.uint8) )

            # apply all combination of deformation
            # parameters to all images and masks
            for alpha in alphas:
                for sigma in sigmas:
                    for affine in affines:
                        image_elastic, mask_elastic, weight_elastic = elastic_transform(image, mask, weight,
                                                                                             image.shape[1]*alpha,
                                                                                             image.shape[1]*sigma,
                                                                                             image.shape[1]*affine)

                        count += 1
                        print("#number {}".format(count))
                        print("transformation applied on {} with alpha={}, sigma={}, affine={}".format(img_name, alpha, sigma, affine))

                        skimage.io.imsave(AUGMENTATION_DIR+'{}_al_{}_si_{}_af_{}.png'.format(img_name[:-4],
                                                                                             int(alpha),
                                                                                             int(sigma*100),
                                                                                             int(affine*100)),
                                                                                             image_elastic)

                        skimage.io.imsave(AUGMENTATION_DIR+'{}_al_{}_si_{}_af_{}_mask.png'.format(img_name[:-4],
                                                                                                  int(alpha),
                                                                                                  int(sigma*100),
                                                                                                  int(affine*100)),
                                                                                                  rgb2gray(mask_elastic) )

                        skimage.io.imsave(AUGMENTATION_DIR+'{}_al_{}_si_{}_af_{}_border.png'.format(img_name[:-4],
                                                                                                  int(alpha),
                                                                                                  int(sigma*100),
                                                                                                  int(affine*100)),
                                                                                                  weight_elastic.astype(np.uint8) )

##################################
##### run the specified step
##################################
prep(STEP)
