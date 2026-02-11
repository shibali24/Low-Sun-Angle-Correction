# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
author: jakeret

modified: S Rezvanbehbahani & R Keramati
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import glob
import numpy as np
from PIL import Image


class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    
    channels = 1
    n_class = 2

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label = self._next_data()
            
        train_data = self._process_data(data)
        labels = self._process_labels(label)
        
        train_data, labels = self._post_process(train_data, labels)
        
        nx = train_data.shape[1]
        ny = train_data.shape[0]

        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class),

    def _load_data_and_label_and_weight(self):
        data, label, weight = self._next_data()
            
        train_data = self._process_data(data)
        labels = self._process_labels(label)
        weights = self._process_weights(weight)

        train_data, labels, weights = self._post_process(train_data, labels, weights)

#        print(train_data.shape, labels.shape, weights.shape)
#        if train_data.shape[-1] != 3:
#            print(self.data_files[self.file_idx])

        nx = train_data.shape[1]
        ny = train_data.shape[0]

        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class), weights.reshape(1, ny, nx, 1)

    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)

            if label.dtype != 'bool':
                label = label.astype(np.bool)

            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels

        return label
    
    def _process_weights(self, weight):
        # FIXME: Quick and dirty fix; sometimes the augmentation saves
        # the weights as an RGB, which causes tensor size mismatch in
        # tensorflow. This is a temporary fix for that issue.
        # The augmentation code needs to be fixed.
        if weight.shape[-1] == 3:
            weight = weight[:, :, 0]
            print("weight shape adjusted")
        nx = weight.shape[1]
        ny = weight.shape[0]
        weights = np.zeros((ny, nx, 1), dtype=np.float32)
        weights[:, :, 0] = weight

        return weights

    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)

        if np.amax(data) != 0:
            data /= np.amax(data)

        return data
    
    def _post_process(self, data, labels, weights):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        """
        return data, labels, weights
    
    def __call__(self, n):
        train_data, labels, weights = self._load_data_and_label_and_weight()
        nx = train_data.shape[1]
        ny = train_data.shape[2]
    
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
#        W = np.zeros((n, nx, ny, self.n_class))
        W = np.zeros((n, nx, ny, 1))
    
        X[0] = train_data
        Y[0] = labels
        W[0] = weights
        for i in range(1, n):
            train_data, labels, weights = self._load_data_and_label_and_weight()
            X[i] = train_data
            Y[i] = labels
            W[i] = weights

        return X, Y, W


class SimpleDataProvider(BaseDataProvider):
    """
    A simple data provider for numpy arrays. 
    Assumes that the data and label are numpy array with the dimensions
    data `[n, X, Y, channels]`, label `[n, X, Y, classes]`. Where
    `n` is the number of images, `X`, `Y` the size of the image.

    :param data: data numpy array. Shape=[n, X, Y, channels]
    :param label: label numpy array. Shape=[n, X, Y, classes]
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    
    def __init__(self, data, label, a_min=None, a_max=None):
        super(SimpleDataProvider, self).__init__(a_min, a_max)
        self.data = data
        self.label = label
        self.file_count = data.shape[0]
        self.n_class = label.shape[-1]
        self.channels = data.shape[-1]

    def _next_data(self):
        idx = np.random.choice(self.file_count)
        return self.data[idx], self.label[idx]


class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'
    Number of pixels in x and y of the images and masks should be even.

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'

    """
    
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".tif", mask_suffix='_mask.tif', shuffle_data=True):
        super(ImageDataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data

        self.data_files = self._find_data_files(search_path)
        
        if self.shuffle_data:
            np.random.shuffle(self.data_files)
        
        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))

        image_path = self.data_files[0]
        label_path = image_path.replace(self.data_suffix, self.mask_suffix)
        img = self._load_file(image_path)
        mask = self._load_file(label_path)
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        self.n_class = 2 if len(mask.shape) == 2 else mask.shape[-1]

        print("Number of channels: %s"%self.channels)
        print("Number of classes: %s"%self.n_class)

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]
    
    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.data_files)
        
    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)
        
        img = self._load_file(image_name, np.float32)
        label = self._load_file(label_name, np.bool)
    
        return img,label



###############################
# FIXME: fix the "border" vs. "weight" variable names - they're the same
class ImageDataProviderWithWeights(BaseDataProvider):
    """
    make it load the borders too!
    """
    
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".png",
                                                            mask_suffix="_mask.png",
                                                            border_suffix="_border.png",
                                                            shuffle_data=True):
        super(ImageDataProviderWithWeights, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.border_suffix = border_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data

        self.data_files = self._find_data_files(search_path)
        
        if self.shuffle_data:
            np.random.shuffle(self.data_files)
        
        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))
        self.whole_batch_size = len(self.data_files) #Rk added this to load the whole data

        image_path = self.data_files[0]
        label_path = image_path.replace(self.data_suffix, self.mask_suffix)
        border_path = image_path.replace(self.data_suffix, self.border_suffix)
        img = self._load_file(image_path)
        mask = self._load_file(label_path)
        border = self._load_file(border_path)
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        self.n_class = 2 if len(mask.shape) == 2 else mask.shape[-1]

        print("Number of channels: %s"%self.channels)
        print("Number of classes: %s"%self.n_class)

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name and not self.border_suffix in name]

    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.data_files)

    def _next_data(self):
        self._cylce_file() # Just for now
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)
        border_name = image_name.replace(self.data_suffix, self.border_suffix)
        
        img = self._load_file(image_name, np.float32)
        label = self._load_file(label_name, np.bool)
        border = self._load_file(border_name, np.float32)

        return img, label, border



###############################
class ImageDataProviderForPrediction(BaseDataProvider):
    """
    - Identical to ImageDataProviderWithWeights but does not look for mask or border suffixes
    - This data provider is solely used for final prediction on a bunch of images
    """

    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".png",
                                                            shuffle_data=True):

        super(ImageDataProviderForPrediction, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data

        self.data_files = self._find_data_files(search_path)

        if self.shuffle_data:
            np.random.shuffle(self.data_files)

        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))
        self.whole_batch_size = len(self.data_files) #Rk added this to load the whole data

        image_path = self.data_files[0]
        img = self._load_file(image_path)
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]

        print("Number of channels: %s"%self.channels)

    def _load_data(self):
        data = self._next_data()

        pred_data = self._process_data(data)
        
        nx = pred_data.shape[1]
        ny = pred_data.shape[0]

        return pred_data.reshape(1, ny, nx, self.channels)

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if self.data_suffix in name]

    def _load_file(self, path, dtype=np.float32):
        # FIXME: for now it forces the model class to 
        # spit out 3d arrays for images - must be fixed
        # to ignore and move on to the next batch
        _image = np.array(Image.open(path), dtype)
        if len(_image.shape)>2:
            return _image
        else:
            return np.zeros((_image.shape[0], _image.shape[1], 3))
#        return np.array(Image.open(path), dtype)

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.data_files)

    def _next_data(self):
        self._cylce_file() # Just for now
        image_name = self.data_files[self.file_idx]

        img = self._load_file(image_name, np.float32)

        return img

    def _post_process_data(self, data):

        return data

    def __call__(self, n):
        pred_data = self._load_data()
        nx = pred_data.shape[1]
        ny = pred_data.shape[2]
    
        X = np.zeros((n, nx, ny, self.channels))
    
        X[0] = pred_data
        for i in range(1, n):
            pred_data = self._load_data()
            X[i] = pred_data

        return X
