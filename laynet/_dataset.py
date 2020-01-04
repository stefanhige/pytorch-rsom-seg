import os
import copy
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class RSOMLayerDataset(Dataset):
    """
    rsom dataset class for layer segmentation
    
    Args:
        root_dir (string): Directory with all the nii.gz files.
        data_str (string): end part of filename of training data.
        label_str (string): end part of filename of segmentation ground truth data.
        transform (callable, optional): Optional transform to be applied
                            on a sample.
    """

    def __init__(self, 
                 root_dir, 
                 data_str='_rgb.nii.gz', 
                 label_str='_l.nii.gz', 
                 transform=None):

        assert os.path.exists(root_dir) and os.path.isdir(root_dir), \
        'root_dir not a valid directory'
        
        self.root_dir = root_dir
        self.transform = transform
        
        assert isinstance(data_str, str) and isinstance(label_str, str), \
        'data_str or label_str not valid.'
        
        self.data_str = data_str
        self.label_str = label_str
        
        # get all files in root_dir
        all_files = os.listdir(path=root_dir)
        # extract the  data files
        self.data = [el for el in all_files if el[-len(data_str):] == data_str]
        
        assert len(self.data) == \
            len([el for el in all_files if el[-len(label_str):] == label_str]), \
            'Amount of data and label files not equal.'

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _readNII(rpath):
        '''
        read in the .nii.gz file
        Args:
            rpath (string)
        '''
        
        img = nib.load(str(rpath))
        
        # TODO: when does nib get_fdata() support rgb?
        # currently not, need to use old method get_data()
        return img.get_data()

    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, 
                            self.data[idx])
        label_path = os.path.join(self.root_dir, 
                                   self.data[idx].replace(self.data_str, self.label_str))
        
        # read data
        data = self._readNII(data_path)
        data = np.stack([data['R'], data['G'], data['B']], axis=-1)
        data = data.astype(np.float32)
        
        # read label
        label = self._readNII(label_path)
        label = label.astype(np.float32)
        
        # add meta information
        meta = {'filename': self.data[idx],
                'dcrop':{'begin': None, 'end': None},
                'lcrop':{'begin': None, 'end': None},
                'weight': 0}

        sample = {'data': data, 'label': label, 'meta': meta}

        if self.transform:
            sample = self.transform(sample)

        return sample

class RSOMLayerDatasetUnlabeled(RSOMLayerDataset):
    """
    rsom dataset class for layer segmentation
    for prediction of unlabeled data only
    
    Args:
        root_dir (string): Directory with all the nii.gz files.
        data_str (string): end part of filename of training data
        transform (callable, optional): Optional transform to be applied
                            on a sample.
    """
    def __init__(self, root_dir, data_str='_rgb.nii.gz', transform=None):
        
        assert os.path.exists(root_dir) and os.path.isdir(root_dir), \
        'root_dir not a valid directory'
        
        self.root_dir = root_dir
        self.transform = transform
        
        assert isinstance(data_str, str), 'data_str or label_str not valid.'
        
        self.data_str = data_str
        
        # get all files in root_dir
        all_files = os.listdir(path=root_dir)
        # extract the  data files
        self.data = [el for el in all_files if el[-len(data_str):] == data_str]
        

    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, 
                            self.data[idx])
        
        # read data
        data = self._readNII(data_path)
        data = np.stack([data['R'], data['G'], data['B']], axis=-1)
        data = data.astype(np.float32)
        
        label = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)
        
        # add meta information
        meta = {'filename': self.data[idx],
                'dcrop':{'begin': None, 'end': None},
                'lcrop':{'begin': None, 'end': None},
                'weight': 0}

        sample = {'data': data, 'label': label, 'meta': meta}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        
        # data is [Z x X x Y x 3] [500 x 171 x 333 x 3]
        # label is [Z x X x Y] [500 x 171 x 333]
        
        # we want one sample to be [Z x Y x 3]  2D rgb image
        
        # numpy array size of images
        # [H x W x C]
        # torch tensor size of images
        # [C x H x W]
        
        # and for batches
        # [B x C x H x W]
        
        # here, X is the batch size.
        # so we want to reshape to
        # [X x C x Z x Y] [171 x 3 x 500 x 333]
        data = data.transpose((1, 3, 0, 2))
        
        # and for the label
        # [X x Z x Y] [171 x 500 x 333]
        label = label.transpose((1, 0, 2))

        return {'data': torch.from_numpy(data),
                'label': torch.from_numpy(label),
                'meta': meta}

class ZeroCenter(object):
    """ 
    Zero center input volumes
    """    
    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        # data still is RGB
        assert data.shape[3] == 3
        
        # compute for all x,y,z mean for every color channel
        # rgb_mean = np.around(np.mean(data, axis=(0, 1, 2))).astype(np.int16)
        # meanvec = np.tile(rgb_mean, (data.shape[:-1] + (1,)))
        # mean has shown to have no effect 
        data -= 127
        
        return {'data': data, 'label': label, 'meta': meta}
    
class DropBlue(object):
    """
    Drop the last slice of the RGB dimension
    RSOM images are 2channel, so blue is empty anyways.
    """
    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        # data still is RGB
        assert data.shape[3] == 3

        data = data[:,:,:,:2]

        assert data.shape[3] == 2

        return {'data': data, 'label': label, 'meta': meta}

class CropToEven(object):
    """ 
    if Volume shape is not even numbers, simply crop the first element
    except for last dimension, this is RGB  = 3
    """
    def __init__(self,network_depth=3):
        # how the unet works, without getting a upscaling error, 
        # the input shape must be a multiplier of 2**(network_depth-1)
        self.maxdiv = 2**(network_depth - 1)
        self.network_depth = network_depth

    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        
        # for backward compatibility
        # easy version: first crop to even, crop rest afterwards, if necessary
        initial_dshape = data.shape
        initial_lshape = label.shape

        IsOdd = np.mod(data.shape[:-1], 2)
        
        data = data[IsOdd[0]:, IsOdd[1]:, IsOdd[2]:, : ]
        label = label[IsOdd[0]:, IsOdd[1]:, IsOdd[2]:]

        if not isinstance(meta['weight'], int):
            raise NotImplementedError('Weight was calulated before. Cropping implementation missing')
            
        
        # save, how much data was cropped
        # using torch tensor, because dataloader will convert anyways
        # dcrop = data
        meta['dcrop']['begin'] = torch.from_numpy(
                np.array([IsOdd[0], IsOdd[1], IsOdd[2], 0], dtype=np.int16))
        meta['dcrop']['end'] = torch.from_numpy(
                np.array([0, 0, 0, 0], dtype=np.int16))
        
        # lcrop = label
        meta['lcrop']['begin'] = torch.from_numpy(
                np.array([IsOdd[0], IsOdd[1], IsOdd[2]], dtype=np.int16))
        meta['lcrop']['end'] = torch.from_numpy(
                np.array([0, 0, 0], dtype=np.int16))

        # check if Z and Y are divisible through self.maxdiv
        rem0 = np.mod(data.shape[0], self.maxdiv)
        rem2 = np.mod(data.shape[2], self.maxdiv)
        
        if rem0 or rem2:
            if rem0:
                # crop Z
                data = data[int(np.floor(rem0/2)):-int(np.ceil(rem0/2)), :, :, :]
                label = label[int(np.floor(rem0/2)):-int(np.ceil(rem0/2)), :, :]

            if rem2:
                # crop Y
                data = data[ :, :, int(np.floor(rem2/2)):-int(np.ceil(rem2/2)), :]
                label = label[:, :, int(np.floor(rem2/2)):-int(np.ceil(rem2/2))]
        
            # add to meta information, how much has been cropped
            meta['dcrop']['begin'] += torch.from_numpy(
                    np.array([np.floor(rem0/2), 0, np.floor(rem2/2), 0], dtype=np.int16))
            meta['dcrop']['end'] += torch.from_numpy(
                    np.array([np.ceil(rem0/2), 0, np.ceil(rem2/2), 0], dtype=np.int16))
                
            meta['lcrop']['begin'] += torch.from_numpy(
                    np.array([np.floor(rem0/2), 0, np.floor(rem2/2)], dtype=np.int16))
            meta['lcrop']['end'] += torch.from_numpy(
                    np.array([np.ceil(rem0/2), 0, np.ceil(rem2/2)], dtype=np.int16))

        assert np.all(np.array(initial_dshape) == meta['dcrop']['begin'].numpy()
                + meta['dcrop']['end'].numpy()
                + np.array(data.shape)),\
                'Shapes and Crop do not match'

        assert np.all(np.array(initial_lshape) == meta['lcrop']['begin'].numpy()
                + meta['lcrop']['end'].numpy()
                + np.array(label.shape)),\
                'Shapes and Crop do not match'

        return {'data': data, 'label': label, 'meta': meta}

def to_numpy(V, meta):
    '''
    inverse function for class ToTensor() in dataloader_dev.py 
    args
        V: torch.tensor volume
        meta: batch['meta'] information

    return V as numpy.array volume
    '''
    # torch sizes X is batch size, C is Colour
    # data
    # [X x C x Z x Y] [171 x 3 x 500-crop x 333] (without crop)
    # and for the label
    # [X x Z x Y] [171 x 500 x 333]
    
    # we want to reshape to
    # numpy sizes
    # data
    # [Z x X x Y x 3] [500 x 171 x 333 x 3]
    # label
    # [Z x X x Y] [500 x 171 x 333]
    
    # here: we only need to backtransform labels
    if not isinstance(V, np.ndarray):
        assert isinstance(V, torch.Tensor)
        V = V.numpy()
    V = V.transpose((1, 0, 2))

    # add padding, which was removed before,
    # and saved in meta['lcrop'] and meta['dcrop']
    # parse label crop
    b = (meta['lcrop']['begin']).numpy().squeeze()
    e = (meta['lcrop']['end']).numpy().squeeze()
    
    pad_width = ((b[0], e[0]), (b[1], e[1]), (b[2], e[2]))
    
    return np.pad(V, pad_width, 'edge')

