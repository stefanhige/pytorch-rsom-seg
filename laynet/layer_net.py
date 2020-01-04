import os
import sys
import copy
import warnings
import numpy as np
from scipy import ndimage

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from ._model import UNet
from ._dataset import RSOMLayerDatasetUnlabeled, \
                      ZeroCenter, CropToEven, DropBlue, \
                      ToTensor, to_numpy
from utils import save_nii

class LayerNetBase():
    """
    stripped base class for predicting RSOM layers.
    for training user class LayerNet
    Args:
        device             torch.device()     'cuda' 'cpu'
        dirs               dict of string      use these directories
        filename           string              pattern to save output
    """
    def __init__(self, 
                 dirs={'train':'', 'eval':'', 'pred':'', 'model':'', 'out':''},
                 device=torch.device('cuda'),
                 model_depth=4
                 ):

        self.model_depth = model_depth
        self.dirs = dirs

        self.pred_dataset = RSOMLayerDatasetUnlabeled(
                dirs['pred'],
                transform=transforms.Compose([
                    ZeroCenter(), 
                    CropToEven(network_depth=self.model_depth),
                    DropBlue(),
                    ToTensor()])
                )

        self.pred_dataloader = DataLoader(
            self.pred_dataset,
            batch_size=1, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True)


        self.size_pred = len(self.pred_dataset)

        self.minibatch_size = 1
        self.device = device
        self.dtype = torch.float32
        
        self.model = UNet(in_channels=2,
                          n_classes=2,
                          depth=self.model_depth,
                          wf=6,
                          padding=True,
                          batch_norm=True,
                          up_mode='upconv').to(self.device)
        
        if self.dirs['model']:
            print('Loading U-Net model from:', self.dirs['model'])
            self.model.load_state_dict(torch.load(self.dirs['model']))
        

    def predict(self):
        self.model.eval()
        iterator = iter(self.pred_dataloader) 

        for i in range(self.size_pred):
            
            batch = next(iterator)
            
            batch['data'] = batch['data'].to(
                    self.device,
                    self.dtype,
                    non_blocking=True
                    )
            
            # divide into minibatches
            minibatches = np.arange(batch['data'].shape[1],
                                    step=self.minibatch_size)
            shp = batch['data'].shape
            # [0 x 2 x 500 x 332]
            prediction_stack = torch.zeros((0, 2, shp[3], shp[4]),
                    dtype=self.dtype,
                    requires_grad=False
                    )
            prediction_stack = prediction_stack.to(self.device)

            for i2, idx in enumerate(minibatches):
                if idx + self.minibatch_size < batch['data'].shape[1]:
                    data = batch['data'][:, idx:idx+self.minibatch_size, :, :]
                else:
                    data = batch['data'][:, idx:, :, :]
     
                data = torch.squeeze(data, dim=0)

                prediction = self.model(data)

                prediction = prediction.detach() 
                prediction_stack = torch.cat((prediction_stack, prediction), dim=0) 
            
            prediction_stack = prediction_stack.to('cpu')
            
            # transform -> labels
            label = (prediction_stack[:,1,:,:] > prediction_stack[:,0,:,:]) 

            label = to_numpy(label, batch['meta'])

            filename = batch['meta']['filename'][0]
            filename = filename.replace('rgb.nii.gz','')
            label = self.smooth_pred(label, filename)

            print('Saving epidermis prediction', filename + 'pred')
            save_nii(label, self.dirs['out'], filename + 'pred')

    @staticmethod
    def smooth_pred(label, filename):
        '''
        smooth the prediction
        '''
        # fill holes inside the label
        ldtype = label.dtype
        label = ndimage.binary_fill_holes(label).astype(ldtype)
        label_shape = label.shape
        label = np.pad(label, 2, mode='edge')
        label = ndimage.binary_closing(label, iterations=2)
        label = label[2:-2,2:-2,2:-2]
        assert label_shape == label.shape
        
        # get 2x 2-D surface data with surface height being the index in z-direction
        
        surf_lo = np.zeros((label_shape[1], label_shape[2]))
        
        # set highest value possible (500) as default. Therefore, empty sections
        # of surf_up and surf_lo will get smoothened towards each other, and during
        # reconstructions, we won't have any weird shapes.
        surf_up = surf_lo.copy()+label_shape[0]

        for xx in np.arange(label_shape[1]):
            for yy in np.arange(label_shape[2]):
                nz = np.nonzero(label[:,xx,yy])
                
                if nz[0].size != 0:
                    idx_up = nz[0][0]
                    idx_lo = nz[0][-1]
                    surf_up[xx,yy] = idx_up
                    surf_lo[xx,yy] = idx_lo
       
        # smooth coarse structure, 
        # eg with a 26x26 average and crop everything which is above average*factor
        #   -> hopefully spikes will be removed.
        surf_up_m = ndimage.median_filter(surf_up, size=(26, 26), mode='nearest')
        surf_lo_m = ndimage.median_filter(surf_lo, size=(26, 26), mode='nearest')
        
        for xx in np.arange(label_shape[1]):
            for yy in np.arange(label_shape[2]):
                if surf_up[xx,yy] < surf_up_m[xx,yy]:
                    surf_up[xx,yy] = surf_up_m[xx,yy]
                if surf_lo[xx,yy] > surf_lo_m[xx,yy]:
                    surf_lo[xx,yy] = surf_lo_m[xx,yy]

        # apply suitable kernel in order to smooth
        # smooth fine structure, eg with a moving average
        surf_up = ndimage.uniform_filter(surf_up, size=(9, 5), mode='nearest')
        surf_lo = ndimage.uniform_filter(surf_lo, size=(9, 5), mode='nearest')

        # reconstruct label
        label_rec = np.zeros(label_shape, dtype=np.uint8)
        for xx in np.arange(label_shape[1]):
            for yy in np.arange(label_shape[2]):

                label_rec[int(np.round(surf_up[xx,yy])):int(np.round(surf_lo[xx,yy])),xx,yy] = 1     

        return label_rec
