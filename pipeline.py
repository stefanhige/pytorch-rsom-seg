
# process raw files

import os
import numpy as np
from scipy import ndimage
import nibabel as nib
import copy
import torch

import shutil

from prep import Rsom, RsomVessel
from utils import get_unique_filepath

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from laynet import LayerNetBase

from vesnet import VesNetBase

from visualization.mip_label_overlay import mip_label_overlay

# define folder
def vessel_pipeline(dirs={'input':'',
                          'output':'',
                          'laynet_model':'',
                          'vesnet_model':''},
                    device=torch.device('cpu'),
                    laynet_depth=4,
                    divs=(1,1,1),
                    ves_probability=0.9,
                    pattern=None,
                    delete_tmp=False):

    os.environ["CUDA_VISIBLE_DEVICES"]='3'

    tmp_layerseg_prep = os.path.join(dirs['output'], 'tmp', 'layerseg_prep')
    tmp_layerseg_out = os.path.join(dirs['output'], 'tmp', 'layerseg_out')
    tmp_vesselseg_prep = os.path.join(dirs['output'], 'tmp', 'vesselseg_prep')
    tmp_vesselseg_out = os.path.join(dirs['output'], 'tmp', 'vesselseg_out')
    tmp_vesselseg_prob = os.path.join(dirs['output'], 'tmp', 'vesselseg_out_prob')

    for tmp_dir in [os.path.join(dirs['output'], 'tmp'),
                    tmp_layerseg_prep,
                    tmp_layerseg_out,
                    tmp_vesselseg_prep,
                    tmp_vesselseg_out,
                    tmp_vesselseg_prob]:

        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        else:
            print(tmp_dir, 'exists already.')
    
    # mode
    if pattern == None:
        cwd = os.getcwd()
        os.chdir(dirs['input'])
        all_files = os.listdir()
        os.chdir(cwd)
    else:
        if isinstance(pattern, str):
            pattern = [pattern]
        all_files = [os.path.basename(get_unique_filepath(dirs['input'], pat)[0]) for pat in pattern]

    # ***** PREPROCESSING FOR LAYER SEGMENTATION *****
    
    filenameLF_LIST = [el for el in all_files if el[-6:] == 'LF.mat']

    print('...............................................')
    print('Starting vessel segmentation pipeline..')
    print('Device is', device)
    print('Files to be processed:')
    for fl in filenameLF_LIST:
        print(fl.replace('LF.mat',' {LF.mat, HF.mat}'))

    print("...............................................")
    print("Starting epidermis segmentation..")

    for idx, filenameLF in enumerate(filenameLF_LIST):
        filenameHF = filenameLF.replace('LF.mat','HF.mat')
        
        # extract datetime
        idx_1 = filenameLF.find('_')
        idx_2 = filenameLF.find('_', idx_1+1)
        filenameSurf = 'Surf' + filenameLF[idx_1:idx_2+1] + '.mat'
        
        fullpathHF = os.path.join(dirs['input'], filenameHF)
        fullpathLF = os.path.join(dirs['input'], filenameLF)
        fullpathSurf = os.path.join(dirs['input'], filenameSurf)
        
        Sample = Rsom(fullpathLF, fullpathHF, fullpathSurf)

        Sample.prepare()

        Sample.save_volume(tmp_layerseg_prep, fstr = 'rgb')
        print('Processing file', idx+1, 'of', len(filenameLF_LIST))

    # ***** LAYER SEGMENTATION *****

    LayerNetInstance = LayerNetBase(dirs={'model': dirs['laynet_model'],
                                          'pred': tmp_layerseg_prep,
                                          'out': tmp_layerseg_out},
                                    model_depth=laynet_depth,
                                    device=device)

    LayerNetInstance.predict()
        
    print("...............................................")
    print("Starting vessel segmentation..")
    # ***** PREPROCESSING FOR VESSEL SEGMENTATION *****
    for idx, filenameLF in enumerate(filenameLF_LIST):
        filenameHF = filenameLF.replace('LF.mat','HF.mat')
        
        # extract datetime
        idx_1 = filenameLF.find('_')
        idx_2 = filenameLF.find('_', idx_1+1)
        filenameSurf = 'Surf' + filenameLF[idx_1:idx_2+1] + '.mat'
        
        fullpathHF = os.path.join(dirs['input'], filenameHF)
        fullpathLF = os.path.join(dirs['input'], filenameLF)
        fullpathSurf = os.path.join(dirs['input'], filenameSurf)
        
        Sample = RsomVessel(fullpathLF, fullpathHF, fullpathSurf)
        
        Sample.prepare(tmp_layerseg_out, mode='pred', fstr='pred.nii.gz')
        Sample.save_volume(tmp_vesselseg_prep, fstr = 'v_rgb')
        print('Processing file', idx+1, 'of', len(filenameLF_LIST))
        
    # ***** VESSEL SEGMENTATION *****
    _dirs={'train': '',
          'eval': '', 
          'model': dirs['vesnet_model'], 
          'pred': tmp_vesselseg_prep,
          'out': tmp_vesselseg_out}

    VesNetInstance = VesNetBase(device=device,
                                dirs=_dirs,
                                divs= divs,
                                ves_probability=ves_probability)

    VesNetInstance.predict(use_best=False, 
                           save_ppred=True)

    # if save_ppred==True   ^
    # we need to move the probability tensors to another folder
    files = [f for f in os.listdir(tmp_vesselseg_out) if 'ppred' in f]
    
    for f in files:
        shutil.move(os.path.join(tmp_vesselseg_out, f), 
                    os.path.join(tmp_vesselseg_prob, f))

    # ***** VISUALIZATION *****
    _dirs = {'in': dirs['input'],
            'layer': tmp_layerseg_out,
            'vessel': tmp_vesselseg_out,
            'out': dirs['output'] }
    
    mip_label_overlay(None, _dirs, plot_epidermis=True)
    
    if delete_tmp:
        shutil.rmtree(os.path.join(dirs['output'],'tmp'))


if __name__ == '__main__':

    dirs = {'input': './data/input',
            'laynet_model': './data/models/unet_depth5.pt',
            'vesnet_model': './data/models/vesnet_gn.pt',
            'output': './data/output'}

    dirs = {k: os.path.expanduser(v) for k, v in dirs.items()}

    # os.environ["CUDA_VISIBLE_DEVICES"]='3'
    
    vessel_pipeline(dirs=dirs,
                    laynet_depth=5,
                    ves_probability=0.96973,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    # pattern=['20200101000000'],  # if list, use these patterns
                                                   # otherwise whole directory
                    divs=(1,1,2),
                    delete_tmp=False)
