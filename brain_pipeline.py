import numpy as np
np.random.seed(5) # for reproducibility
import random
from glob import glob
from skimage import io
from sklearn.feature_extraction.image import extract_patches_2d

class BrainPipeline(object):
    '''
    A class for processing brain scans for each patient
    INPUT: path to directory of one patient. Contains mha following mha files:
    flair, t1, t1c, t2, ground truth (gt)
    '''
    def __init__(self, path):
        self.path = path
        self.modes = ['flair', 't1', 't1c', 't2', 'gt']
        # slices=[[flair x 155], [t1], [t1c], [t2], [gt]], 155 per modality
        self.slices_by_mode, n = self.read_scans()
        # [ [slice1 x 5], [slice2 x 5], ..., [slice155 x 5]]
        self.slices_by_slice = n
        self.normed_slices = self.norm_slices()

    def read_scans(self):
        '''
        goes into each modality in patient directory and loads individual scans. transforms scans of same slice into strip of 5 images
        '''
        # import pdb; pdb.set_trace()
        print 'Loading scans...'
        slices_by_mode = np.zeros((5, 155, 240, 240))
        slices_by_slice = np.zeros((155, 5, 240, 240))
        scans = glob(self.path + '**/*.mha') # directories to each image (5 total)
        for scan_idx in xrange(len(scans)):
            # read each image directory, save to self.slices
            slices_by_mode[scan_idx] = io.imread(scans[scan_idx], plugin='simpleitk').astype(float)
        for mode_ix in xrange(slices_by_mode.shape[0]): # modes 1 thru 5
            for slice_ix in xrange(slices_by_mode.shape[1]): # slices 1 thru 155
                slices_by_slice[slice_ix][mode_ix] = slices_by_mode[mode_ix][slice_ix] # reshape by slice
        return slices_by_mode, slices_by_slice

    def norm_slices(self):
        '''
        normalizes each slice in self.slices_by_slice, excluding gt
        subtracts mean and div by std dev for each slice
        clips top and bottom one percent of pixel intensities
        '''
        print 'Normalizing slices...'
        normed_slices = np.zeros((155, 5, 240, 240))
        for slice_ix in xrange(155):
            normed_slices[slice_ix][-1] = self.slices_by_slice[slice_ix][-1]
            for mode_ix in xrange(4):
                normed_slices[slice_ix][mode_ix] =  self._normalize(self.slices_by_slice[slice_ix][mode_ix])
        print 'Done.'
        return normed_slices

    def _normalize(self, slice):
        '''
        INPUT: a single slice of any given modality (excluding gt)
        OUTPUT: normalized slice
        '''
        b, t = np.percentile(slice, (1,99))
        slice = np.clip(slice, b, t)
        if np.std(slice) == 0:
            return slice
        else:
            return (slice - np.mean(slice)) / np.std(slice)

    def generate_patches(self, patch_size=(33,33), num_patches = 50):
        '''
        INPUT:  (1) tuple 'patch_size': dimensions of patches to be used in net
                (2) int 'num_patches': number of patches to be generated per slice.
        OUTPUT: (1) list of scan patches: (num_slices * num_patches, num_channels, patch_h, patch_w)
                (2) list of label patches: (num_slices * num_patches, patch_h, patch_w)
        '''
        self.patches = [] # (num_scans, num_patches, 4 modes, patch_h, patch_w)
        self.patch_labels = [] # (num_scans, num_patches, patch_h, patch_w)
        patch_list = [] # list of lists: patches for each slice (same idxs)
        for slice_strip in self.normed_slices: # slice = strip of 5 images
            slices = slice_strip.reshape(5,240,240)
            for img in slices:
                # get list of patches corresponding to each image in slices
                patch_list.append(extract_patches_2d(img, patch_size, max_patches = num_patches, random_state=5)) #set rs for same patch ix among modes
            self.patches.append(zip(patch_list[0], patch_list[1], patch_list[2], patch_list[3]))
            self.patch_labels.append(patch_list[-1])
