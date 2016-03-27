import numpy as np
from glob import glob
from skimage import io

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
        self.slices_by_mode = self.read_scans()[0]
        # [ [slice1 x 5], [slice2 x 5], ..., [slice155 x 5]]
        self.slices_by_slice = self.read_scans()[1]
        self.normed_slices = np.zeros((155, 5, 240, 240))

    def read_scans(self):
        '''
        goes into each modality in patient directory and loads individual scans. transforms scans of same slice into strip of 5 images
        '''
        # import pdb; pdb.set_trace()
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
        for slice_ix in xrange(155):
            self.normed_slices[slice_ix][-1] = self.slices_by_slice[slice_ix][-1]
            for mode_ix in xrange(4):
                self.normed_slices[slice_ix][mode_ix] =  self._normalize(self.slices_by_slice[slice_ix][mode_ix])

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

    def save(self, keyword):
        pass
