import os
import numpy as np
from glob import glob
from skimage import io
from scipy.misc import imsave

class GetFiles(object):
    '''
    INPUT: (string) pulse sequence OR ground truth, grade
    pulse can be T1, T2, T1c, all, gt (for ground truth), grade = hgg, lgg or both
    defaults to t2 and both, respectively
    OUTPUT: all BRATS data for given sequence
    '''
    def __init__(self, sequence = 'all', grade = 'both', limit = None):
        self.sequence = sequence.lower()
        self.grade = grade
        self.limit = limit

    def _get_gt_(self):
        gr = self.grade
        dir = 'Training/'
        if self.grade == 'both':
            gr = '**'
        return glob(dir + gr + '/**/VSD.Brain_*/*.mha')

    def _get_t2_(self):
        gr = self.grade
        dir = 'Training/'
        if self.grade == 'both':
            gr = '**'
        return glob(dir + gr + '/**/VSD.Brain.XX.O.MR_T2*/*.mha')

    def _get_t1_(self):
        gr = self.grade
        dir = 'Training/'
        if self.grade == 'both':
            gr = '**'
        return glob(dir + gr + '/**/VSD.Brain.XX.O.MR_T1.*/*.mha')

    def _get_t1c_(self):
        gr = self.grade
        dir = 'Training/'
        if self.grade == 'both':
            gr = '**'
        return glob(dir + gr + '/**/VSD.Brain.XX.O.MR_T1c*/*.mha')

    def _get_flair_(self):
        gr = self.grade
        dir = 'Training/'
        if self.grade == 'both':
            gr = '**'
        return glob(dir + gr + '/**/VSD.Brain.XX.O.MR_Flair*/*.mha')

    def _get_all_(self):
        gr = self.grade
        dir = 'Training/'
        if self.grade == 'both':
            gr = '**'
        return glob(dir + gr + '/**/**/*.mha')

    def path_list(self):
        if not self.limit:
            self.limit = len(self._get_t2_())
        if self.sequence == 't2':
            return self._get_t2_()[:self.limit]
        elif self.sequence == 't1':
            return self._get_t1_()[:self.limit]
        elif self.sequence == 't1c':
            return self._get_t1c_()[:self.limit]
        elif self.sequence == 'flair':
            return self._get_flair_()[:self.limit]
        elif self.sequence == 'gt':
            return self._get_gt_()[:self.limit]
        elif self.sequence == 'all':
            return self._get_all_()
        else:
            return 'please initialize with a valid sequence, ground truth ("gt"), or "all"'


def to_png(paths):
    '''
    INPUT: list of file paths leading to MR images
    Creates png version of MR scans for each slice.
    Saves png to file path where original scan is found
    '''
    for path in paths:
        scan = io.imread(path, plugin='simpleitk')
        # loop thru slices, save as png in respective file path
        for slice_ix in xrange(len(scan)):
            imsave(path + str(slice_ix) + '.png', scan[slice_ix])

def reshape_scans(patient_paths):
    '''
    INPUT: filepath for each patient
    OUTPUT: each scan as vertically stacked image of channels + ground truth
    '''
    for patient in xrange(len(patient_paths)):
        for slice in xrange(155):
            # get each channel for individual slices
            modalities = glob(patient_paths[patient] + '/**/*mha{}.png'.format(slice))
            channels = np.array([io.imread(mode) for mode in modalities])
            # reshape 5 channels into vertical strip of channels
            channels = channels.reshape(channels.shape[0] * channels.shape[1], channels.shape[2])
            io.imsave('Training_PNG/{}_'.format(patient)+str(slice)+'.png', channels)

def normalize_scans(files):
    '''
    INPUT: list of png files to normalize
    subtract mean and divde by standard deviation for each slice.
    save normalized images to directory Normed_PNG/
    '''
    for img in files: # strip of images
        slices = io.imread(img).astype(float)
        slices = slices.reshape(5, 240,240)
        for mode in xrange(4):
            normed = normalize_slice(slices[mode])
            slices[mode] = normed
        slice = slices.reshape((5 * 240), 240)
        imsave('Normed_png/' + img[13:-4] + '_n.png', slice)

def normalize_slice(slice):
    '''
    INPUT: a single slice in a single modality
    subtracts the mean, divides by standard deviation
    removes top and bottom 1 percent intensities
    '''
    l, h = np.percentile(slice, (1,99))
    slice = np.clip(slice, l, h)
    if np.std(slice) == 0:
        return slice
    else:

        return (slice - np.mean(slice)) / np.std(slice)

if __name__ == '__main__':
    paths = GetFiles(sequence = 'all').path_list()
    patient_paths = glob('Training/*GG/**/')
    files = glob('Training_PNG/*.png')
    # to_png(paths) ## don't run this again!
    # reshape_scans(patient_paths) Don't rerun this either!
    # normalize_scans(files) # don't run again
