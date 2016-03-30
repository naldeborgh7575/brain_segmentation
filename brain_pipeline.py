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
        scans = glob(self.path + '/**/*.mha') # directories to each image (5 total)
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
        INPUT:  (1) a single slice of any given modality (excluding gt)
                (2) index of modality assoc with slice (0=flair, 1=t1, 2=t1c, 3=t2)
        OUTPUT: normalized slice
        '''
        b, t = np.percentile(slice, (0.5,99.5))
        slice = np.clip(slice, b, t)
        if np.std(slice) == 0:
            return slice
        else:
            return (slice - np.mean(slice)) / np.std(slice)

    def generate_patches(self, patch_size=(65,65), num_patches = 50):
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

    def save_patient(self, patient_num):
        '''
        INPUT: int 'patient_num': unique identifier for each patient
        OUTPUT: saves png in Norm_PNG directory
        '''
        print 'Saving scans for patient {}...'.format(patient_num)
        for slice_ix in xrange(155):
            strip = self.slices_by_slice[slice_ix].reshape(1200, 240)
            if np.max(strip) != 0:
                strip /= np.max(strip)
            io.imsave('Training_PNG/{}_{}.png'.format(patient_num, slice_ix), strip)

    def save_norm_patient(self, patient_num):
        '''
        INPUT: int 'patient_num': unique identifier for each patient
        OUTPUT: saves png in Norm_PNG directory
        '''
        print 'Saving scans for patient {}...'.format(patient_num)
        for slice_ix in xrange(155):
            strip = self.normed_slices[slice_ix].reshape(1200, 240)
            if np.max(strip) != 0:
                strip /= np.max(strip)
            if np.min(strip) <= -1:
                strip /= abs(np.min(strip))
            io.imsave('Norm_PNG/{}_{}.png'.format(patient_num, slice_ix), strip)

    def n4itk_norm(self, path, n_dims=3, n_iters=[20,20,10,5]):
        '''
        INPUT:  (1) filepath 'path': path to mha T1 or T1c file
                (2) directory 'parent_dir': parent directory to mha file
        OUTPUT: writes n4itk normalized image to parent_dir under orig_filename_n.mha
        '''
        output_fn = path[:-4] + '_n.mha'
        run n4_bias_correction.py path n_dim n_iters output_fn

if __name__ == '__main__':
    patients = glob('Training/HGG/**')
    for patient_num, path in enumerate(patients):
        a = BrainPipeline(path)
        a.save_norm_patient(patient_num)
        # a.save_patient(patient_num)
    # TO DO
    # resize images?
    # make generator to train net in batches (~32)
