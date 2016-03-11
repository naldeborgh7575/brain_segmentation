import numpy as np
import simpleitk
import skimage.io as io
from sklearn.cross_validation import train_test_split
from GetFiles import GetFiles

class PickleFormatter(object):
    '''
    INPUT: all slices from a specific pulse sequence, ground truth (labels)
    Creates a tuple of three lists: training set, validation set, testing set
    Each list is a pair: list of scan slices, list of ground truth slices
    '''
    def __init__(self, sequence = 't2', test_prop = 0.2):
        self.sequence = sequence.lower()
        self.test_prop = test_prop
        self.slices = _get_slices_labels_()[0] # all scan images
        self.labels = _get_slices_labels_()[1] # all ground truth images
        self.train_X = None
        self.test_X = None
        self.train_Y = None
        self.test_Y = None


    def _get_slices_labels_(self):
        '''
        IMPUT: MR pulse sequence as a string (t1, t1c, t2, flair)
        OUTPUT: None.
        Writes scans acquired with input sequence into Pickle format
        '''
        slices, labels = [], []
        scan_loc_lst = GetFiles(self.sequence).path_list() # list of paths to each scan
        ground_truth = GetFiles('gt').path_list() # paths to corresponding ground truths

        for path_idx in xrange(len(scan_loc_lst)): # loop through paths, read scans
            scan = io.imread(scan_loc_lst[path_idx], plugin='simpleitk')
            label = io.imread(ground_truth[path_idx], plugin='simpleitk')

            for slice_idx in xrange(len(scan)): # save each slice to list
                slices.append(scan[slice_idx])
                labels.append(label[slice_idx])

        return slices, labels

    def get_train_test(self):
