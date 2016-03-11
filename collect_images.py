import numpy as np
import simpleitk
import skimage.io as io
from GetFiles import GetFiles

class PickleFormatter(object):
    def __init__(self, ):





def sequence_to_pickle(sequence):
    '''
    IMPUT: MR pulse sequence as a string (t1, t1c, t2, flair)
    OUTPUT: None.
    Writes scans acquired with input sequence into Pickle format
    '''
    slices, labels = [], [] # slices from scans and ground_truths
    seq = sequence.lower()
    scan_loc_lst = GetFiles(seq).path_list() # list of paths to each scan
    ground_truth = GetFiles('gt').path_list() # paths to corresponding ground truths

    for path_idx in xrange(len(scan_loc_lst)): # loop through paths, read scans
        scan = io.imread(scan_loc_lst[path_idx], plugin='simpleitk')
        label = io.imread(ground_truth[path_idx], plugin='simpleitk')

        for slice_idx in xrange(len(scan)): # save each slice to list
            slices.append(scan[slice_idx])
            labels.append(label[slice_idx])

    return slices, labels
