import numpy as np
import h5py
import skimage.io as io
from sklearn.cross_validation import train_test_split
from GetFiles import GetFiles

class GetSlices(object):
    '''
    INPUT: MR pulse sequence as a string (default to t2), test size (default to 0.2)
    Creates a tuple of two lists: training set, testing set from BraTS dataset
    Each list is a pair: list of scan slices, list of ground truth slices
    '''
    def __init__(self, sequence = 't2', test_size = 0.2, limit = None):
        self.sequence = sequence.lower()
        self.test_size = test_size
        self.slices = None # all scan images
        self.labels = None # all ground truth images
        self.train_X = None
        self.test_X = None
        self.train_Y = None
        self.test_Y = None
        self.limit = limit
        self._get_slices_labels_()
        self._get_train_test_()

    def _get_slices_labels_(self):
        '''
        Creates slices and labels lists
        '''
        slices, labels = [], []
        scan_loc_lst = GetFiles(self.sequence, limit = self.limit).path_list() # list of paths to each scan
        ground_truth = GetFiles(sequence = 'gt').path_list() # paths to corresponding ground truths

        for path_idx in xrange(len(scan_loc_lst)): # loop through paths, read scans
            scan = io.imread(scan_loc_lst[path_idx], plugin='simpleitk')
            label = io.imread(ground_truth[path_idx], plugin='simpleitk')

            for slice_idx in xrange(len(scan)): # save each slice to list
                slices.append(scan[slice_idx])
                labels.append(label[slice_idx])

        self.slices = np.array(slices)
        self.labels = np.array(labels)

    def _get_train_test_(self):
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(self.slices, self.labels, test_size = self.test_size)

    def save_h5f(self, filename):
        h5f = h5py.File(filename, 'w')
        h5f.create_dataset('X_train', data = self.train_X)
        h5f.create_dataset('Y_train', data = self.train_Y)
        h5f.create_dataset('X_test', data = self.test_X)
        h5f.create_dataset('Y_test', data = self.test_Y)
        h5f.close()

def load_data(filename):
    '''
    INPUT: filename- existing h5 file.
    OUTPUT: data: X_train, X_test, Y_train, Y_test
    '''
    h5f = h5py.File(filename, 'r')
    X_train = h5f['X_train'][:]
    X_test = h5f['X_test'][:]
    Y_train = (h5f['Y_train'][:]
    Y_test = h5f['Y_test'][:])
    h5f.close()
    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    slices = GetSlices()


    ## GRAVEYARD ##

    # pickles.save_pickle('T2_weighted')

    # def save_pickle(self, filename):
    #     train = [self.train_X, self.train_Y]
    #     test = [self.test_X, self.test_Y]
    #     doc = (train, test)
    #     with open(filename, 'wb') as f:
    #         pickle.dump(doc, f)
