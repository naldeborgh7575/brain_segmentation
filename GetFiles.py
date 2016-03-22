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

if __name__ == '__main__':
    paths = GetFiles(sequence = 'all').path_list()
    # to_png(paths) ## don't run this again!


## GRAVEYARD ##

# def save_png(sequence = 'all'):
#     paths = GetFiles(sequence).path_list() # list of all paths to scans
#     seqs = flair, t1, t1c, t2, gt = [],[],[],[],[] # lists of paths by pulse sequence
#     seqs_str = ['flair', 't1', 't1c', 't2', 'gt']
#     for scan_idx in xrange(len(paths)):
#         if scan_idx % 5 == 0:
#             flair.append(paths[scan_idx])
#         elif (scan_idx - 1) % 5 == 0:
#             t1.append(paths[scan_idx])
#         elif (scan_idx - 2) % 5 == 0:
#             t1c.append(paths[scan_idx])
#         elif (scan_idx - 3) % 5 == 0:
#             t2.append(paths[scan_idx])
#         else:
#             gt.append(scan_idx)
#     for sequence_idx in xrange(5): # index of sequence in seq list
#         for scan_idx in xrange(len(seqs[sequence_idx])): #index of scan file in sequence list
#             scan = io.imread(seqs[sequence_idx][scan_idx], plugin='simpleitk') # read scan file
#             for slice_idx in xrange(len(scan)): # index of slice(image) within scan
#                 imsave(seqs_str[sequence_idx] + str(scan_idx) + str(slice_idx) + '.png', scan[slice_idx])

    # for scan_idx in xrange(len(paths)):
    #     scan = io.imread(scan_idx, plugin='simpleitk')
    #     for slice_idx in xrange(len(scan)):
    #         imsave()
