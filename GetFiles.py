from glob import glob

class GetFiles(object):
    '''
    INPUT: (string) pulse sequence OR ground truth, grade
    pulse can be T1, T2, T1c, all, gt (for ground truth), grade = hgg, lgg or both
    defaults to t2 and both, respectively
    OUTPUT: all BRATS data for given sequence
    '''
    def __init__(self, sequence = 't2', grade = 'both', limit = None):
        self.sequence = sequence.lower()
        self.grade = grade
        self.limit = limit

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
