import numpy as np
import random
from sklearn.feature_extraction.image import extract_patches_2d

class PatchCreator(object):
    '''
    Takes image and randomly generates patches of given size
    '''
    def __init__(self, img, img_hw = (240,240), patch_hw = (33,33)):
        self.patch_hw = patch_hw
        self.img = self._reshape_norm_image(img)
        self.img_hw = img_hw
        self.img_h = img_hw[0]
        self.img_h = img_hw[1]
        self.patch_h = patch_hw[0]
        self.patch_w = patch_hw[1]
        self.center_idx = (self.patch_h / 2), (self.patch_w / 2)

    def _reshape_image(self, img):
        '''
        break image into idividual modalities
        '''
        img_res = np.array(img).reshape(5, img.shape[0] / 5, img.shape[1])
        for mode in xrange(4):
            normed = self._normalize_slice(img_res[mode])
            img_res[mode] = normed
        return img_res

    def _normalize_slice(self, slice):
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

    def get_patches(self, num_patches = 50):
        all_patches = [extract_patches_2d(self.img[i], self.patch_hw) for i in self.img]
        patch_idx = random.randrange(xrange(len(all_patches)), num_patches)
        self.patches = [[all_patches[j][i] for i in patch_idx] for j in all_patches]

    def even_classes(self):
        tumor_ct, back_ct = 0, 0
        gt = self.patches[-1]
        for i in gt:
            if i[self.center_idx] == 0:
                back_ct += 1
            else:
                tumor_ct += 1
