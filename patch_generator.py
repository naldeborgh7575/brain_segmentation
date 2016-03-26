import numpy as np
import random
from sklearn.feature_extraction.image import extract_patches_2d

class PatchCreator(object):
    def __init__(self, img, num_patches = 50, img_hw = (240,240), patch_hw = (33,33)):
        self.patch_hw = patch_hw
        self.img = self._reshape_img(img)
        self.img_hw = img_hw
        self.img_h = img_hw[0]
        self.img_h = img_hw[1]
        self.patch_h = patch_hw[0]
        self.patch_w = patch_hw[1]

    def _reshape_image(self, img):
        return np.array(img).reshape(self.img.shape[0] / self.img_h, self.img_h, self.img_w)

    def get_patches(self, img):
        all_patches = extract_patches_2d(img[0], patch_hw)
        patch_idx = random.randrange(xrange(len(all_patches)), num_patches)
        self.patches = [all_patches[i] for i in patch_idx]
