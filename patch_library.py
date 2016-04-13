import numpy as np
import random
import os
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import progressbar
from sklearn.feature_extraction.image import extract_patches_2d

progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])
np.random.seed(5)


class PatchLibrary(object):
    def __init__(self, patch_size, num_samples, train_data classes=[0,1,2,3,4]):
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.train_data = train_data
        self.classes = classes
        self.h = self.patch_size[0]
        self.w = self.patch_size[1]

    def find_patches(self, class_num, num_patches):
        '''
        Helper function for sampling slices with evenly distributed classes
        INPUT:  (1) list 'training_images': all training images to select from
                (2) int 'class_num': class to sample from choice of {0, 1, 2, 3, 4}.
                (3) tuple 'patch_size': dimensions of patches to be generated defaults to 65 x 65
        OUTPUT: (1) num_samples patches from class 'class_num' randomly selected.
        '''
        h,w = self.patch_size[0], self.patch_size[1]
        patches, labels = [], np.full(num_patches, class_num, 'float')
        print 'Finding patches of class {}...'.format(class_num)
        progress.currval = 0
        for i in progress(xrange(num_patches)):
            im_path = random.choice(self.train_data)
            fn = os.path.basename(im_path)
            label = io.imread('Labels/' + fn[:-4] + 'L.png')

            # resample if class_num not in selected slice
            while len(np.argwhere(label == class_num)) < 10:
                im_path = random.choice(self.train_data)
                fn = os.path.basename(im_path)
                label = io.imread('Labels/' + fn[:-4] + 'L.png')

            # select centerpix (p) and patch (p_ix)
            img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float')
            p = random.choice(np.argwhere(label == class_num))
            p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
            patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])

            # resample it patch is empty or too close to edge
            while patch.shape != (4, h, w) or len(np.unique(patch)) == 1:
                p = random.choice(np.argwhere(label == class_num))
                p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
                patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])

            patches.append(patch)
        return np.array(patches), labels

    def center_n(self, n, patches):
        '''
        Takes list of patches and returns center nxn for each patch. Use as input for cascaded architectures.
        INPUT   (1) int 'n': size of center patch to take (square)
                (2) list 'patches': list of patches to take subpatch of
        OUTPUT: list of center nxn patches.
        '''
        sub_patches = []
        for mode in patches:
            subs = np.array([patch[(self.h/2) - (n/2):(self.h/2) + ((n+1)/2),(self.w/2) - (n/2):(self.w/2) + ((n+1)/2)] for patch in mode])
            sub_patches.append(subs)
        return np.array(sub_patches)

    def slice_to_patches(self, filename):
        '''
        Converts an image to a list of patches with a stride length of 1. Use as input for image prediction.
        INPUT: str 'filename': path to image to be converted to patches
        OUTPUT: list of patched version of imput image.
        '''
        slices = io.imread(filename).astype('float').reshape(5,240,240)[:-1]
        plist=[]
        for slice in slices:
            if np.max(img) != 0:
                img /= np.max(img)
            p = extract_patches_2d(img, (h,w))
            plist.append(p)
        return np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]), np.array(plist[3])))

    def patches_by_entropy(self, num_patches=self.num_samples):
        patches, labels = [], []
        ct = 0
        while ct < num_patches:
            im_path = random.choice(training_images)
            fn = os.path.basename(im_path)
            label = io.imread('Labels/' + fn[:-4] + 'L.png')

            # pick again if slice is only background
            if len(np.unique(label)) == 1:
                continue
            img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float')
            l_ent = entropy(label, disk(self.h))
            ct += 1


def make_training_patches(training_images, num_total, balanced_classes = True, patch_size = (65,65)):
    '''
    Outputs an X and y to train cnn on
    INPUT   (1) list 'training_images': list of all training images to draw from randomly
            (2) int 'num_total': total number pf patches to produce
            (3) bool 'balanced_classes': True for balanced classes. If false, will randomly sample patches leading to high amnt of background
            (4) tuple 'patch_size': size(in pixels) of patches to select
    OUTPUT  (1) array 'patches': randomly selected patches. shape:
                (num_total, n_chan (4), patch height, patch width)
    '''
    if balanced_classes:
        per_class = num_total / 5
        patches, labels = [], [] # list of tuples (patches, label)
        for i in xrange(5):
            p, l = find_patches(training_images, i, per_class, patch_size=patch_size)
            for p_l in xrange(len(p)):
                if np.max(p[p_l]) != 0:
                    p[p_l] /= 65535.
            patches.append(p)
            labels.append(l)
        # print 'Finding high-entropy patches...'
        # for i in progress(xrange(4)):
        #     p_e,l_e = patches_by_entropy(training_images, num_total/8)
        #     patches.append(p_e)
        #     labels.append(l_e)
        # import pdb; pdb.set_trace()
        return np.array(patches).reshape(np.shape(patches)[0]*np.shape(patches)[1], 4, patch_size[0], patch_size[1]), np.array(labels).reshape(np.shape(patches)[0]*np.shape(patches)[1])
    else:
        patches, labels = random_patches(training_images, num_total, patch_size=patch_size)
        return np.array(patches), np.array(labels)


if __name__ == '__main__':
    pass
