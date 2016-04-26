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
    def __init__(self, patch_size, train_data, num_samples):
        '''
        class for creating patches and subpatches from training data to use as input for segmentation models.
        INPUT   (1) tuple 'patch_size': size (in voxels) of patches to extract. Use (33,33) for sequential model
                (2) list 'train_data': list of filepaths to all training data saved as pngs. images should have shape (5*240,240)
                (3) int 'num_samples': the number of patches to collect from training data.
        '''
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.train_data = train_data
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

        ct = 0
        while ct < num_patches:
            im_path = random.choice(self.train_data)
            fn = os.path.basename(im_path)
            label = io.imread('Labels/' + fn[:-4] + 'L.png')

            # resample if class_num not in selected slice
            # while len(np.argwhere(label == class_num)) < 10:
            #     im_path = random.choice(self.train_data)
            #     fn = os.path.basename(im_path)
            #     label = io.imread('Labels/' + fn[:-4] + 'L.png')
            if len(np.argwhere(label == class_num)) < 10:
                continue

            # select centerpix (p) and patch (p_ix)
            img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float')
            p = random.choice(np.argwhere(label == class_num))
            p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
            patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])

            # resample it patch is empty or too close to edge
            # while patch.shape != (4, h, w) or len(np.unique(patch)) == 1:
            #     p = random.choice(np.argwhere(label == class_num))
            #     p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
            #     patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])
            if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > (h * w):
                continue

            patches.append(patch)
            ct += 1
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

    def patches_by_entropy(self, num_patches):
        '''
        Finds high-entropy patches based on label, allows net to learn borders more effectively.
        INPUT: int 'num_patches': defaults to num_samples, enter in quantity it using in conjunction with randomly sampled patches.
        OUTPUT: list of patches (num_patches, 4, h, w) selected by highest entropy
        '''
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
            top_ent = np.percentile(l_ent, 90)

            # restart if 80th entropy percentile = 0
            if top_ent == 0:
                continue

            highest = np.argwhere(l_ent >= top_ent)
            p_s = random.sample(highest, 3)
            for p in p_s:
                p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
                patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])
                # exclude any patches that are too small
                if np.shape(patch) != (4,65,65):
                    continue
                patches.append(patch)
                labels.append(label[p[0],p[1]])
            ct += 1
            return np.array(patches[:num_samples]), np.array(labels[:num_samples])

    def make_training_patches(self, entropy=False, balanced_classes=True, classes=[0,1,2,3,4]):
        '''
        Creates X and y for training CNN
        INPUT   (1) bool 'entropy': if True, half of the patches are chosen based on highest entropy area. defaults to False.
                (2) bool 'balanced classes': if True, will produce an equal number of each class from the randomly chosen samples
                (3) list 'classes': list of classes to sample from. Only change default oif entropy is False and balanced_classes is True
        OUTPUT  (1) X: patches (num_samples, 4_chan, h, w)
                (2) y: labels (num_samples,)
        '''
        if balanced_classes:
            per_class = self.num_samples / len(classes)
            patches, labels = [], []
            progress.currval = 0
            for i in progress(xrange(len(classes))):
                p, l = self.find_patches(classes[i], per_class)
                # set 0 <= pix intensity <= 1
                for img_ix in xrange(len(p)):
                    for slice in xrange(len(p[img_ix])):
                        if np.max(p[img_ix][slice]) != 0:
                            p[img_ix][slice] /= np.max(p[img_ix][slice])
                patches.append(p)
                labels.append(l)
            return np.array(patches).reshape(self.num_samples, 4, self.h, self.w), np.array(labels).reshape(self.num_samples)

        else:
            print "Use balanced classes, random won't work."



if __name__ == '__main__':
    pass
