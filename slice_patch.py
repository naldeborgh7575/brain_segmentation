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


def find_patches(training_images, class_num, num_samples, patch_size=(65,65)):
    '''
    Method for sampling slices with evenly distributed classes
    INPUT:  (1) list 'training_images': all training images to select from
            (2) int 'class_num': class to sample from choice of {0, 1, 2, 3, 4}.
            (3) tuple 'patch_size': dimensions of patches to be generated defaults to 65 x 65
    OUTPUT: (1) num_samples patches from class 'class_num' randomly selected. note- if class_num is 0, will choose patches randomly, not exclusively 0s.
    '''
    h,w = patch_size[0], patch_size[1]
    patches, labels = [], np.full(num_samples, class_num).astype('float') # Xy
    print 'Finding patches of class {}...'.format(class_num)
    # progress.currval = 0
    ct = 0
    while ct < num_samples:
        im_path = random.choice(training_images)
        fn = os.path.basename(im_path)
        label = io.imread('Labels/' + fn[:-4] + 'L.png')

        # resample if class_num not in selected image
        if len(np.argwhere(label == class_num)) < 10:
            continue
            # im_path = random.choice(training_images)
            # fn = os.path.basename(im_path)
            # label = io.imread('Labels/' + fn[:-4] + 'L.png')

        # select centerpix index (p) and extrapolate patch (p_ix)
        img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float')
        p = random.choice(np.argwhere(label == class_num))
        p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
        patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])

        # if patch is too small, reselect center pix
        if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > (h * w)*2:
            continue

        patches.append(patch)
        ct += 1
    return np.array(patches), labels

def random_patches(im_path, num, patch_size = (65,65)):
    '''
    method for randomly sampling patches from a slice
    INPUT:  (1) string 'im_path': path to image to sample from
            (2) tuple 'patch_size': size (in pixels) of patches.
    '''
    fn = os.path.basename(im_path)
    patch_lst = []
    label = io.imread('Labels/' + fn[:-4] + 'L.png')
    imgs = (io.imread(im_path).reshape(5, 240, 240)[:-1])
    for img in imgs:
        p = extract_patches_2d(img, patch_size, max_patches = num, random_state=5)[0] # set rs for same patch ix among modes
        for i in p:
            if np.std(p) != 0:
                p /= 65535.
            patch_lst.append(p)
        patch = np.array(patch_lst[:-1])
        patch_label = np.array(patch_lst[-1][(patch_size[0] + 1) / 2][(patch_size[1] + 1) / 2]) # center pixel of patch
    return np.array(patch), patch_label

def patches_by_entropy(training_images, num_samples, patch_size=(65,65)):
    '''
    INPUT:  (1) list 'training_images': list of filepaths to training images
            (2) int 'num_samples': total number of patches to extract
            (3) tupel 'patch_size': defaults to 65,65. pixel size of patches to extract
    OUTPUT: (1) numpy array 'patches': high entropy patches (num_samples, n_chan, patch_size)
    Finds high-entropy patches based on label, allows net to learn borders more effctively.
    '''
    h,w = patch_size[0], patch_size[1]
    patches, labels = [], [] #Xy
    ct = 0
    while ct < num_samples:
        im_path = random.choice(training_images)
        fn = os.path.basename(im_path)
        label = io.imread('Labels/' + fn[:-4] + 'L.png')

        # no tumor in ground truth
        while len(np.unique(label)) < 2:
            im_path = random.choice(training_images)
            fn = os.path.basename(im_path)
            label = io.imread('Labels/' + fn[:-4] + 'L.png')

        img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float')
        l_ent = entropy(label, disk(h)) # calculate entropy of assoc patches
        top_ent = np.percentile(l_ent, 80) # list of highest entropy patches
        pix = np.argwhere(l_ent >= top_ent)
        also = np.argwhere(label != 0.) # high entropy patch, non-background
        non_zero = [i for i in also if i in pix]
        # randomly generate 2 patches, select 3 from non-background
        p_s = random.sample(pix, 2)
        p_s += random.sample(non_zero, 3)
        for p in p_s:
            p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
            patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])
            #exclude any patches that are too small
            if np.shape(patch) != (4,65,65):
                continue
            patches.append(patch)
            labels.append(label[p[0],p[1]])
            ct += 1
    return np.array(patches[:num_samples]), np.array(labels[:num_samples])

def center_33(patches):
    '''
    For use with cascaded architecture model with multiple inputs
    INPUT: list 'patches': list of randomly sampled 65x65 patches
    OUTPUT: list of center 33x33 sub-patch for each input patch
    '''
    sub_patches = []
    for mode in patches:
        subs = np.array([patch[16:49, 16:49] for patch in mode])
        sub_patches.append(subs)
    return np.array(sub_patches)

def center_5(patches):
    '''
    INPUT: list 'patches': list of randomly sampled 33x33 patches
    OUTPUT: list of center 33x33 sub-patch for each input patch
    '''
    sub_patches = []
    for mode in patches:
        subs = np.array([patch[14:19, 14:19] for patch in mode])
        sub_patches.append(subs)
    return np.array(sub_patches)

def core_tumor_patches(training_images, num_total, patch_size=(65,65)):
    h, w = patch_size[0], patch_size[1]
    per_class = num_total / 4
    patches, labels = [], []
    for i in xrange(1,5):
        p, l = find_patches(training_images, i, per_class, patch_size = patch_size)
        for img_ix in xrange(len(p)): # 0 <= pixel intensity <= 1
            for slice in xrange(len(p[img_ix])):
                if np.max(p[img_ix][slice]) != 0:
                    p[img_ix][slice] /= np.max(p[img_ix][slice])
        patches.append(p)
        labels.append(l)
    return np.array(patches).reshape(num_total, 4, h, w), np.array(labels).reshape(num_total)

def tumor_134(training_images, num_total, patch_size=(65,65)):
    h, w = patch_size[0], patch_size[1]
    per_class = num_total / 3
    patches, labels = [], []
    for i in [1,3,4]:
        p, l = find_patches(training_images, i, per_class, patch_size = patch_size)
        for img_ix in xrange(len(p)): # 0 <= pixel intensity <= 1
            for slice in xrange(len(p[img_ix])):
                if np.max(p[img_ix][slice]) != 0:
                    p[img_ix][slice] /= np.max(p[img_ix][slice])
        patches.append(p)
        labels.append(l)
    return np.array(patches).reshape((num_total/3)*3, 4, h, w), np.array(labels).reshape((num_total/3)*3)

def make_training_patches(training_images, num_total, balanced_classes = True, half_entropy = False, patch_size = (65,65)):
    '''
    Outputs an X and y to train cnn on
    INPUT   (1) list 'training_images': list of all training images to draw from randomly
            (2) int 'num_total': total number pf patches to produce
            (3) bool 'balanced_classes': True for balanced classes. If false, will randomly sample patches leading to high amnt of background
            (4) # TODO bool 'half_entropy': True if half of batch should be patches based on high-entropy. other half will be balances classes
            (5) tuple 'patch_size': size(in pixels) of patches to select
    OUTPUT  (1) array 'patches': randomly selected patches. shape:
                (num_total, n_chan (4), patch height, patch width)
    '''
    h, w = patch_size[0], patch_size[1]
    if balanced_classes:
        per_class = num_total / 5
        patches, labels = [], []
        for i in xrange(5):
            p, l = find_patches(training_images, i, per_class, patch_size = patch_size)
            for img_ix in xrange(len(p)): # 0 <= pixel intensity <= 1
                for slice in xrange(len(p[img_ix])):
                    p[img_ix][slice] /= np.max(p[img_ix][slice])
            patches.append(p)
            labels.append(l)
        return np.array(patches).reshape(num_total, 4, h, w), np.array(labels).reshape(num_total)

    else:
        patches, labels = random_patches(training_images, num_total, patch_size=patch_size)
        return np.array(patches), np.array(labels)

if __name__ == '__main__':
    train_imgs = glob('train_data/*.png')
    n_patch = int(raw_input('Number of patches to train on: '))
    X, y = make_training_patches(train_imgs, n_patch, patch_size=(65,65))
    # X_5 = center_5(X)
    X_33 = center_33(X)
    # X, y = core_tumor_patches(train_imgs, n_patch)
    # X_33 = center_33(X)
