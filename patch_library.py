import numpy as np
import random
import os
from glob import glob
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import progressbar
from sklearn.feature_extraction.image import extract_patches_2d

progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])
np.random.seed(5)
train_files = glob('Patches_Train/**')

def find_patches(training_images, class_num, num_samples, patch_size=(65,65)):
    '''
    Method for sampling slices with evenly distributed classes
    INPUT:  (1) list 'training_images': all training images to select from
            (2) int 'class_num': class to sample from choice of {0, 1, 2, 3, 4}.
            (3) tuple 'patch_size': dimensions of patches to be generated defaults to 65 x 65
    OUTPUT: (1) num_samples patches from class 'class_num' randomly selected. note- if class_num is 0, will choose patches randomly, not exclusively 0s.
    '''
    ct = 0 # keep track of patch number
    h,w = patch_size[0], patch_size[1]
    patches = [] #list of all patches (X)
    labels = np.full(num_samples, class_num).astype('float') # y
    print 'Finding patches of class {}...'.format(class_num)
    while ct < num_samples:
        im_path = random.choice(training_images) # select image to sample from
        fn = os.path.basename(im_path)
        label = io.imread('Labels/' + fn[:-4] + 'L.png')
        if class_num not in np.unique(label): # no pixel label class_num in img
            continue
        img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float') # exclude label slice
        p = random.choice(np.argwhere(label == class_num)) # center pixel
        p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2)) # patch index
        patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])
        if len(np.unique(patch)) == 1 or patch.shape != (4,65,65):
            continue # no all-zero or too small patches
        patches.append(patch) # patch = (n_chan, h, w)
        ct += 1
    return np.array(patches), labels

def patches_by_entropy(training_images, num_samples, patch_size=(65,65)):
    '''
    INPUT:  (1) list 'training_images': list of filepaths to training images
            (2) int 'num_samples': total number of patches to extract
            (3) tupel 'patch_size': defaults to 65,65. pixel size of patches to extract
    OUTPUT: (1) numpy array 'patches': high entropy patches (num_samples, n_chan, patch_size)
    Finds high-entropy patches based on label, allows net to learn borders more effctively.
    '''
    ct = 0 # keep track of patch number
    h,w = patch_size[0], patch_size[1]
    patches = [] #list of all patches (X)
    labels = [] # y
    while ct < num_samples:
        im_path = random.choice(training_images) # select image to sample from
        fn = os.path.basename(im_path)
        label = io.imread('Labels/' + fn[:-4] + 'L.png')
        if len(np.unique(label)) <= 2: # no pixel label class_num in img
            continue
        img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float') # exclude label slice
        l_ent = entropy(label, disk(65))
        top_ent = np.percentile(l_ent, 80)
        pix = np.argwhere(l_ent >= top_ent)
        also = np.argwhere(label != 0.)
        non_zero = [i for i in also if i in pix]
        p_s = random.sample(pix, 3)
        p_s += random.sample(non_zero, 2)
        for p in p_s:
            p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
            patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])
            if np.shape(patch) != (4,65,65):
                continue
            patches.append(patch)
            labels.append(label[p[0],p[1]])
            ct += 1
    return np.array(patches[:num_samples]), np.array(labels[:num_samples])

def random_patches(im_path, num, patch_size = (65,65)):
    '''
    method for randomly sampling patches from a slice
    INPUT:  (1) string 'im_path': path to image to sample from
            (2) tuple 'patch_size': size (in pixels) of patches.
    '''
    fn = os.path.basename(im_path)
    patch_lst = []
    label = io.imread('Labels/' + fn[:-4] + 'L.png')
    imgs = (io.imread(im_path).reshape(5, 240, 240))
    imgs[4] = label
    for img in imgs:
        p = extract_patches_2d(img, patch_size, max_patches = num, random_state=5)[0]
    for i in p:
        if np.std(p) != 0:
            p = (p - np.mean(p)) / np.std(p)
        patch_lst.append(p) #set rs for same patch ix among modes
    patch = np.array(patch_lst[:-1])
    patch_label = np.array(patch_lst[-1][(patch_size[0] + 1) / 2][(patch_size[1] + 1) / 2]) # center pixel of patch
    return np.array(patch), patch_label

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
                    p[p_l] /= np.max(p[p_l])
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

def center_33(patches):
    '''
    INPUT: list 'patches': list of randomly sampled patches
    OUTPUT: list of center 33x33 sub-patch for each input patch
    '''
    sub_patches = []
    for mode in patches:
        subs = np.array([patch[16:49, 16:49] for patch in mode])
        sub_patches.append(subs)
    return np.array(sub_patches)

if __name__ == '__main__':
    train_imgs = glob('train_data/*.png')
    # test_imgs=glob('Patches_Test/*.png')
    n_patch = int(raw_input('Number of patches to train on:'))
    # n_train = int(raw_input('Number of patches to train on:'))
    X, y = make_training_patches(train_imgs, n_patch)
    X_33 = center_33(X)

    # X_test, y_test = make_training_patches(test_imgs, n_train)
    # X_33_test = center_33(X_33_test)
