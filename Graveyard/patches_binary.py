import numpy as np
import random
import os
from glob import glob
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.feature_extraction.image import extract_patches_2d

def find_patches_binary(training_images, num_samples, patch_size=(33,33)):
    ct = 0
    h,w = patch_size[0], patch_size[1]
    patches, labels = [], [] # X, y
    print 'Finding tumorous patches...'
    while ct < num_samples / 2:
        im_path = random.choice(training_images)
        fn = os.path.basename(im_path)
        label = io.imread('Labels/' + fn[:-4] + 'L.png')
        if len(np.unique(label)) == 1:
            continue
        imgs = io.imread(im_path).reshape(5,240,240)
        label[label > 0] = 1
        imgs[4] = label
        p = random.choice(np.argwhere(label == 1)) # center pixel
        if p[0] < (h/2 + 1) or (240 - p[0]) < (h/2 + 1) or p[1] < (h/2 + 1) or  (240 - p[1]) < (h/2 + 1): # if patch won't fit
            continue
        p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2)) # patch index
        patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in imgs[:-1]])
        if len(np.unique(patch)) == 1:
            continue
        patches.append(patch)
        labels.append(1)
        ct += 1

    print 'Finding healthy patches...'
    while ct < num_samples:
        im_path = random.choice(training_images)
        fn = os.path.basename(im_path)
        label = io.imread('Labels/' + fn[:-4] + 'L.png')
        if len(np.unique(label)) == 1:
            continue
        imgs = io.imread(im_path).reshape(5,240,240)
        label[label > 0] = 1
        imgs[4] = label
        p = random.choice(np.argwhere(label == 0)) # center pixel
        if p[0] < (h/2 + 1) or (240 - p[0]) < (h/2 + 1) or p[1] < (h/2 + 1) or  (240 - p[1]) < (h/2 + 1): # if patch won't fit
            continue
        p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
        patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in imgs[:-1]])
        patches.append(patch)
        labels.append(0)
        ct += 1
    print 'Done'
    return np.asarray(patches), labels

def patch_generator():
    pass
if __name__ == '__main__':
    train_imgs = glob('Patches_Train/*.png')
    X, y = find_patches_binary(train_imgs, 100)
