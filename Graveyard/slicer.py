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
