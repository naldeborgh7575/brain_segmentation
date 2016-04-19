# Automatic Multimodal Brain Tumor Segmentation

Brain tumor segmentation seeks to separate healthy tissue from tumorous regions such as the advancing tumor, necrotic core and surrounding edema. This is an essential step in diagnosis and treatment planning, both of which need to take place quickly in the case of a malignancy in order to maximize the likelihood of successful treatment. Due to the slow and tedious nature of manual segmentation, there is a high demand for computer algorithms that can do this quickly and accurately.



## MRI Background:

Magnetic Resonance Imaging (MRI) is the most common diagnostic tool brain tumors due primarily to it's noninvasive nature and ability to image diverse tissue types and physiological processes. MRI uses a magnetic gradient and radio frequency pulses to take repetitive axial 'slices' of the brain and produce a 3-dimensional representation of the brain (Figure 1). Each brain scan 155 slices, with each pixel representing a 1mm<sup>3</sup> 'voxel.'  

<img alt="Basic MRI Workflow" src="images/MRI_workflow.png" width=450>
<img alt="3D rendering produced by T2 MRI scan" src="images/t29_143.gif" width=250>  
<sub> <b> Figure 1: </b> (Left) Basic MRI workflow. Slices are taken axially at 1mm increments, creating the 3-dimensional rendering (right). Note that this is only one of four commonly-used modalities used for tumor segmentation. </sub>

There are multiple radio frequency pulse sequences that can be used to illuminate different classes of tissues. For adequate segmentation there are often four different scans acquired: Flair, T1, T1-contrasted, and T2 (Figure 2). Each of these pulse sequences takes advantage of the chemical and physiological characteristics of specific parts of the brain. Notice the variability in contrast of the four images in Figure 2, which are all the same slice of the same brain, but with different regions of the tumor seen at brighter intensities.

<img alt="The four MRI modalities used in brain tumor segmentation: Flair, T1, T1-contrasted and T2" src="images/modalities.png" width=200>
<img src="images/brain_grids.png" width=650>  
<sub><b> Figure 2: </b> (Right) Flair (top left), T1, T1C and T2 (bottom right) pulse sequences. (Left) Representative scans from each tumor modality. Approximately 600 images need to be analyzed per brain for a segmentation.

Notice now that a single patient will produce upwards of 600 images from a single MRI, given that all four modalities produce 155 slices each. To get an acceptably accurate segmented brain manually, a radiologist has to spend hours in front of a computer tediously determining which voxels belong to which class.


## Background:

Automatic tumor segmentation has the potential to decrease lag time between diagnostic tests and treatment by providing an efficient and standardized report of tumor location in a fraction of the time it would take a radiologist to do so.

High-grade malignant brain tumors are generally associated with a short life expectancy and limited treatment options. The aggressive nature of this illness necessitates efficient diagnosis and treatment planning to improve quality of and extend patient life. Magnetic Resonance Imaging (MRI) is a common technique for assessing brain tumors and determining next steps, but the large quantity of data produced by these scans prohibits precise manual segmentation in a feasible amount of time.



<img alt="Example of tumor segmentation overlay on T2" src="images/segmented_slice.png" width='400'>

There is therefore a need for reliable and automatic segmentation methods in clinical settings. However, brain tumors are structurally and spatially diverse by nature, which makes this a challenging problem that has yet to be adequately conquered.

<img alt="Diversity of tumor size, shape and location" src="images/tumor_diversity.png" width='400'>


### Model

I use a four-layer Convolutional Neural Network (CNN) model that that requires minimal pre-processing and can distinguish healthy tissue, actively enhancing tumor and non-advancing tumor regions.  The local invariant nature of CNNs allows for abstraction of token features for classification without relying on large-scale spatial information that is not consistent in tumor location.

<img alt="Basic ConvNet model architecture" src="images/model_architecture.png" width=800>

The model is trained on randomly selected 33x33 patches of MRI images in order to classify the center pixel. Each input has 4 channels, one for each imaging modality (T1, T1c, T2 and Flair).

### Results

<img alt="Result Frame" src="images/results.png" width=404>

<img alt='Ground Truth: Professional Segmentation' src='images/gt.gif' width=200>
<img alt='Results of CNN Model' src='images/my_res.gif' width=200>

### Dataset

This dataset was provided by the [2015 MICCAI BraTS Challenge](http://www.braintumorsegmentation.org)
