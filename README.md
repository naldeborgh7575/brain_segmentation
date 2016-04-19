# Automatic Brain Tumor Segmentation

Brain tumor segmentation seeks to separate healthy tissue from tumorous regions such as the advancing tumor, necrotic core and surrounding edema. This is an essential step in diagnosis and treatment planning, both of which need to take place quickly in the case of a malignancy in order to maximize the likelihood of successful treatment. Due to the slow and tedious nature of manual segmentation, there is a high demand for computer algorithms that can do this quickly and accurately.

## Table of Contents
1. [Dataset](#dataset)
2. [MRI Background](#mri-background)
    * [Pulse Sequences](#pulse-sequences)
    * [Segmentation](#segmentation)
3. [High Grade Gliomas](#high-grade-gliomas)
4. [Convolutional Neural Networks](#convolutional-neural-networks)
    * [Implementation](#implementation)
5. [Results](#results)

## Dataset

All MRI data was provided by the [2015 MICCAI BraTS Challenge](http://www.braintumorsegmentation.org), which consists of approximately 250 high-grade glioma cases and 50 low-grade cases. However, due to the limited time  Each dataset contains four different MRI [pulse sequences](http://radiopaedia.org/articles/mri-sequences-overview), each of which is comprised of 155 brain slices, for a total of 620 images per patient. Professional segmentation is provided as ground truth labels for each case. Figure 1 is an example of a scan with the ground truth segmentation. The segmentation labels are represented as follows:

<img alt="Example of tumor segmentation overlay on T2" src="images/segmented_slice.png" width='400'>
<sub><b>Figure 1: </b> Ground truth segmentation overlay on a T2 weighted scan. </sub>   


## MRI Background

Magnetic Resonance Imaging (MRI) is the most common diagnostic tool brain tumors due primarily to it's noninvasive nature and ability to image diverse tissue types and physiological processes. MRI uses a magnetic gradient and radio frequency pulses to take repetitive axial slices of the brain and construct a 3-dimensional representation(Figure 2). Each brain scan 155 slices, with each pixel representing a 1mm<sup>3</sup> 'voxel.'  

<img alt="Basic MRI Workflow" src="images/MRI_workflow.png" width=450>
<img alt="3D rendering produced by T2 MRI scan" src="images/t29_143.gif" width=250>  
<sub> <b> Figure 2: </b> (Left) Basic MRI workflow. Slices are taken axially at 1mm increments, creating the 3-dimensional rendering (right). Note that this is only one of four commonly-used pulse sequences used for tumor segmentation. </sub>

### Pulse sequences
There are multiple radio frequency pulse sequences that can be used to illuminate different types of tissue. For adequate segmentation there are often four different unique sequences acquired: Fluid Attenuated Inversion Recovery (FLAIR), T1, T1-contrasted, and T2 (Figure 3). Each of these pulse sequences exploits the distinct chemical and physiological characteristics of various tissue types, resulting in contrast between the individual classes. Notice the variability in intensities among the four images in Figure 3, all of which are images of the same brain taken with different pulse sequences.

<img alt="The four MRI sequences used in brain tumor segmentation: Flair, T1, T1-contrasted and T2" src="images/modalities.png" width=200>  
<sub><b> Figure 3: </b> Flair (top left), T1, T1C and T2 (bottom right) pulse sequences. </sub>

### Segmentation
Notice now that a single patient will produce upwards of 600 images from a single MRI, given that all four sequences produce 155 slices each (Figure 4). To get a satisfactory manual segmentation a radiologist must spend several hours tediously determining which voxels belong to which class. In the setting of malignant brain tumors, an algorithmic alternative would give clinicians more time focusing on the wellbeing of the patient, allowing for more immediate patient care and higher throughput treatment times.

<img alt="All images produced from a single patient brain scan." src="images/brain_grids.png" width=800>  

<img alt="Results of the complete segmentation of a single brain" src='images/segment.png' width=800>  
<sub> <b>Figure 4:</b> (Top) Representative scans from each tumor imaging sequence. Approximately 600 images need to be analyzed per brain for a segmentation. (Bottom) The results of a complete tumor segmentation.</sub>

Automatic tumor segmentation has the potential to decrease lag time between diagnostic tests and treatment by providing an efficient and standardized report of tumor location in a fraction of the time it would take a radiologist to do so.

## High Grade Gliomas

<b>Glioblastoma cases each year (US)</b><sup>[4](#references)</sup>: 12,000  
<b>Median survival</b>: 14.6 months  
<b>Five-year survival rate</b>: < 10%

High-grade malignant brain tumors are generally associated with a short life expectancy and limited treatment options. The aggressive nature of this illness necessitates efficient diagnosis and treatment planning to improve quality of and extend patient life. This urgency reinforces thee need for reliable and fast automatic segmentation methods in clinical settings. Unfortunately, algorithmic segmentation of these particular tumors has proven to be a very challenging task, due primarily to the fact that they tend to be very structurally and spatially diverse (Figure 5).

<img alt="Diversity of tumor size, shape and location" src="images/tumor_diversity.png" width='400'>  
<sub><b>Figure 5: </b> Three different examples of high grade gliomas, tumor segmentations are outlined on the bottom images. Notice the variation in size, shape and location in the brain, a quality of these tumors that makes them difficult to segment. </sub>

## Convolutional Neural Networks

Convolutional Neural Networks(CNNs) are a powerful tool in the field of image recognition. They were inspired in the late 1960s by the elucidation of how the [mammalian visual cortex works](https://en.wikipedia.org/wiki/Receptive_field): many networks neurons sensitive to a given 'receptive field' tiled over the entire visual field<sup>[1](#references)</sup>. This aspect of CNNs contributes to their high flexibility and spatial invariance, making them ideal candidates for semantic segmentatiaon of images with high disparity in locations of objects of interest. CNNs are a powerful tool in machine learning that are well suited for the challenging problem tackled in this project.

### Implementation

I use a four-layer Convolutional Neural Network (CNN) model that that requires minimal pre-processing and can distinguish healthy tissue, actively enhancing tumor and non-advancing tumor regions (Figure 6).  The local invariant nature of CNNs allows for abstraction of token features for classification without relying on large-scale spatial information that is inconsistent in the case of tumor location.

<img alt="Basic ConvNet model architecture" src="images/model_architecture.png" width=800>  
<sub><b>Figure 6: </b> Basic model architecture of my segmentation model. Input is four 33x33 patches from a randomly selected slice. Each imaging pulse sequence is input as a channel into the net, followed by four convolution/max pooling layers (note- the last convolutional layer is not followed by max pooling). </sub>

The model is trained on randomly selected 33x33 patches of MRI images in order to classify the center pixel. Each input has 4 channels, one for each imaging pulse sequence (T1, T1c, T2 and Flair).

### Results

<img alt="Result Frame" src="images/results.png" width=404>

<img alt='Ground Truth: Professional Segmentation' src='images/gt.gif' width=200>
<img alt='Results of CNN Model' src='images/my_res.gif' width=200>

### References

    1. Hubel, D. and Wiesel, T. (1968). Receptive fields and functional architecture of monkey striate cortex. Journal of Physiology (London), 195, 215–243.
    2. Kistler et. al, The virtual skeleton database: an open access repository for biomedical research and collaboration. JMIR, 2013.
    3. Menze et al., The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS), IEEE Trans. Med. Imaging, 2015.
    4. Stupp et al., Effects of radiotherapy with concomitant and adjuvant temozolomide versus radiotherapy alone on survival in glioblastoma in a randomised phase III study: 5-year analysis of the EORTC-NCIC trial, The Lancet Onc., 2009.
