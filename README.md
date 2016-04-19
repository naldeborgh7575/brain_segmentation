# Automatic Multimodal Brain Tumor Segmentation

Brain tumor segmentation seeks to separate healthy tissue from tumorous regions such as the advancing tumor, necrotic core and surrounding edema. This is an essential step in diagnosis and treatment planning, both of which need to take place quickly in the case of a malignancy in order to maximize the likelihood of successful treatment. Due to the slow and tedious nature of manual segmentation, there is a high demand for computer algorithms that can do this quickly and accurately.



## MRI Background:

Magnetic Resonance Imaging (MRI) is the most common diagnostic tool brain tumors due primarily to it's noninvasive nature and ability to image diverse tissue types and physiological processes. MRI uses a magnetic gradient and radio frequency pulses to take repetitive axial 'slices' of the brain (Figure 1). Each brain scan produces 155 images.

<img alt="Basic MRI Workflow" src="images/MRI_workflow.png" width=450>
<img alt="3D rendering produced by T2 MRI scan" src="images/t29_143.gif" width=250>
<sub> <b> Figure 1: </b> (Left) Basic MRI workflow. Slices are taken axially at 1mm increments, creating the 3-dimensional rendering on the right. </sub>




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
