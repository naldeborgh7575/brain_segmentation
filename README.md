# Automatic Multimodal Brain Tumor Segmentation

### Background:

High-grade malignant brain tumors are generally associated with a short life expectancy and limited treatment options. The aggressive nature of this illness necessitates efficient diagnosis and treatment planning to improve quality of and extend patient life. Magnetic Resonance Imaging (MRI) is a common technique for assessing brain tumors and determining next steps, but the large quantity of data produced by these scans prohibits precise manual segmentation in a feasible amount of time.

<img src="images/segmented_slice.png" width='400'>

There is therefore a need for reliable and automatic segmentation methods in clinical settings. However, brain tumors are structurally and spatially diverse by nature, which makes this a challenging problem that has yet to be adequately conquered.

<img src="images/tumor_diversity.png" width='250'>

### Dataset

### Model

I present a four-layer Convolutional Neural Network (CNN) model that that requires minimal pre-processing and can distinguish healthy tissue, actively enhancing tumor and non-advancing tumor regions. The model is trained on randomly selected 33x33 patches of MRI images. Each input has 4 channels, one for each imaging modality.
