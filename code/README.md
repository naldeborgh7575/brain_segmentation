# Brain Segmentation Workflow
Below is the basic workflow for the code. [BrainPipeline](https://github.com/naldeborgh7575/brain_segmentation/blob/master/code/brain_pipeline.py), [PatchLibrary](https://github.com/naldeborgh7575/brain_segmentation/blob/master/code/patch_library.py) and [SegmentationModel](https://github.com/naldeborgh7575/brain_segmentation/blob/master/code/Segmentation_Models.py) are classes that handle each of these processes. [n4_bias_correction](https://github.com/naldeborgh7575/brain_segmentation/blob/master/code/n4_bias_correction.py) is a helper function that handles MRI bias and artifact corrections called by BrainPipeline.

<img alt='brain segmentation workflow' src='brain_seg_workflow.png' width=300>  
