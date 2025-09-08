Work done during my time at CATEC as part of the project to create a complete workflow to detect defects in an automated way through the use of deep learning models. The scripts are organized as follows:

1: Classification: Training of the models to do binary detection (images with defects / images without defects). We use transfer learning to adapt pretrained models to fit our task.

2: Segmentation: Training for defect segmentation. There are two different approaches, supervised and unsupervised. In the supervised case we train the networks with masks which show the models where the defects are located. In the unsupervised case we use a ceVAE and we train it with defect-free images. Then, when the model tries to reconstruct images with defects, the loss is higher in the defect areas, and we can detect and segment the defects through this defect map.
