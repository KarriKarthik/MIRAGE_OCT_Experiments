# MIRAGE_OCT_Experiments
The repository is a demo of the use of fine-tuning of an OCT foundation model (https://github.com/j-morano/MIRAGE) for a four-class image classification problem.

# Data
The data for this project is available as a public database is not stored in this repository. The data can be downloaded from multiple sources. The data is available under CC BY-NC-SA 4.0 license. The data was originaly mention in a 2018 paper about using deep learning for medical image classification [[1](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)].  
1. Mendeley Data : [Labeled Optical Coherence Tomography (OCT) for Classification](https://data.mendeley.com/datasets/rscbjbr9sj/1)
2. Kaggle : [Retinal OCT Images (optical coherence tomography)](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)

# MIRAGE


[MIRAGE](https://github.com/j-morano/MIRAGE) is a multimodal foundation model for comprehensive retinal OCT/SLO image analysis. It is trained on a large-scale dataset of multimodal data, and is designed to perform a wide range of tasks, including disease staging, diagnosis, and layer and lesion segmentation. MIRAGE is based on the MultiMAE architecture, and is pretrained using a multi-task learning strategy. The model, based on ViT, is available in two sizes: MIRAGE-Base and MIRAGE-Large.

The original code of MIRAGE is available as a Github repository. The development work has be led by José Morano and Hrvoje Bogunović, from the CD-AIR lab of the Medical University of Vienna. The pre-print of the research [[2](https://arxiv.org/abs/2506.08900)], is availbe as an arXiv and has been accepted for publication in npj Digital Medicine.



# References

[1] Kermany, Daniel S., et al. "Identifying medical diagnoses and treatable diseases by image-based deep learning." cell 172.5 (2018): 1122-1131.

[2] Morano, José, et al. "MIRAGE: Multimodal foundation model and benchmark for comprehensive retinal OCT image analysis." arXiv preprint arXiv:2506.08900 (2025).
