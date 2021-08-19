# Robust Deep AUC Maximization  [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/abs/2012.03173)

This is the official implementation of the paper "**Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification**" published on **ICCV2021**. 

Requirement
---------
`
pip install libauc
`

CheXpert
---------
CheXpert is a large dataset of chest X-rays and competition, which consists of 224,316 chest radiographs of 65,240 patients.The details about the dataset can be found at https://stanfordmlgroup.github.io/competitions/chexpert/.


Train DenseNet121 with AUCM Loss on CheXpert 
---------
Please follow this tutoroial [Notebook](https://github.com/yzhuoning/LibAUC/blob/main/examples/05_Optimizing_AUROC_Loss_with_DenseNet121_on_CheXpert.ipynb). The dataloader can be downloaded [here](https://github.com/Optimization-AI/ICCV2021_DeepAUC/blob/main/chexpert.py)

Example 
---------
```
root="YOUR_DATA_PATH"
class_id="CLASS_ID"
traindSet = CheXpert(csv_path=root+'train.csv', image_root_path=root, use_frontal=True, image_size=224, mode='train', class_index=class_id)
testSet =  CheXpert(csv_path=root+'valid.csv',  image_root_path=root, use_frontal=True, image_size=224, mode='valid', class_index=class_id)
trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=32, num_workers=2, shuffle=True)
testloader =  torch.utils.data.DataLoader(testSet, batch_size=32, num_workers=2, shuffle=False)
```

Citation
---------
If you find this repo helpful in your work, please cite the following paper:
```
@inproceedings{yuan2021robust,
	title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
	author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
	booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
	year={2021}
	}
```
