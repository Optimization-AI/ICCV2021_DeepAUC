# Robust Deep AUC Maximization  [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/abs/2012.03173)

This is the official implementation of the paper "**Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification**" published on **ICCV2021**. 

Requirements
---------
```python
pip install libauc
```

Benchmarks
---------
Benchmark datasets contain Cat&Dog, CIFAR10, CIFAR100, STL10, which can be accessed by our library. 


### Example

#### Importing LibAUC & Loading Datasets
```python
from libauc.datasets import CIFAR10
(train_data, train_label), (test_data, test_label) = CIFAR10()
```

#### Constructing Imbalanced Datasets
```python
from libauc.datasets import imbalance_generator
SEED = 123
imratio = 0.1 # postive_samples/(total_samples)
(train_images, train_labels) = imbalance_generator(train_data, train_label, imratio=imratio, shuffle=True, random_seed=SEED)
(test_images, test_labels) = imbalance_generator(test_data, test_label, is_balanced=True, random_seed=SEED)
```

#### Making Dataloader for Training and Testing 
```python
trainloader = torch.utils.data.DataLoader(ImageDataset(train_images, train_labels), batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
testloader = torch.utils.data.DataLoader( ImageDataset(test_images, test_labels, mode='test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=1,  pin_memory=True)
```

For more details, please refer to [here](https://github.com/yzhuoning/LibAUC/blob/main/examples/02_Optimizing_AUROC_with_ResNet20_on_Imbalanced_CIFAR10.ipynb). 


CheXpert
---------
CheXpert is a large dataset of chest X-rays and competition, which consists of 224,316 chest radiographs of 65,240 patients.The details about the dataset can be found at https://stanfordmlgroup.github.io/competitions/chexpert/. The dataloader used in the paper can be downloaded [here](https://github.com/Optimization-AI/ICCV2021_DeepAUC/blob/main/chexpert.py).

### Example 

```python
root="YOUR_DATA_PATH"
class_id="CLASS_ID"
traindSet = CheXpert(csv_path=root+'train.csv', image_root_path=root, use_frontal=True, image_size=224, mode='train', class_index=class_id)
testSet =  CheXpert(csv_path=root+'valid.csv',  image_root_path=root, use_frontal=True, image_size=224, mode='valid', class_index=class_id)
trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=32, num_workers=2, shuffle=True)
testloader =  torch.utils.data.DataLoader(testSet, batch_size=32, num_workers=2, shuffle=False)
```

### Train DenseNet121 with AUCM Loss on CheXpert 
Please follow this tutorial [Notebook](https://github.com/yzhuoning/LibAUC/blob/main/examples/05_Optimizing_AUROC_Loss_with_DenseNet121_on_CheXpert.ipynb) to train the models. 


Citation
---------
If you find this repo helpful, please cite the following paper:
```
@inproceedings{yuan2021robust,
	title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
	author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
	booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
	year={2021}
	}
```
