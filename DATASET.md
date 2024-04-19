## Dataset

The overall directory structure should be:
```
│MaskLRF/
├──cfgs/
├──data/
│   ├──ModelNetFewshot/
│   ├──OmniObject3D/
│   ├──ScanObjectNN/
│   ├──ShapeNet55_withnormal/
│   ├──ShapeNetPart/
├──datasets/
├──.......
```

### ModelNet Few-shot Dataset (for finetuning):
```
│ModelNetFewshot/
├──5way10shot/
│  ├── 0.pkl
│  ├── ...
│  ├── 9.pkl
├──5way20shot/
│  ├── ...
├──10way10shot/
│  ├── ...
├──10way20shot/
│  ├── ...
```

Download: Please download the data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md). We use the same data split as theirs.

### OmniObject3D Dataset (for finetuning):
```
│OmniObject3D/
├──5way10shot/
│  ├── 0.pkl
│  ├── ...
│  ├── 9.pkl
├──5way20shot/
│  ├── ...
├──10way10shot/
│  ├── ...
├──10way20shot/
│  ├── ...
```
TODO

### ScanObjectNN Dataset (for finetuning):
```
│ScanObjectNN/
├──main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
├──main_split_nobg/
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
```
Download: Please download the data from the [official website](https://hkust-vgd.github.io/scanobjectnn/).

### ShapeNet55 Dataset (for pretraining):
```
│ShapeNet55_withnormal/
├──shapenetcore55_2048pts_withnormal_all.h5
├──shapenetcore55_2048pts_withnormal_test.h5
```

Download: Please download the data from [Google Drive]().

### ShapeNetPart Dataset:

```
|shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──02691156/
│  ├── 1a04e3eab45ca15dd86060f189eb133.txt
│  ├── .......
│── .......
│──train_test_split/
│──synsetoffset2category.txt
```

Download: Please download the data from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). 
