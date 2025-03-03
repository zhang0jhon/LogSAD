# Towards Training-free Anomaly Detection with Vision and Language Foundation Models (CVPR 2025)

By Jinjin Zhang, Guodong Wang, Yizhou Jin, Di Huang.


## Installation

Install the required packages:

```
pip install -r requirements.txt
```


Download the checkpoint for [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and put in the checkpoint folder.


## Instructions for MVTEC LOCO dataset

### Few-shot Protocol
Run the script for few-shot protocal:

```
python evaluation.py --module_path model_ensemble_few_shot --category CATEGORY  --dataset_path DATASET_PATH
```

### Full-data Protocol
Run the script to compute coreset for full-data scenarios:

```
python compute_coreset.py --module_path model_ensemble --category CATEGORY  --dataset_path DATASET_PATH
```

Run the script for full-data protocol:

```
python evaluation.py --module_path model_ensemble --category CATEGORY  --dataset_path DATASET_PATH
```


## Acknowledgement
We are grateful for the following awesome projects when implementing LogSAD:
* [SAM](https://github.com/facebookresearch/segment-anything), [OpenCLIP](https://github.com/mlfoundations/open_clip), [DINOv2](https://github.com/facebookresearch/dinov2) and [NACLIP](https://github.com/sinahmr/NACLIP).


## Citation
If you find our paper is helpful in your research or applications, generously cite with
```
@inproceedings{zhang2025logsad,
      title={Towards Training-free Anomaly Detection with Vision and Language Foundation Models},
      author={Jinjin Zhang, Guodong Wang, Yizhou Jin, Di Huang},
      year={2025},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    }
```
