
# Feed-forward Likelihood Maximization for Gaussian-based Occupancy Prediction [ECCV 2026]

- Feed-forward Likelihood Maximization for Occupancy Prediction (FLM-Occ) learns to iteratively maximize the Gaussian likelihood over the voxel distribution (test-time unrolled optimization).
- This method largely simplifies the implementation of Gaussian-based occupancy prediction, and achieves state-of-the-art performance on the ScanNet-Occ dataset with significantly reduced computational cost.
- The superquadric-based implementation further improves the efficiency and accuracy.

> [arXiv](https://arxiv.org/abs/2606.21373v1) | [Project Page](https://gcchen97.github.io/flm-occ/)


## Installation
```bash
conda create -n flm_occ python=3.12
conda activate flm_occ

pip3 install torch torchvision
pip3 install lightning rotary_embedding_torch
# optional libs for logging
pip3 install tensorboard swanlab
```


## Dataset Preparation
1. Download the [Occ-ScanNet](https://huggingface.co/datasets/hongxiaoy/OccScanNet) dataset and the train/val split files from [EmbodiedOcc/data/occscannet](https://github.com/YkiWu/EmbodiedOcc/tree/main/data/occscannet)
2. Set up the folder structure:
```
/path/to/
├── occscannet/
│   ├── gathered_data/
│   ├── posed_images/
│   ├── train_final.txt
│   ├── train_mini_final.txt
│   ├── test_final.txt
│   ├── test_mini_final.txt
```
3. Modify the dataset path in `configs/scannet.py` or `configs/scannet_mini.py` to point to the downloaded dataset.


## Training
1. Modify the dataset path and depth anything v2 path in `configs/scannet.py` before training.
2. Modify the imported config file in `train.py` if you want to use a different config.
```bash
python train.py
```


## Acknowledgement
This project is built upon the following open-source repositories:
- [EmbodiedOcc](https://github.com/ykiwu/embodiedocc)
- [GaussianFormer](https://github.com/huang-yh/GaussianFormer)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [DINOv2](https://github.com/facebookresearch/dinov2)

Thanks for their contributions to the research community!


## Reference
```
@inproceedings{gcchen2024pisr,
  title={FLM-Occ: Feed-forward Likelihood Maximization for Efficient Indoor Occupancy Prediction},
  author={Guangcheng, Chen and Lihuang, Fang and Huaqi, Tao and Yicheng, He and Li, He and Hong, Zhang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2026},
}
```


## License
This project is licensed under the MIT License, except for third-party code under `third_party/`, which is subject to its original licenses and notices.

