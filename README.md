# GVB
Code release for ["Gradually Vanishing Bridge for Adversarial Domain Adaptation"]() (CVPR 2020)

## Dataset

Office-31 dataset can be found [here](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/). 

Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

VisDA 2017 dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public) in the classification track.

## Requirements
The code is implemented with Python(3.6) and Pytorch(1.0.0).

To install the required python packages, run

```
pip install -r requirements.txt
```

## Training
Training instructions for GVB-GD and CDAN-GD are in the `README.md` in [GVB-GD](GVB-GD) and [CDAN-GD](CDAN-GD) respectively.

## Citation
If you use this code for your research, please consider citing:
```
@inproceedings{cui2020gvb,
  title={Gradually Vanishing Bridge for Adversarial Domain Adaptation},
  author={Cui, Shuhao and Wang, Shuhui and Zhuo, Junbao and Su, Chi and Huang, Qingming and Tian Qi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Contact
If you have any problem about our code, feel free to contact
- hassassin1621@gmail.com

or describe your problem in Issues.
