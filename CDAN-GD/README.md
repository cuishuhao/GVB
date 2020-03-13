# CDAN-GD implemneted in PyTorch

## Prerequisites
- pytorch = 1.0.1 
- torchvision = 0.2.1
- numpy = 1.17.2
- pillow = 6.2.0
- python3.6
- cuda10

## Training
The following are the command for each task. The GVBG and GVBD represents the parameter for GVB on the generator and discriminator. if GVBG==0, GVBG is not utilized for the network. The test_interval can be changed, which is the number of iterations between near test. The num_iterations can be changed, which is the total training iteration number.

Office-31
```
python train_image.py CDAN+E --gpu_id 0 --GVBG 1 --GVBD 1 --num_iterations 8004  --dset office --s_dset_path data/office/amazon_list.txt --t_dset_path data/office/dslr_list.txt --test_interval 500 --output_dir gvbgd/adn
```

Office-Home
```
python train_image.py CDAN+E --gpu_id 0 --GVBG 1 --GVBD 1 --num_iterations 8004  --dset office-home --s_dset_path data/office-home/Art.txt --t_dset_path data/office-home/Clipart.txt --test_interval 500 --output_dir gvbgd/ArCl
```

VisDA 2017
```
python train_image.py CDAN+E --gpu_id 0 --GVBG 1 --GVBD 1 --num_iterations 15002 --dset visda --s_dset_path data/visda-2017/train_list.txt --t_dset_path data/visda-2017/validation_list.txt --test_interval 1000  --output_dir gvbgd
```

The codes are heavily borrowed from [CDAN](https://github.com/thuml/CDAN)
