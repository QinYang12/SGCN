# SGCN
Code for the paper "Rotation Equivariant Graph Convolutional Network for Spherical Image Classification", CVPR 2020

## Requirements
* Python3 == 3.6.5
* Tensorflow == 1.10.0

## Usages
1.Install the dependencies
~~~
pip install -r requirements.txt
~~~
2.Generate the dataset
~~~
python3.6 data_generate.py
~~~

3.Train and test the model
~~~
python3.6 main.py
~~~

## Citation
If you find this code is useful for your research, please cite our paper "Rotation Equivariant Graph Convolutional Network for Spherical Image Classification"

~~~
@inproceedings{yang2020rotation,
  title={Rotation Equivariant Graph Convolutional Network for Spherical Image Classification},
  author={Yang, Qin and Li, Chenglin and Dai, Wenrui and Zou, Junni and Qi, Guo-Jun and Xiong, Hongkai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4303--4312},
  year={2020}
}
~~~
