# DL_Stu_Demo
⏰**记录自己做过的一些DL实验**
## DenseNet_CIFAR-10
- 该项目完整文件如上
- 本实验测试了DenseNet网络模型在CIFAR-10数据集上的表现
- cifar10.py为数据处理文件
- Densenet_Cifar10.py为主程序，运行即可
- 未解决小bug：cifar10.py文件中，在解压部分，可能会报”不是gzip文件“，可手动解压即可

## ResNet50_CIFAR-10
- 在学习了ResNet的基本结构和思想之后，使用**ResNet50+TensorFlow+CIAFR-10**来深入学习网络结构。
- 该项目所使用的环境如下：
  - **Python3.6+TensorFlow1.13+numpy1.16**
  - **CUDA 8.061 + CUDNN 5.1 +tensorflow_gpu 1.2.0**
- 博文地址：[CIFAR-10_ResNet50](https://yearing1017.site/2019/09/30/CIFAR10-ResNet50/)
- 项目简介：
  - ResNet_model.py：建立ResNet50网络的结构，返回经过该网络训练后的结果
  - CIFARHelper.py：对CIFAR10数据集进行一些特定的处理，及加载数据等功能
  - Main.py：主函数，结合上述文件，使用残差网络对数据集进行训练

## Tensorflow_Discuz_验证码识别
使用深度学习框架Tensorflow训练出一个用于破解Discuz验证码的模型
- 环境：**GTX1080的显卡，GPU，安装tensorflow_gpu版本，CUDA，CUDNN**。
- 运行train.py即可训练完模型
- 关于本项目的具体博文地址：[Tensorflow_Discuz](https://yearing1017.site/2019/10/23/Tensorflow-Discuz%E9%AA%8C%E8%AF%81%E7%A0%81%E8%AF%86%E5%88%AB/)
- Discuz验证码数据集下载地址：
  - [链接](https://pan.baidu.com/s/10TzEvjToYjOgzlOpIgx1tg&shfl=sharepset)  密码:6qto
  
## Kaggle_Cat&Dog
- 本项目是kaggle中的入门级项目，本篇博文采用**Keras+CNN**网络模型完成该项目。
- kaggle的项目地址：[链接](https://www.kaggle.com/c/dogs-vs-cats/overview)
- kaggle_NoteBook: [notebook地址](https://www.kaggle.com/yearing1017/keras-cnn)
- 详细博文地址：[kaggle_Cat&Dog](https://yearing1017.site/2019/11/18/Kaggle-Cat-Dog/)

## Kaggle_Cat&Dog_TransferLearning
- 本项目是kaggle中的入门级项目，本实验采用**Keras+InceptionV3**网络模型完成该项目。
- 本实验采用了迁移学习的思想，借助已有的网络模型，不进行训练而直接进行识别任务。
- kaggle的项目地址：[链接](https://www.kaggle.com/c/dogs-vs-cats/overview)
- kaggle_NoteBook: [notebook地址](https://www.kaggle.com/yearing1017/dogs-vs-cats-inceptionv3-fine-tuning)
- 详细博文地址：[kaggle_Cat&Dog_迁移学习](https://yearing1017.site/2019/11/19/Kaggle-Cat-Dog-%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0/)
