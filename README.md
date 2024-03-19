# GradMF
This is a torch implementation of the paper."TARGET OPTIMIZATION DIRECTION GUIDED TRANSFER LEARNING FOR IMAGE
CLASSIFICATION"
 by Kelvin Ting Zuo Han, Shengxuming Zhang, Gerard Marcos Freixas, Zunlei Feng, Cheng Jin.

 Fudan University,ZheJiang University,Innovation Center of Calligraphy and Painting Creation Technolog

In this paper, we propose a new transfer learning method guided by the direction of objective optimization from the perspective of gradient. This
method guides the gradient direction of the source task towards the gradient direction of the target task. In several similar and conflicting tasks, this method has achieved good results in efficiency and performance. In comparison with other transfer learning methods, the results shown by this method are generally better.

<div align="center">
 <img src="https://github.com/KelvinTingZuoHan/GradMF/blob/fc07faeea2fffd070495cf7ae712ff8b73073120/GradMF%20.png" height="200px">
 <img src="https://github.com/KelvinTingZuoHan/GradMF/blob/2efef0f4b7ad28d41cd60b9934fa2f470e186036/Exploded%20view.png" height="200px">
</div>

## Installation

1. Clone our repository

   
2. Make conda environment

   ```
   conda create -n GradMF python=3.8
   conda activate GradMF
   ```
   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
   pip3 install matplotlib scikit-learn scikit-image opencv-python numpy pdb cv2 einops scipy struct
   ```
## Data Preparation

1. Download the TransferLearning dataset(SVHN\Mnist\Office31\caltech256...)

2. Using dataLoader for data preprocessing
   ```
   python MnistDataLoader.py or python SVHN.py
   If you want to process other data, you can design a new dataloader to handle the data yourself
   ```
**Note:** Our method requires the application of both source domain data and target domain data as training data for the network, so we hope to train the processed data at the same size.

## Training!


1. Attempting to use GradMF for training
   ```
   python train_Mnist_SVHN.py or Other training codes
   ```
Of course, if you want to customize your training model and process, you just need to insert GradMF into the loss function when calculating gradients. You can refer to the training code for details.But you need to pay attention to the training order of the two datasets and the entire training process, and try to maintain the original training process as much as possible. This will make your training very smooth and also keep GradMF's original effect.

To save the model, you need to create a model folder in the root directory and save it there.

 ## Citation
 If you are interested in this work, please consider citing:

    @INPROCEEDINGS{10447810,
      author={Han, Kelvin Ting Zuo and Zhang, Shengxuming and Freixas, Gerard Marcos and Feng, Zunlei and Jin, Cheng},
      booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
      title={Target Optimization Direction Guided Transfer Learning for Image Classification}, 
      year={2024},
      volume={},
      number={},
      pages={6445-6449},
      keywords={Training;Deep learning;Image segmentation;Transfer learning;Optimization methods;Artificial neural networks;Signal processing;Transfer learning;Deep learning;GradMF;Gradient projection},
      doi={10.1109/ICASSP48485.2024.10447810}}
      
## Acknowledgment

This work is supported by Ningbo Natural Science Foundation (2022J182).

## Contact
If you have any question or suggestion, please contact zuohanchen22@m.fudan.edu.cn







