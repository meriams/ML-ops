# ML-ops

## Overall goal of the project
The overall goal of the project is to classify facial expressions into different emotions such as happiness, sadness, anger, surprise, fear, disgust, or neutral using deep learning techniques. This involves training a model to recognize patterns in facial images and associate them with specific emotions.
x
## What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics)

In this project, we intend to use PyTorch, a popular deep learning framework, for building and training the models. Specifically, we'll likely utilize standard PyTorch functionalities and potentially leverage pre-trained models and components from the PyTorch ecosystem to enhance model performance.

## How to you intend to include the framework into your project
We will integrate PyTorch into the project by setting up a PyTorch project structure, utilizing PyTorch for model construction, defining data loaders, loss functions, optimizers, and training loops. Additionally, we may employ specific PyTorch components like PyTorch Image Models or Transformers if they align with the project requirements.

## What data are you going to run on (initially, may change)
The initial dataset for training and testing is the FER2013 dataset, which contains facial expression images labeled with corresponding emotions. The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image. There is seven categories of emotions (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

## What deep learning models do you expect to use
We expect to use a deep learning models for this project to find the best one, including:

- Convolutional Neural Networks (CNNs): A common choice for image classification tasks.
- Transfer Learning: Utilizing pre-trained CNNs (e.g., ResNet, VGG, etc.) and fine-tuning them for our specific emotion classification task.

