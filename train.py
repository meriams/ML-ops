import torch
from torchvision.transforms import RandomHorizontalFlip
from torch.utils.data import WeightedRandomSampler 
from sklearn.metrics import classification_report 
from torchvision.transforms import RandomCrop 
from torchvision.transforms import Grayscale 
from torchvision.transforms import ToTensor 
from torch.utils.data import random_split 
from torch.utils.data import DataLoader 
import config as cfg 
from utils  import EarlyStopping
from utils  import LRScheduler 
from torchvision import transforms 
from model import EmotionNet 
from torchvision import datasets 
import matplotlib.pyplot as plt 
from collections import Counter 
from datetime import datetime
from torch.optim import SGD 
import torch.nn as nn 
import pandas as pd 
import argparse 
import math 


# initialize the parser and establish the arguments required 
parser = argparse.ArgumentParser
parser.add_argument('-m', '--model', type=str, help='Path to save the train/')
parser.add_argument('-m', '--plot', type=str, help='Path to save the loss/')
args = vars(parser.parse_args())

# configure the device to use for the training the mode
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[INFO] Current training device: {device}')

# Initialize a list of preprocessing steps to apply on each image during training/validation and testing 
train_transform = transforms.Compose([
    Grayscale(num_output_channels=1),
    RandomHorizontalFlip(),
    RandomCrop((48,48)),
    ToTensor()
])

test_transform = transforms.Compose([
    Grayscale(num_output_channel=1),
    ToTensor()
])

# Load all the images whithin the specified folder and apply different augmentation 
train_data = datasets.ImageFolder(cfg.trainDirectory, transform=train_transform)
test_data = datasets.ImageFolder(cfg.testDirectory, transform=test_transform)

# extract class labels and the total number of classes
classes = train_data.classes
num_of_classes = len(classes)
print(f'[INFO] Class labels: {classes}')

# Using tain samples to generate train/validation set 
num_train_samples = len(train_data)
train_size = math.floor(num_train_samples *cfg.TRAIN_SIZE)
val_size = math.ceil(num_train_samples*cfg.VAL_SIZE)
print(f'[INFO] Train samples: {train_size} ... \t Validaiton samples: {val_size} ...')

# random split the training dataset into train and validation set 
train_data, val_data = random_split(train_data, [train_size, val_size])

# Modify the data transform applied towards the validaiton set 
val_data.dataset.transform = test_transform

#get the labels within the traning set 
train_classes = [label for _, label in train_data]

# count each labels within each classes 
class_count = Counter(train_classes)
print(f'[INFO] Total sample: {class_count}')

# depending on the number of samples available 
class_weight = torch.Tensor([len(train_classes) / c for c in pd.Series(class_count).sort_index.values])

# corresponding class weight already computed 
sample_weight = [0] * len(train_data)
for idx, (image, label) in enumerate(train_data):
    weight = class_weight[label]
    sample_weight[idx] = weight 

# Define a sampler which randomly sample labels fromthe train dataset 
sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(train_data, replacement=True))

# Load our own dataset and store each sample with their corresponding labels 
train_dataloader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, sampler=sampler)
val_dataloader = DataLoader(val_data, batch_size=cfg.BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=cfg.BATCH_SIZE)

# initialize the mdoel and send it to device  
model = EmotionNet(num_of_channels=1, num_of_classes=num_of_classes)

# initialize our optimizer and loss function 
optimizer = SGD(model.parameters(), cfg.LR)
criterion = nn.CrossEntropyLoss()

# Initialize the learning rate scheduler and early stopping mechanism 
lr_schduler = LRScheduler(optimizer)
early_stopping = EarlyStopping()

#Calculate the steps per epoch for training and validation 
train_steps = len(train_dataloader.dataset) // cfg.BATCH_SIZE
val_steps = len(val_dataloader.dataset) // cfg.BATCH_SIZE

# initialize dictionary 
history = {
    'train_acc': [], 
    'train_loss': [], 
    'val_acc': [], 
    'val_loss': []
}


#iterate through the epochs 
print(f'[INFO] Training the model...')
start_time = datetime.now()

for epoch in range(0, cfg.NUM_OF_EPOCHS):
    print(f'[INFO] eopch: {epoch + 1}/{cfg.NUM_OF_EPOCHS}')

    '''
    Training 
    '''

    # set the model to training mode

    model.train()

    # initialize the total training and validation loss and the total 
    # number of correct predictions in both steps 
    total_train_loss = 0 
    total_val_loss = 0 
    total_correct = 0 
    val_correct = 0 

    for (data, target) in train_dataloader:
        data, target = data.to(device), target.to(device)

        predictions = model(data)
        loss = criterion(predictions, target)

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        total_train_loss += loss
        train_correct += (predictions.argmax(1) == target).type(torch.float).sum().item()

    '''
    Validation the model 
    '''

    model.eval()

    with torch.set_grad_enabled(False):

        for (data, target) in val_dataloader:
            data, target = data.to(device), target.to(device)

            predictions = model(data)
            loss = criterion(predictions, target)

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            total_val_loss += loss
            val_correct += (predictions.argmax(1) == target).type(torch.float).sum().item()


    #   calculate the average training and validation loss 
    avg_train_loss = total_train_loss/train_steps
    avg_val_loss = total_val_loss/val_steps

    # calculate the train and validation accuracy 
    train_correct = train_correct /len(train_dataloader.dataset)
    val_correct = val_correct /len(val_dataloader.dataset)

    #print model  training and validation 
    print(f'train loss: {avg_train_loss:.3f} .. train accuracy: {train_correct:.3f}')
    print(f'val loss: {avg_val_loss:.3f} .. val accuracy: {val_correct:.3f}', end= '\n\n')

    history['train_loss'].append(avg_train_loss.cpu().detach().numpy())
    history['train_acc'].append(train_correct)
    history['val_loss'].append(avg_val_loss.cpu().detach().numpy())
    history['val_acc'].append(val_correct)


    # execute the learning rate scheduler and early stopping 
    validation_loss = avg_val_loss.cpu().detach().numpy()
    lr_schduler(validation_loss)
    early_stopping(validation_loss)

    if early_stopping.early_stop_enabled:
        break

print(f'[INFO] Total training time: {datetime.now() - start_time}...')

# move model back to cpu and save the trained mdoel to disk 
if device=='cuda':
    model = model.to('cpu')
torch.save(model.state_dict(), args['model'])

# Plot the training loss and the accuracy overtime 
plt.style.use('ggplot')
plt.figure()
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.ylabel('Loss/Accuracy')
plt.xlabel('#No of Epochs')
plt.title('Training Loss and Accuracy on FER2013')
plt.legend(loc='upper right')
plt.savefig(args['plot'])

with torch.set_grad_enabled(False):
    model.eval()

    predictions = []

    for (data, _) in test_dataloader:
        data = data.to(device)
        output = model(data)
        output = output.argmax(axis=1).cpu().numpy()
        predictions.extend(output)

#Evaluate the model 
print('[INFO] evaluating model...' )
actual = [label for _, label in test_data]
print(classification_report(actual, predictions, target_names=test_data.classes))

