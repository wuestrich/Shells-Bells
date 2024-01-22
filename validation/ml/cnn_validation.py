import sys
import os
import inspect

import pandas as pd
import numpy as np
from PIL import Image
import datetime
import random

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
pparentdir = os.path.dirname(parentdir) # two parents up
sys.path.insert(0, pparentdir) 
sys.path.insert(0, currentdir)
from data_handling import ValidationData, ToTensor
from sidechannel.audio.sound_module.change_extraction import highlight_significant_changes

# adapted from https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/
batch_size = 64
num_classes = 5 # classes: normal, early, delay, spoofing, masquerading
learning_rate = 0.001
num_epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


all_transforms = transforms.Compose([transforms.Resize((256,646)),
                                     transforms.ToTensor()
                                     ])


class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        #self.fc1 = nn.Linear(1600, 128)
        self.fc1 = nn.Linear(616832, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

def load_bmp(d_frame, row:int, col:str, d_dir:str="."):
    loaded_bmp = np.array(Image.open(f"{d_dir}/{d_frame.iloc[row][col]}").convert("L"))
    bmp = []
    for x in range(len(loaded_bmp)):
        row = []
        for y in range(len(loaded_bmp[x])):
            if loaded_bmp[x][y] == 30:
                row.append(0)
            else:
                row.append(1)
        bmp.append(row)
    return np.array(bmp)

def load_bmp_img(d_frame, row:int, col:str, d_dir:str="."):
    return np.array(Image.open(f"{d_dir}/{d_frame.iloc[row][col]}").convert("L"))

def test_model(model, data):
    len_data = 14720
    print(f"Testing model with {len_data} entries")
    with torch.no_grad():
        correct = 0
        total = 0
        stats = {0: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}, 1: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}, 2: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}, 3: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}, 4: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}}
        for samples in data:
            bitmaps = samples["bitmap"].to(device)
            labels = samples["label_class"]
            labels = torch.squeeze(labels).to(device)
            outputs = model(bitmaps)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted = predicted.tolist()
            labels = labels.tolist()
            for i in range(len(labels)):
                stats[labels[i]][predicted[i]] += 1
                stats[labels[i]][5] += 1 #total occurences
        print('Accuracy of the network on the {} test images: {} %'.format(len_data, 100 * correct / total))
    return stats

def pretty_print_stats(stats: dict):
    """Prints out statistics of a test """
    labels = {0: "normal", 1: "spoofing", 2: "delay", 3: "early_execution", 4: "masquerading", 5: "total"}
    for k in stats:
        s = f"Class: {labels[k]:<15} Occurrences: {stats[k][5]:<7} classified as: "
        for l in stats[k]:
            if l == 5:
                continue
            s += f" {labels[l]}:{stats[k][l]:<7}\t"
        s += f"({stats[k][k]/stats[k][5]} correct)"
        print(s)


def load_model(f_path):
    model = ConvNeuralNet(num_classes)
    model.load_state_dict(torch.load(f_path))
    model.eval()
    return model

def train_network(data):
    model = ConvNeuralNet(num_classes)

    # Set Loss function with criterion
    criterion = nn.CrossEntropyLoss()

    # Set optimizer with optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

    total_step = len(data)
    for epoch in range(num_epochs):
        print(datetime.datetime.now())
	    #Load in the data in batches using the train_loader object
        for i, sample in enumerate(data):  
            # Move tensors to the configured device
            bitmaps = sample["bitmap"].to(device)
            labels = sample["label_class"]
            labels = torch.squeeze(labels).to(device)
            #print(labels.shape)
            # Forward pass
            outputs = model(bitmaps)
            loss = criterion(outputs, labels)
        
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        with open("./ndss_models/loss_info", "a") as lossfile:
            lossfile.write(f'{datetime.datetime.now()}: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}\n')
        torch.save(model.state_dict(), f"./ndss_models/ndss_cnn_model_{epoch+1}_epochs.model")
    return model


def cnn_validation(f_name_recording, f_name_reference):
    labels = {0: "normal", 1: "spoofing", 2: "delay", 3: "early_execution", 4: "masquerading", 5: "total"}
    model = load_model("./validation/ml/cnn_model_17_epochs.model")
    # file reading and change extraction
    print("Highlighting changes in recording")
    changes_recording = highlight_significant_changes(f_name_recording)
    recording_bmp_fname = f"recording_bmp_{1}.bmp" # TODO random identifier
    print("highlighting changes in reference")
    changes_reference = highlight_significant_changes(f_name_reference)
    if changes_recording.shape[1] != changes_reference.shape[1]:
        print(f"Recording {f_name_recording} shape:", changes_recording)
        print(f"Reference {f_name_reference} shape:", changes_reference)
        # need to extend recording bitmap by a timestep (128,645) to (128,646)
        changes_recording = np.insert(changes_recording, changes_recording.shape[1]-1, 0, axis=1)
        print(changes_recording.shape)
    reference_bmp_fname = f"reference_bmp_{1}.bmp" # TODO random identifier
    #matplotlib.image.imsave(f"{reference_bmp_fname}", changes_reference)
    #matplotlib.image.imsave(f"{recording_bmp_fname}", changes_recording)
    # combination direct (currently not working)
    validation_bmp = np.concatenate((changes_reference, changes_recording), axis=0)
    validation_bmp = np.array(validation_bmp, dtype="float32")
    validation_bmp = np.expand_dims(validation_bmp, axis=-1) # add channel dimension for cnn input
    validation_bmp = validation_bmp.transpose((2,0,1))
    validation_tensor = torch.from_numpy(validation_bmp)
    
    # combination via files
    #reference_bmp = Image.open(reference_bmp_fname).convert("L")
    #constructed_bmp = Image.open(recording_bmp_fname).convert("L")
    #validation_bmp = Image.new('L', (reference_bmp.width, reference_bmp.height + constructed_bmp.height))
    #validation_bmp.paste(reference_bmp, (0, 0))
    #validation_bmp.paste(constructed_bmp, (0, reference_bmp.height))
    #validation_bmp = np.array(validation_bmp, dtype="float32")
    #validation_bmp[validation_bmp==30] = 0
    #validation_bmp[validation_bmp!=0] = 1
    #validation_bmp = np.expand_dims(validation_bmp, axis=-1) # add dimension for grayscale image (channels=1) https://stackoverflow.com/questions/49237117/python-add-one-more-channel-to-image
    #validation_tensor = torch.from_numpy(validation_bmp)
    
    
    # LABELLING
    t = torch.unsqueeze(validation_tensor,0).to(device)
    outputs = model(t)
    _, predicted = torch.max(outputs.data, 1)
    print(labels[predicted.tolist()[0]])
    plt.imshow(validation_tensor.numpy().transpose(1,2,0))
    plt.show()
    # cleanup of artifacts
    #os.remove(recording_bmp_fname)
    #os.remove(reference_bmp_fname)
    return labels[predicted.tolist()[0]]

def main(args):
    pass

if __name__ == "__main__":
    main(sys.argv)