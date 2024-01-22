import sys

import pandas as pd
import numpy as np
from PIL import Image
import datetime
import random

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
#from sidechannel.audio.sound_module.change_extraction import highlight_significant_changes

# adapted from https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/
batch_size = 64
num_classes = 5 # classes: normal, early, delay, spoofing, masquerading
learning_rate = 0.01 #0.001
num_epochs = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


all_transforms = transforms.Compose([transforms.Resize((256,646)),
                                     transforms.ToTensor()
                                     ])

class ConvNeuralNet(nn.Module):
    # Layers and order in CNN object
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3) # initial out channels: 32 # we only need two channels as it is grayscale anyways
        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3) # inital out 32
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
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

def get_data_point(d_frame, i, data_dir):
    bmp_reference = load_bmp_img(d_frame,i, "reference", data_dir)
    bmp_constructed = load_bmp_img(d_frame, i, "constructed", data_dir)
    label = d_frame.iloc[i]["label"]
    label_class = d_frame.iloc[i]["label_class"]
    return (bmp_reference, bmp_constructed, label, label_class)

def load_data(f_name):
    data_info = pd.read_csv(f_name)
    return data_info

def generate_validation_tensor(bmp_constructed, bmp_recorded):
    val_bmp =  np.concatenate((bmp_constructed, bmp_recorded), dtype="float32") # conversrion necessary for torch to accept input
    tensor = torch.tensor(val_bmp)
    return tensor.view(1, 1, 256, 646)

def get_label(label_str:str):
    if label_str == "normal":
        return [0]
    elif label_str == "spoofing":
        return [1]
    elif label_str == "delayed":
        return [2]
    elif label_str == "early_execution":
        return [3]
    else: # masquerading
        return [4]

def train_network(data, ids):
    l = len(ids)
    model = ConvNeuralNet(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)
    total_step = 1
    clip_value = 5
    for epoch in range(num_epochs):
        print(datetime.datetime.now())
        for i in range(len(ids)):
            #f = random.randint(0, len(ids))
            data_point = get_data_point(data, ids.iloc[i]["id"], "./ml_data")
            validation_tensor = generate_validation_tensor(data_point[0], data_point[1])
            label_class_tensor = torch.tensor(get_label(data_point[3]))
            im = validation_tensor.to(device)
            label = label_class_tensor.to(device)
            #print(label)

            # Forward pass
            outputs = model(im)
            #print("Outputs:", outputs)
            loss = criterion(outputs, label)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            if i % 100 == 0:
                print(f"{i} of {l}")
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        torch.save(model.state_dict(), f"./cnn_model_{epoch+1}_epochs.model")

    return model

def test_model(model, data, ids):
    correct = 0
    total = 0
    l = len(ids)
    for i in range(len(ids)):
        data_point = get_data_point(data, ids.iloc[i]["id"], "./ml_data")
        validation_tensor = generate_validation_tensor(data_point[0], data_point[1])
        label_class_tensor = torch.tensor(get_label(data_point[3]))
        im = validation_tensor.to(device)
        label = label_class_tensor.to(device)
        outputs = model(im)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        if i % 100 == 0:
            print(f"{i} of {l}")
    print('Accuracy of the network on the {} train images: {} %'.format(13440, 100 * correct / total))


def retrain_model(model, data, ids, add_epochs, initial_epochs):
    l = len(ids)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)
    clip_value = 5
    for epoch in range(add_epochs):
        print(datetime.datetime.now())
        for i in range(len(ids)):
            #f = random.randint(0, len(ids))
            data_point = get_data_point(data, ids.iloc[i]["id"], "./ml_data")
            validation_tensor = generate_validation_tensor(data_point[0], data_point[1])
            label_class_tensor = torch.tensor(get_label(data_point[3]))
            im = validation_tensor.to(device)
            label = label_class_tensor.to(device)
            #print(label)

            # Forward pass
            outputs = model(im)
            #print("Outputs:", outputs)
            loss = criterion(outputs, label)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            if i % 100 == 0:
                print(f"{i} of {l}")
        print('Epoch [{}/{}], Loss: {:.4f}'.format(initial_epochs + epoch+1, num_epochs, loss.item()))
        torch.save(model.state_dict(), f"./cnn_model_{initial_epochs + epoch+1}_epochs.model")

def load_model(f_path):
    model = ConvNeuralNet(num_classes)
    model.load_state_dict(torch.load(f_path))
    model.eval()
    return model

def main(args):
    data_file = "./ml_data/labels.csv"
    data = load_data(data_file)
    training_ids = load_data("./ml_data/train_ids.csv")    
    test_ids = load_data("./ml_data/test_ids.csv")
    #model = train_network(data, training_ids[:len(training_ids)])
    #model = ConvNeuralNet(num_classes)
    #model.load_state_dict(torch.load("./cnn_model_1_epoch_v2.model"))
    #model.eval()
    model = load_model("cnn_model_3_epochs.model")
    retrain_model(model, data, training_ids, 2, 3)
    test_model(model, data, test_ids)
    
    pass

if __name__ == "__main__":
    main(sys.argv)