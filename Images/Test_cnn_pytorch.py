# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable
import os
import pandas as pd

data_dir = '/data/Robby/AQC/Data/dataset/test'
test_image_files = os.listdir(data_dir)
test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
                                     
model_fullname = "/data/Robby/AQC/Data/Model/_run_1_2020-06-21_First_Model_.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    

model_conv = torch.load(model_fullname)
model_conv.to(device)
model_conv.eval()
print("Model loaded : " + model_fullname)


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model_conv(input)
    index = output.data.cpu().numpy().argmax()
    return index  

def get_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    # np.random.shuffle(indices)
    idx = indices[:num]
    # from torch.utils.data.sampler import SubsetRandomSampler
    # sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels    
    
to_pil = transforms.ToPILImage()
images, labels = get_images(100)
index = []
for ii in range(len(images)):
    image = to_pil(images[ii])
    index.append(predict_image(image))
pd.DataFrame(index,columns =['Predict']).to_csv('hasil.csv')
    
