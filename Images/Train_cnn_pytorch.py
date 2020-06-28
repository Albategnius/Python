from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import datetime
import os
import copy
from tensorboardX import SummaryWriter

plt.ion() 

#Data Augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.RandomRotation(90),
        #transforms.ColorJitter(contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#train model function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    
    writer = SummaryWriter()
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase=='train':
                writer.add_scalar('train_epoch_loss',epoch_loss,epoch)
            elif phase=='val':
                writer.add_scalar('val_epoch_loss',epoch_loss,epoch)

            if phase=='train':
                writer.add_scalar('train_epoch_acc',epoch_acc,epoch)
            elif phase=='val':
                writer.add_scalar('val_epoch_acc',epoch_acc,epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


#1. specify path 
#change here
data_dir = '/data/Robby/AQC/Data/dataset/'

#2. load train images
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=15,
                                             shuffle=True, num_workers=15)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

#3. define gpu device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#4. Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

#5. define the architecture and parameters to use
##train only the last fully connected layer, freeze all the parameter except the last one 
##Set requires_grad == False to freeze the parameters so that the gradients are not computed in backward().
model_conv = torchvision.models.resnet18(pretrained=True)

#overwrite using EPEL 2M RI model 
#model_fullname = "/pfiadmin_data/u_deasy/Data/Prod_DL_Model/EPEL_2m_RI/Epel_2M_model_transfer2.pt"
#model_conv = torch.load(model_fullname)

for param in model_conv.parameters():
    param.requires_grad = False

#Parameters of newly constructed modules have requires_grad=True by default

#nn.linear(num_features,classes)
#change here
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 41)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

#6. train the model
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=40)

#visualize_model(model_conv)

#7. save model
#change here
model_path = "/data/Robby/AQC/Data/Model/"
todayDate = datetime.datetime.today().strftime('%Y-%m-%d')
runNo = 1
today_run = []
if len(os.listdir(model_path) ) == 0:
	runNo = runNo
else:
	for name in os.listdir(model_path):
		name_lst = name.split(".")
		name_char = name_lst[0].split("_")
		if name_char[2] == todayDate:
			today_run.append(int(name_char[1]))
	if len(today_run) != 0:
		runNo = max(today_run)+1
	else :
		runNo = 1

modelname = 'run_' + str(runNo) + '_' + todayDate	
print("model :" + modelname)
#model_fullname = model_path+modelname+".pth"
#torch.save(model_conv.state_dict(),model_fullname)

model_fullname = model_path+"_"+modelname+"_Second_Model_"+".pt"
torch.save(model_conv,model_fullname)

