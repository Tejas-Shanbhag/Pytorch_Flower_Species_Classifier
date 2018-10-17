import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
from torch import nn
from torch import optim
from torchvision import transforms , datasets , models 
from torch.autograd import Variable

from PIL import Image
import sys


#----------------------------------------------------------------------------------------------------------------------------------
def get_args():
 parser = argparse.ArgumentParser()
 parser.add_argument('data_dir' , type=str ,  help = "main directory containing data")
 parser.add_argument('--save_dir' , type=str ,default ='checkpoint.pth' , help = "save directory")
 parser.add_argument('--arch' , type = str , default= 'alexnet', help = "The Model architecture(vgg or alexnet)")
 parser.add_argument('--learning_rate' , type=float, default=0.001 ,help = "The learning rate of the model")
 parser.add_argument('--hidden_units' , type=int ,default=4096 , help = "Number of hidden units")
 parser.add_argument('--epochs' , type= int , default = 1, help = "Number of epochs")
 parser.add_argument('--gpu' , action ='store_true', help = "main directory containing data")
 return parser.parse_args()

#--------------------------------------------------------------------------------------------------------------------------------

def load(path):
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30) ,
                                           transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406] ,[0.229,0.224 ,0.225]) ])

    valid_transforms =transforms.Compose([ transforms.Resize(256) , transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406] ,[0.229,0.224 ,0.225])])

    test_transforms =transforms.Compose([ transforms.Resize(256) , transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406] ,[0.229,0.224 ,0.225])])


    # TODO: Load the datasets with ImageFolder
    trainset =datasets.ImageFolder(train_dir,transform=train_transforms)
    validset =datasets.ImageFolder(valid_dir,transform=valid_transforms)
    testset =datasets.ImageFolder(test_dir,transform=test_transforms)
    var = trainset.class_to_idx


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    
    return trainloader ,validloader ,testloader ,var
##---------------------------------------------------------------------------------------------------------------------------

def model_arch(arch ,hidden_units):
    if arch=='alexnet':
      model = models.alexnet(pretrained = True)
    
    elif arch =='vgg':
      model = models.vgg16(pretrained = True) 
    
    else:
      print("Enter a valid Architecture ")
      sys.exit()
    
    num_labels=102
    for param in model.parameters():
        param.requires_grad = False
    
    # Features, removing the last layer
    features = list(model.classifier.children())[:-1]
  
    # Number of filters in the bottleneck layer
    num_filters = model.classifier[len(features)].in_features

    # Extend the existing architecture with new layers
    features.extend([
        nn.Dropout(),
        nn.Linear(num_filters, hidden_units),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(True),
        nn.Linear(hidden_units, num_labels),
        #nn.Softmax(dim=1) 
        # Please, notice Softmax layer has not been added as per Pytorch answer:
        # https://github.com/pytorch/vision/issues/432#issuecomment-368330817
        # It is not either included in its transfer learning tutorial:
        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        ])
    model.classifier = nn.Sequential(*features)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.001)
    return model , optimizer ,criterion

##--------------------------------------------------------------------------------------------------------------------------------
def valid(model ,validloader ,criterion):
 test_loss = 0
 accuracy = 0
 for images, labels in validloader:
        if torch.cuda.is_available():
         images ,labels ,model =images.to("cuda") , labels.to('cuda') , model.to('cuda')
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
    
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
 return test_loss, accuracy

##-----------------------------------------------------------------------------------------------------------------------
def model_train(model ,epochs ,trainloader ,validloader ,learning_rate ,gpu ,optimizer ,criterion):   
 print_every = 40
 steps = 0

 if gpu and torch.cuda.is_available:
  model.to('cuda')
 for e in range(epochs):
    model.train()
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        if gpu and torch.cuda.is_available:
         inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if steps % print_every == 0:
          model.eval()
          with torch.no_grad():
            valid_loss, accuracy = valid(model, validloader, criterion)
        
          print("Epoch: {}/{}.. ".format(e+1, epochs),
                 "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))  
    
          running_loss = 0
          model.train()
 return model
##---------------------------------------------------------------------------------------------------------------------------
 
def test_mode(model ,testloader ,gpu ):
    correct = 0
    total = 0
    with torch.no_grad():
     for images ,labels in testloader:
        if gpu and torch.cuda.is_available:
         images,labels,model=images.to('cuda'),labels.to('cuda'),model.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

     print('Accuracy of the network is : %d %%' % (100 * correct / total))
##------------------------------------------------------------------------------------------------------------------------
    
def save_model(model ,save_dir , epochs , optimizer , class_idx , arch ):
    checkpoint ={'state' :model.state_dict ,'epoch':epochs ,
                "optimizer_dict" :optimizer.state_dict , "class_to_idx" : class_idx ,"arch" :arch , "classifier" : model.classifier }
    
    torch.save(checkpoint,save_dir)
    print("model saved at" , save_dir)

                 
#----------------------------------------------------------------------------------------------------------------------------------
def main():
 args = get_args()
 trainloader ,validloader ,testloader ,class_to_idx = load(args.data_dir)
 model ,optimizer ,criterion = model_arch(args.arch, args.hidden_units)
 print(model)
 model = model_train(model ,args.epochs ,trainloader ,validloader ,args.learning_rate ,args.gpu,optimizer ,criterion) 
 test_mode(model , testloader ,args.gpu )
 save_model(model ,args.save_dir ,args.epochs ,optimizer ,class_to_idx ,args.arch )
 
    
#----------------------------------------------------------------------------------------------------------------------------------

    
if __name__=='__main__':
    main()