import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import json

import torch
from torch import nn
from torch import optim
from torchvision import transforms , datasets , models 
from torch.autograd import Variable

from PIL import Image
import sys
##----------------------------------------------------------------------------------------------------------------

def arguments():
   parser = argparse.ArgumentParser()
   parser.add_argument('image_path' , type = str ,help = "path to the image file")
   parser.add_argument('checkpoint' , type = str ,help = "load saved model")
   parser.add_argument('--top_k' , type = int ,default = 5 ,help = "top five probability ans indexes")
   parser.add_argument('--category_names' , type = str ,default="",help = "map classes to file names")
   parser.add_argument('--gpu' , action = 'store_true' ,help = "use the gpu")
   
   return parser.parse_args()
   
    
##------------------------------------------------------------------------------------------------------------------

def load_model(checkpoint):
   checkpoint = torch.load(checkpoint)
   arch = checkpoint['arch']
   if arch=='alexnet':
    model=models.alexnet(pretrained = True)
   elif arch=='vgg':
    model=models.vgg16(pretrained = True)
   else:
    print("Enter the correct architecture name")
    sys.exit()
   model.load_state_dict = checkpoint['state']
   model.classifier = checkpoint['classifier']
   class_to_idx = checkpoint['class_to_idx']
   optimizer = checkpoint["optimizer_dict"]
    
   return model ,class_to_idx
##---------------------------------------------------------------------------------------------------------------------
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image)
    
    image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (np.transpose(image,(1,2,0)) - mean)/std    
    image = np.transpose(image, (2, 0, 1))
            
    return image
    
 ##---------------------------------------------------------------------------------------------------------------------
def predict(model ,image ,topk ,category_names,class_to_idx ,gpu):
 image = Variable(torch.FloatTensor(image), requires_grad=True)
 image = image.unsqueeze(0) # this is for VGG
 if gpu and torch.cuda.is_available():
   image , model = image.to('cuda') , model.to('cuda')
 else:
   image , model = image.to('cpu') , model.to('cpu')
 result = model(image).topk(topk)  
 print(result)
 if gpu and torch.cuda.is_available():
  probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
  classes = result[1].data.cpu().numpy()[0]
 else:
  probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
  classes = result[1].data.numpy()[0]
    

 class_to_idx = class_to_idx
 inv = {v: k for k, v in class_to_idx.items()}

 por=[]
 for i in classes:
  por.append(inv[i])
 
 if category_names:
  with open(category_names, 'r') as f:
     cat_to_name = json.load(f)
  var =[]
  for item in por:
   var.append(cat_to_name[item])   
  classes =var
  probs = list(probs)

 else:
  classes =por
  probs = list(probs)

 return classes ,probs

##--------------------------------------------------------------------------------------------------------------------------------
def print_predict(classes , probs):
    print(" ")
    var = {"Flowers":classes , "Probabilities":probs }
    df =pd.DataFrame(var )
    print(df)

##---------------------------------------------------------------------------------------------------------------------------------
def main():
 args = arguments()
 model ,class_to_idx= load_model(args.checkpoint)
 print(model)
 image = process_image(args.image_path)
 classes ,probs =predict(model ,image ,args.top_k ,args.category_names, class_to_idx , args.gpu)
 print_predict(classes ,probs)
 
##-----------------------------------------------------------------------------------------------------------------------

if __name__== '__main__':
 main()