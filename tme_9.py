# -*- coding: utf-8 -*-

import pickle, PIL
import numpy as np
import torch as t
import os, torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from sklearn.svm import LinearSVC
svm = LinearSVC(C=1.0)  

    
torchvision.models.vgg.model_urls["vgg16"] = "http://webia.lip6.fr/~robert/cours/rdfia/vgg16-397923af.pth"

os.environ["TORCH_MODEL_ZOO"] = "/tmp/torch"
vgg16 = torchvision.models.vgg16(pretrained=True)
imagenet_classes = pickle.load(open("imagenet_classes.pkl", "rb"))
# chargement des classes
img = PIL.Image.open("bb.jpg")
img = img.resize((224, 224), PIL.Image.BILINEAR)
mean =  np.array([0.485,0.456,0.406])
std = np.array([0.229,0.224,0.225])
#transform = torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
#img = transform(img)
img = np.array(img, dtype=np.float32)
img = ((img/255.)-mean)/std
img = img.transpose((2, 0, 1))

# TODO preprocess image
img = np.expand_dims(img, 0)
# transformer en batch contenant une image
x = Variable(t.Tensor(img))
y = F.softmax(vgg16(x))

# TODO calcul forward
y = np.argmax(y.data.numpy())
print(imagenet_classes[y])
# transformation en array numpy
# TODO récupérer la classe prédite et son score de confiance

