import argparse
import os
import time

from PIL import Image

import numpy as np
from matplotlib import pyplot as plt
import torch, torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.svm import LinearSVC

torchvision.models.vgg.model_urls["vgg16"] = "http://webia.lip6.fr/~robert/cours/rdfia/vgg16-397923af.pth"
os.environ["TORCH_MODEL_ZOO"] = "/tmp/torch"
PRINT_INTERVAL = 50
CUDA = False


def get_dataset(batch_size, path):
    """
    Cette fonction charge le dataset et effectue des transformations sur chaqu
    image (listées dans `transform=...`).
    """
    mean =  np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    train_dataset = datasets.ImageFolder(path+'/train',
        transform=transforms.Compose([ # TODO Pré-traitement à faire
            transforms.Scale((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ]))
    val_dataset = datasets.ImageFolder(path+'/test',
        transform=transforms.Compose([ # TODO Pré-traitement à faire
            transforms.Scale((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2,drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2,drop_last=True)

    return train_loader, val_loader


def extract_features(data, model):

    X = []
    y = []
    print("Before")
    for i, (input, target) in enumerate(data):
        if i % PRINT_INTERVAL == 0:
            print('Batch {0:03d}/{1:03d}'.format(i, len(data)))
        if CUDA:
            input = input.cuda()

        #Feature extraction
        x_inter = model(Variable(input,volatile=True))
        x_ = x_inter.cpu().data.numpy()

        # Feature L2-normalizing
        for elt in range(len(x_)):
            x_[elt] = x_[elt] / np.linalg.norm(x_[elt],2)
            #print("feature ",x_[elt].shape)        
            X.append(x_[elt].reshape(4096))
            y.append(target[elt])
        
    return np.array(X), y


def main_experiment_svm(params):
    print('Instanciation de VGG16')
    vgg16 = models.vgg16(pretrained=True)    
    
    class VGG16relu7(nn.Module):
 
	    def __init__(self):
        		super(VGG16relu7, self).__init__()
        		
        		# recopier toute la partie convolutionnelle
        		self.features = nn.Sequential( *list(vgg16.features.children()))
        		
        		# garder une partie du classifieur, -2 pour s’arrêter à relu7
        		self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])
                        
	    def forward(self, x):
        		x = self.features(x)
        		x = x.view(x.size(0), -1)
        		x = self.classifier(x)
        		return x
            
    print('Instanciation de VGG16relu7')
    model = VGG16relu7()
    model.eval()
    if CUDA: # si on fait du GPU, passage en CUDA
        model = model.cuda()

    # On récupère les données
    print('Récupération des données')
    train, test = get_dataset(params.batch_size, params.path)
    #print('size of train dataset :', train.shape)
    # Extraction des features
    print('Feature extraction')
    
    X_train, y_train = extract_features(train, model)
    #print("Shapes :",X_train.shape,y_train.shape)
    X_test, y_test = extract_features(test, model)
    # TODO Apprentissage et évaluation des SVM à faire

    print('Apprentissage des SVM')
    C_VAL = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.51,0.52,0.53,1.,1e+1,1e+2,1e+3,1e+4,1e+5,1e+6]
    accuracies=[]
    for c_value in C_VAL:        
        print('Instanciation de LinSVM avec C = ',str(c_value))    
        svm = LinearSVC(C=c_value)

        svm.fit(X_train,y_train)
    
        accuracy = svm.score(X_test, y_test)
        accuracies.append(accuracy)
        print('Accuracy : ',accuracy)

    return C_VAL,accuracies

def main_learn_last_vgg16(params):

    print('Instanciation de VGG16')
    vgg16 = models.vgg16(pretrained=True)

    for param in vgg16.parameters():
        param.requires_grad = False

    class VGG16lastNew(nn.Module):
 
	    def __init__(self):
                super(VGG16lastNew, self).__init__()
                num_classes=15
                # recopier toute la partie convolutionnelle
                self.features = nn.Sequential( *list(vgg16.features.children()))
        		
                # garder une partie du classifieur, -2 pour s’arrêter à relu7
                l = list(vgg16.classifier.children())[:-1]
                l.append(nn.Linear(4096, num_classes))
                self.classifier = nn.Sequential(*l)

                        
	    def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return np.argmax(F.softmax(x).data.numpy())
            
    print('Instanciation de VGG16relu7')
    model = VGG16lastNew()
    model.eval()

    if CUDA: # si on fait du GPU, passage en CUDA
        model = model.cuda()



if __name__ == '__main__':

    # Paramètres en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/tmp/datasets/mnist', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='activate GPU acceleration')

    args = parser.parse_args()
    if args.cuda:
        CUDA = True
        cudnn.benchmark = True

        c_vals, acc= main_learn_last_vgg16(args) #main_experiment_svm(args)

    """Plotting the evolution accuracy """
    fig = plt.figure(figsize=(11,7))
    id_max = np.argmax(acc)
    max_val =str(c_vals[id_max])
    print(" Accuracy depending on C value, best(C, Acc) = ("+max_val+', '+str(acc[id_max])+ ') ')

    fig.suptitle(" Accuracy depending on C value",fontsize=15)
    plt.xlabel("Log Value of C  ", fontsize=12)
    plt.ylabel("Accuracy of LinSVM", fontsize=12)
    interval = [np.log10(it) for it in c_vals]
    plt.plot(interval, acc,color = "blue") #, label = "map test random")    
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('Accuracy_C_Value.png')
    plt.show()

    #Accuracy depending on C value, best(C, Acc) = (0.52, 0.891085790885)
