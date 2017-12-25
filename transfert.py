import argparse
import os
import time
import torch.optim as optim
from torch.optim import lr_scheduler

from PIL import Image
import torch.nn.functional as F
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
    # Extraction des features
    print('Feature extraction')
    
    X_train, y_train = extract_features(train, model)
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
                return x
                #return np.argmax(F.softmax(x).cpu().data.numpy())
            
    print('Instanciation de VGG16relu7')
    model = VGG16lastNew()
    model.eval()

    if CUDA: # si on fait du GPU, passage en CUDA
        model = model.cuda()

    # On récupère les données
    print('Récupération des données')
    train, test = get_dataset(params.batch_size, params.path)
    
    data = {"train":train, "test":test}
    dataset_sizes = {"train":len(train)*params.batch_size,"test":len(test)*params.batch_size}
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    train_model(model, data, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=40)
    
def train_model(model, d,dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    accus_=[]
    losss_=[]
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in d[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if CUDA:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                #print(type(outputs.data))
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            accus_.append(epoch_acc)
            losss_.append(epoch_loss)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


    """Plotting the evolution accuracy """
    fig = plt.figure(figsize=(11,7))

    fig.suptitle(" Accuracy & Loss",fontsize=15)
    plt.xlabel("Epoch ", fontsize=12)
    plt.ylabel("Accuracy ", fontsize=12)
    interval = range(len(losss_))
    plt.plot(interval, accus_,color = "blue", label = "Accuracy")    
    plt.plot(interval, losss_,color = "red", label = "Loss")    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
    plt.savefig('Accuracy_&_Loss.png')
    #plt.show()
    ## load best model weights
    #model.load_state_dict(best_model_wts)
    return None,None
    

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

    if(not CUDA):
    
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
