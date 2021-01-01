import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np

def ConvBNReLU(in_channels,out_channels,kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,padding=kernel_size//2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class InceptionV1Module(nn.Module):
    def __init__(self, in_channels,out_channels1, out_channels2reduce,out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV1Module, self).__init__()

        self.branch1_conv = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels2reduce,kernel_size=1)
        self.branch2_conv2 = ConvBNReLU(in_channels=out_channels2reduce,out_channels=out_channels2,kernel_size=3)

        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=5)

        self.branch4_pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.branch4_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1)

    def forward(self,x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv2(self.branch2_conv1(x))
        out3 = self.branch3_conv2(self.branch3_conv1(x))
        out4 = self.branch4_conv1(self.branch4_pool(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionAux(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(InceptionAux, self).__init__()

        self.auxiliary_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.auxiliary_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.auxiliary_linear1 = nn.Linear(in_features=128 * 4 * 4, out_features=1024)
        self.auxiliary_relu = nn.ReLU6(inplace=True)
        self.auxiliary_dropout = nn.Dropout(p=0.7)
        self.auxiliary_linear2 = nn.Linear(in_features=1024, out_features=out_channels)

    def forward(self, x):
        x = self.auxiliary_conv1(self.auxiliary_avgpool(x))
        x = x.view(x.size(0), -1)
        x= self.auxiliary_relu(self.auxiliary_linear1(x))
        out = self.auxiliary_linear2(self.auxiliary_dropout(x))
        return out

class InceptionV1(nn.Module):
    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV1, self).__init__()
        self.stage = stage

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block3 = nn.Sequential(
            InceptionV1Module(in_channels=192,out_channels1=64, out_channels2reduce=96, out_channels2=128, out_channels3reduce = 16, out_channels3=32, out_channels4=32),
            InceptionV1Module(in_channels=256, out_channels1=128, out_channels2reduce=128, out_channels2=192,out_channels3reduce=32, out_channels3=96, out_channels4=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block4_1 = InceptionV1Module(in_channels=480, out_channels1=192, out_channels2reduce=96, out_channels2=208,out_channels3reduce=16, out_channels3=48, out_channels4=64)

        if self.stage == 'train':
            self.aux_logits1 = InceptionAux(in_channels=512,out_channels=num_classes)

        self.block4_2 = nn.Sequential(
            InceptionV1Module(in_channels=512, out_channels1=160, out_channels2reduce=112, out_channels2=224,
                              out_channels3reduce=24, out_channels3=64, out_channels4=64),
            InceptionV1Module(in_channels=512, out_channels1=128, out_channels2reduce=128, out_channels2=256,
                              out_channels3reduce=24, out_channels3=64, out_channels4=64),
            InceptionV1Module(in_channels=512, out_channels1=112, out_channels2reduce=144, out_channels2=288,
                              out_channels3reduce=32, out_channels3=64, out_channels4=64),
        )

        if self.stage == 'train':
            self.aux_logits2 = InceptionAux(in_channels=528,out_channels=num_classes)

        self.block4_3 = nn.Sequential(
            InceptionV1Module(in_channels=528, out_channels1=256, out_channels2reduce=160, out_channels2=320,
                              out_channels3reduce=32, out_channels3=128, out_channels4=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block5 = nn.Sequential(
            InceptionV1Module(in_channels=832, out_channels1=256, out_channels2reduce=160, out_channels2=320,out_channels3reduce=32, out_channels3=128, out_channels4=128),
            InceptionV1Module(in_channels=832, out_channels1=384, out_channels2reduce=192, out_channels2=384,out_channels3reduce=48, out_channels3=128, out_channels4=128),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(in_features=1024,out_features=num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        aux1 = x = self.block4_1(x)
        aux2 = x = self.block4_2(x)
        x = self.block4_3(x)
        out = self.block5(x)
        out = self.avgpool(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.stage == 'train':
            aux1 = self.aux_logits1(aux1)
            aux2 = self.aux_logits2(aux2)
            return aux1, aux2, out
        else:
            return out
            
class FurnitureOCR(object):
    """docstring for FurnitureOCR"""
    def __init__(self, in_path, epoch, batch_size, lr):
        super(FurnitureOCR, self).__init__()
        self.in_path = in_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr

        self.classes = ['Bed', 'Chair', 'Wardrobes']

        self.checkdevice()
        self.prepareData()
        self.getModel()
        self.train_acc = self.train()
        self.saveModel()
        self.test()

        #self.showWeights()

    def checkdevice(self):
        # To determine if your system supports CUDA
        print("Check devices...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Current device:", self.device)

        # Also can print your current GPU id, and the number of GPUs you can use.
        print("Our selected device:", torch.cuda.current_device())
        print(torch.cuda.device_count(), "GPUs is available")
        return

    def prepareData(self):
        # The output of torchvision datasets are PILImage images of range [0, 1]
        # We transform them to Tensor type
        # And normalize the data
        # Be sure you do same normalization for your train and test data
        print('Preparing dataset...')

        # The transform function for train data
        transform_train = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
            # you can apply more augment function
            # [document]: https://pytorch.org/docs/stable/torchvision/transforms.html
        ])

        # The transform function for test data
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ])


        # TODO
        self.trainset = torchvision.datasets.ImageFolder(root='./Furniture',transform=transform_train)
        self.testset = torchvision.datasets.ImageFolder(root='./Furniture',transform=transform_test)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
        # you can also split validation set
        # self.validloader = torch.utils.data.DataLoader(self.validset, batch_size=self.batch_size, shuffle=True)
        return

    def getModel(self):
        # Build a Convolution Neural Network
        self.net = InceptionV1()
        print(self.net)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        return
        
    def train(self):
        print('Training model...')
        # Change all model tensor into cuda type
        # something like weight & bias are the tensor 
        self.net = self.net.to(self.device) 
        
        # Set the model in training mode
        # because some function like: dropout, batchnorm...etc, will have 
        # different behaviors in training/evaluation mode
        # [document]: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train
        self.net.train()
        for e in range(self.epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            correct = 0
            for i, (inputs, labels) in enumerate(self.trainloader, 0):
                
                #change the type into cuda tensor 
                inputs = inputs.to(self.device) 
                labels = labels.to(self.device) 

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                # select the class with highest probability
                _, pred = outputs.max(1)
                # if the model predicts the same results as the true
                # label, then the correct counter will plus 1
                correct += pred.eq(labels).sum().item()
                
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (e + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

            print('%d epoch, training accuracy: %.4f' % (e+1, 100.*correct/len(self.trainset)))
        print('Finished Training')
        return 100.*correct/len(self.trainset)

    def test(self):
        print('==> Testing model..')
        # Change model to cuda tensor
        # or it will raise when images and labels are all cuda tensor type
        self.net = self.net.to(self.device)

        # Set the model in evaluation mode
        # [document]: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.eval 
        self.net.eval()

        correct = 0
        running_loss = 0.0
        iter_count = 0
        class_correct = [0 for i in range(len(self.classes))]
        class_total = [0 for i in range(len(self.classes))]
        with torch.no_grad(): # no need to keep the gradient for backpropagation
            for data in self.testloader:
                images, labels = data
                images = images.to(self.device) 
                labels = labels.to(self.device)
                outputs = self.net(images)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                c_eachlabel = pred.eq(labels).squeeze()
                loss = self.criterion(outputs, labels)
                iter_count += 1
                running_loss += loss.item()
                for i in range(len(labels)):
                    cur_label = labels[i].item()
                    try:
                        class_correct[cur_label] += c_eachlabel[i].item()
                    except:
                        print(class_correct[cur_label])
                        print(c_eachlabel[i].item())
                    class_total[cur_label] += 1

        print('Total accuracy is: {:4f}% and loss is: {:3.3f}'.format(100 * correct/len(self.testset), running_loss/iter_count))
        print('For each class in dataset:')
        for i in range(len(self.classes)):
            print('Accruacy for {:18s}: {:4.2f}%'.format(self.classes[i], 100 * class_correct[i]/class_total[i]))

    def saveModel(self):
        # After training , save the model first
        # You can saves only the model parameters or entire model
        # Some difference between the two is that entire model 
        # not only include parameters but also record hwo each 
        # layer is connected(forward method).
        # [document]: https://pytorch.org/docs/master/notes/serialization.html
        print('Saving model...')

        # only save model parameters
        torch.save(self.net.state_dict(), './weight.t7')

        # you also can store some log information
        state = {
            'net': self.net.state_dict(),
            'acc': self.train_acc,
            'epoch': self.epoch
        }
        torch.save(state, './weight.t7')

        # save entire model
        torch.save(self.net, './model.pt')
        return

    def loadModel(self, path):
        print('Loading model...')
        if path.split('.')[-1] == 't7':
            # If you just save the model parameters, you
            # need to redefine the model architecture, and
            # load the parameters into your model
            self.net = googLeNet()
            checkpoint = torch.load(path)
            self.net.load_state_dict(checkpoint['net'])
        elif path.split('.')[-1] == 'pt':
            # If you save the entire model
            self.net = torch.load(path)
        return

    def showWeights(self):
        # TODO
        # w_conv1 = ...
        # w_conv2 = ...
        # w_fc1 = ...
        # w_fc2 = ...
        # w_fc3 = ...

        plt.figure(figsize=(24, 6))
        plt.subplot(1,5,1)
        plt.title("conv1 weight")
        plt.hist(w_conv1)

        plt.subplot(1,5,2)
        plt.title("conv2 weight")
        plt.hist(w_conv2)

        plt.subplot(1,5,3)
        plt.title("fc1 weight")
        plt.hist(w_fc1)

        plt.subplot(1,5,4)
        plt.title("fc2 weight")
        plt.hist(w_fc2)

        plt.subplot(1,5,5)
        plt.title("fc3 weight")
        plt.hist(w_fc3)

        plt.savefig('weights.png')

if __name__ == '__main__':
    # you can adjust your hyperperamers
    ocr = FurnitureOCR('./Furniture', 75, 32, 0.001)