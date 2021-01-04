import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
class InceptionV4(nn.Module):
    def __init__(self):
        super(InceptionV4, self).__init__()

    def forward(self):
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
        #self.test()

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
        self.net = InceptionV4()
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
        torch.save(self.net.state_dict(), './v3weight.t7')

        # you also can store some log information
        state = {
            'net': self.net.state_dict(),
            'acc': self.train_acc,
            'epoch': self.epoch
        }
        torch.save(state, './v3weight.t7')

        # save entire model
        torch.save(self.net, './v3model.pt')
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
    ocr = FurnitureOCR('./Furniture', 120, 32, 0.001)