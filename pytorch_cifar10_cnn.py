"""
run this commond on cmd: tensorboard --logdir=/PATH_OF_LOG_FILE/ --port 6006
---------------
tensorboardX to plot the graph.
"""

import time
import io

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_curve, auc
from tensorboardX import SummaryWriter  
import matplotlib.pyplot as plt
import itertools

# Dataset path
dataset_path = '../Pytorch_CIFAR10_CNN/data'
# Log path
log_path = '../Pytorch_CIFAR10_CNN/log'

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# TensorBoard Writter
writer = SummaryWriter(log_path)

# Transform
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Batch Size
batch_size = 128

# Create and load DataSets
trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNModel().to(device)

# Add model graph
train_data_sample, label_sample = iter(trainloader).next()
with writer:
    writer.add_graph(model, train_data_sample.to(device))  # model graph, with input

# Parameters
epochs = 10
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Cross Entropy Loss 
criterion = nn.CrossEntropyLoss()

# CNN model training
print('Start Training: ')
startTime = time.time()

for epoch in range(epochs): 
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0

    for times, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # taking the highest value of prediction.
        _, preds = torch.max(outputs, 1) 
        running_loss += loss.item()

        # calculating te accuracy by taking the sum of all the correct predictions in a batch.
        running_corrects += torch.sum(preds == labels.data) 

    else:
        # Test per epoch
        with torch.no_grad():
            for data in testloader:
                val_inputs, val_labels = data[0].to(device), data[1].to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                
                _, val_preds = torch.max(val_outputs, 1)
                val_running_loss += val_loss.item()
                val_running_corrects += torch.sum(val_preds == val_labels.data)

        # loss and accuracy per epoch
        epoch_loss = running_loss/len(trainloader) 
        epoch_acc = running_corrects.float()/ len(trainloader) 
        val_epoch_loss = val_running_loss/len(testloader)
        val_epoch_acc = val_running_corrects.float()/ len(testloader)

        writer.add_scalars('Loss/Epoch', {'train': epoch_loss}, epoch)
        writer.add_scalars('Loss/Epoch', {'val': val_epoch_loss}, epoch)
        writer.flush()

        writer.add_scalars('Accuracy/Epoch', {'train': epoch_acc}, epoch)
        writer.add_scalars('Accuracy/Epoch', {'val': val_epoch_acc}, epoch)
        writer.flush()

        print('epoch[%02d/%02d] | training loss: %.4f | accuracy: %4.2f%%' % (epoch + 1, epochs, epoch_loss, epoch_acc))
        print('             |      val loss: %.4f | accuracy: %4.2f%%' % (val_epoch_loss, val_epoch_acc))

endTime = time.time()
print('Finished Training. Time used: %.5f' % (endTime - startTime))

# Utils
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for data in loader:
        images, labels = data[0].to(device), data[1].to(device)
        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )

    return all_preds

def test_class_probabilities(whichClass):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            actuals.extend(labels.view_as(predicted) == whichClass)
            probabilities.extend(np.exp(outputs[:, whichClass]))
    return actuals, probabilities

# Draw Confusion Matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):

    fig = plt.figure(figsize=(10,10))

    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig

# Draw ROC
def plot_roc( actuals,  probabilities, n_classes):
    
    fig = plt.figure()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(n_classes)):
        fpr[i], tpr[i], _ = roc_curve(actuals[i], probabilities[i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='%s (area = %.2f)' % (classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve to multi-class')
    plt.legend(loc="lower right")

    return fig

# Calculate Confusion Matrix, Precision, Recall, F1-Score for each class
recall_list = []
precision_list = []
actuals_list = []
probabilities_list = []
with torch.no_grad():
    test_preds = get_all_preds(model, testloader)

    stacked = torch.stack(
        (
            torch.as_tensor(testset.targets)
            ,test_preds.argmax(dim=1)
        )
        ,dim=1
    )

    cmt = torch.zeros(10,10, dtype=torch.int64)

    print ('\nClassification Report: ')
    print (' class | precision | recall | f1-score ' )
    print ('-------+-----------+--------+----------' )

    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1

    FP = (cmt.sum(axis=0) - np.diag(cmt)).numpy()
    FN = (cmt.sum(axis=1) - np.diag(cmt)).numpy()
    TP = np.diag(cmt)
    TN = (cmt.sum() - (FP + FN + TP)).numpy()

    for i in range(len(classes)):
        precision_list.append(TP[i] / (TP[i] + FP[i]))
        recall_list.append(TP[i] / (TP[i] + FN[i]))

        fscore = (2 * (precision_list[i] * recall_list[i])) / (precision_list[i] + recall_list[i])

        print(' %5s |    %2.2f   |  %2.2f  |   %2.2f ' % (classes[i], precision_list[i], recall_list[i], fscore))

        #ROC
        actuals, class_probabilities = test_class_probabilities(i)
        actuals_list.append(actuals)
        probabilities_list.append(class_probabilities)
    
    # Plot the Graph to TensorboardX
    fig = plot_confusion_matrix(cmt, testset.classes)
    writer.add_figure('Confusion Matrix', fig)
    writer.flush()

    fig = plot_roc(actuals_list, probabilities_list, classes)
    writer.add_figure('ROC', fig)
    writer.flush()
