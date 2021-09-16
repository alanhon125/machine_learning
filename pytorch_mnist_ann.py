import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dset
from torchvision import datasets, transforms

import numpy as np
from sklearn.metrics import roc_curve, auc
from tensorboardX import SummaryWriter  
import matplotlib.pyplot as plt
import itertools

# Dataset path
dataset_path = '../Pytorch_MNIST_ANN/data'
# Log path
log_path = '../Pytorch_MNIST_ANN/log'

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# TensorBoard Writter
writer = SummaryWriter(log_path)

# Transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),]
)

# Data
batch_size = 128
trainset = datasets.MNIST(root=dataset_path, download=True, train=True, transform=transform)
testset = datasets.MNIST(root=dataset_path, download=True, train=False, transform=transform)
trainloader = dset.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = dset.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Create Ann Model
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        return self.main(input)

# Using gpu if CUDA supported
model = ANNModel().to(device)

# Add model graph
train_data_sample, _ = iter(trainloader).next()
with writer:
    train_data_sample = train_data_sample.view(train_data_sample.shape[0], -1)
    writer.add_graph(model, train_data_sample.to(device))  # model graph, with input

# Parameters
epochs = 10
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Cross Entropy Loss 
criterion = nn.CrossEntropyLoss()

# ANN model training
print('Start Training: ')
startTime = time.time()

for epoch in range(epochs):
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0
    train_total = 0
    val_total = 0
    for _, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.view(inputs.shape[0], -1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Foward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # taking the highest value of prediction.
        _, preds = torch.max(outputs, 1) 
        running_loss += loss.item()

        # calculating te accuracy by taking the sum of all the correct predictions in a batch.
        running_corrects += torch.sum(preds == labels.data)
        train_total += len(labels.data)
    else:
        # Test per epoch
        with torch.no_grad():
            for data in testloader:
                val_inputs, val_labels = data[0].to(device), data[1].to(device)
                val_inputs = val_inputs.view(val_inputs.shape[0], -1)

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                
                _, val_preds = torch.max(val_outputs, 1)
                val_running_loss += val_loss.item()
                val_running_corrects += torch.sum(val_preds == val_labels.data)

                val_total += len(val_labels.data)

        # loss and accuracy per epoch
        epoch_loss = running_loss/len(trainloader) 
        epoch_acc = 100 * (running_corrects.float()/ train_total)
        val_epoch_loss = val_running_loss/len(testloader)
        val_epoch_acc = 100 * (val_running_corrects.float()/ val_total)

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
        images = images.view(images.shape[0], -1)
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
            inputs = inputs.view(inputs.shape[0], -1)
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
def plot_roc( actuals,  probabilities):
    
    fig = plt.figure()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(actuals[i], probabilities[i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='%s (area = %.2f)' % (i, roc_auc[i]))

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

    for i in range(10):
        precision_list.append(TP[i] / (TP[i] + FP[i]))
        recall_list.append(TP[i] / (TP[i] + FN[i]))

        fscore = (2 * (precision_list[i] * recall_list[i])) / (precision_list[i] + recall_list[i])

        print(' %5s |    %2.2f   |  %2.2f  |   %2.2f ' % (i + 1, precision_list[i], recall_list[i], fscore))

        #ROC
        actuals, class_probabilities = test_class_probabilities(i)
        actuals_list.append(actuals)
        probabilities_list.append(class_probabilities)
    
    # Plot the Graph to TensorboardX
    fig = plot_confusion_matrix(cmt, testset.classes)
    writer.add_figure('Confusion Matrix', fig)
    writer.flush()

    fig = plot_roc(actuals_list, probabilities_list)
    writer.add_figure('ROC', fig)
    writer.flush()
