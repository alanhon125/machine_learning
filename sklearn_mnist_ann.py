"""
Version: 1.2
Dataset: MNIST

## Input pipeline
1. Download dataset from openml
2. Cache to memory
3. Shuffle data
4. Split into train (60000) and test (10000)
4. Normalize from uchar to float

## Scikit learn features
- No GPU support
- Can't do CNN
- loss function only support cross entropy
- normalization only support L2 penality

## Model architecture
ANN: fully connected multilayer perceptron
* Input layer:
* Hidden layer: 5, (128, 256, 128, 256, 128) nodes
* Output layer:

## Training Strategy
* Solver: Adam, init learning rate = 0.01, adaptive learning rate
* Regularization: L2 penalty, alpha=1e-4 
* batch size: 128
* epoches (max_iter): 10

""" 

import time, csv, json, os
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from pathlib import Path
from matplotlib.pyplot import xcorr
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

SHOW_FIG_FLAG = False

def show_dataset(dataset, label):
    # Confirm the properties of input dataset
    print('shape of data = {}'.format(dataset[0].shape))
    print('shape of label = {}'.format(label.shape))
    #print(dataset[0])
    #print(label[0])

    # show some data
    plt.figure(figsize=(10, 5))
    for i in range(10):
        l1_plot = plt.subplot(2, 5, i + 1)
        l1_plot.imshow(dataset[i].reshape(28, 28), interpolation='nearest',
                       cmap=plt.cm.Greys, 
                       vmin=0, vmax=255)
    plt.show()

    return

def visualize_mlp_weights(mlp):
    fig, axes = plt.subplots(4, 4)
    
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
    for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
                   vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())

    plt.show()
    return

def plot_training_history(mlp, loss, scores_train, scores_test):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(scores_train, label='train_accuracy')
    axs[0].plot(scores_test, label = 'test_accuracy')
    #axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')
    axs[0].grid(True)

    axs[1].plot(loss, label='loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='lower right')
    axs[1].grid(True)
    
    if SHOW_FIG_FLAG:
        plt.show()
    plt.savefig('results/accuracy_vs_epoch.png')

    with open("results/history.csv", "w") as file:
        file.write("{0},{1},{2},{3}\n".format(
            "epoch", "accuracy", "val_accuracy", "loss"
        ))
        for i in range(len(scores_train)):
            file.write("{0},{1},{2},{3}\n".format(
                i, scores_train[i], scores_test[i], loss[i]
            ))

    return

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

def shuffle_dataset(X, y):
    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    print('Original shape = {}'.format(X.shape))
    X = X.reshape((X.shape[0], -1)) # why??
    print('After reshape = {}'.format(X.shape))
    return X,y

def normalize_image(images):
    return images / 255.0

def buildin_training(X_train, y_train, X_test, y_test, N_EPOCHS, N_BATCH):
    # build model
    mlp = MLPClassifier(hidden_layer_sizes=(128, 256, 128, 256, 128),
        shuffle=True, #shuffle samples in each iteration
        random_state=1, batch_size=N_BATCH,
        max_iter=N_EPOCHS, 
        alpha=1e-4, # L2 penality
        solver='adam', learning_rate_init=.01, learning_rate='adaptive', #beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        verbose=10, # print msg
    )

    # Training with scikit-learn API
    mlp.fit(X_train, y_train)
    return mlp

# It seems scikit learn don't have build-in function to log accuracy of each epoch
# Train and do custom logging: https://stackoverflow.com/a/46913459/9265852
def customized_training(X_train, y_train, X_test, y_test, N_EPOCHS, N_BATCH):
    N_TRAIN_SAMPLES = X_train.shape[0]
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []
    loss = []
    
    # build model
    mlp = MLPClassifier(hidden_layer_sizes=(128, 256, 128, 256, 128),
        shuffle=True, #shuffle samples in each iteration
        random_state=1, #batch_size=128,
        max_iter=N_EPOCHS, 
        alpha=1e-4, # L2 penality
        solver='adam', learning_rate_init=.01, learning_rate='adaptive', beta_1=0.9, beta_2=0.999, epsilon=1e-7,
        verbose=0, # print msg
    )

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # Accuracy TRAIN
        scores_train.append(mlp.score(X_train, y_train))

        # Accuracy TEST
        scores_test.append(mlp.score(X_test, y_test))
        
        # Loss 
        loss.append(mlp.loss_)
        
        # NOTE(michael): mlp.loss_curve_ contains loss value after training for every batch,
        # which is not necessory

        epoch += 1

    plot_training_history(mlp, loss, scores_train, scores_test)

    return mlp

def Calculate_confusion_matrix(mlp, X_train, y_train, X_test, y_test):
    # Confusion matrix
    # Accuracy of the model can also calculated by mlp.score(X_train, y_train)
    y_train_predict = mlp.predict(X_train)
    cm_train = confusion_matrix(y_train, y_train_predict)
    #print("Accuracy of MLPClassifier on training set = {}".format(accuracy(cm)))
    
    y_test_predict = mlp.predict(X_test)
    cm_test = confusion_matrix(y_test, y_test_predict)
    #print("Accuracy of MLPClassifier on test set = {}".format(accuracy(cm)))

    with open("results/ConfusionMatrix_train.csv", "w", newline = "") as file:
        spamwriter = csv.writer(file, delimiter=',')
        spamwriter.writerows(cm_train)
    
    with open("results/ConfusionMatrix_test.csv", "w", newline = "") as file:
        spamwriter = csv.writer(file, delimiter=',')
        spamwriter.writerows(cm_test)

    return

def output_json(filepath, dic):
    for k in dic:
        val = dic[k]
        if type(val).__module__ == np.__name__:
            dic[k] = val.tolist()
    with open(filepath, "w", newline = "") as f:
        json.dump(dic, f)
    return 
    
"""
 For using scikit-learn roc_curve function, see 
 https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
 
 For ploting roc_curve in tensorflow, see
 https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
"""
def Calculate_roc(model, dataset, label, filename):
    classes = [i for i in range(0,10)]
    n_classes = len(classes)

    # true label
    label = np.asarray([int(l) for l in label])
    true_categories = label_binarize(label, classes=classes)

    # predicted label
    y_pred = model.predict_proba(dataset)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_categories[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_categories.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2 # line width
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    
    if SHOW_FIG_FLAG:
        plt.show()
    plt.savefig('results/roc_curve{}.png'.format(filename))

    output_json("results/FalsePositiveRate_{}.json".format(filename), fpr)
    output_json("results/TruePositiveRate_{}.json".format(filename), tpr)
    output_json("results/UC_ROC_{}.json".format(filename), roc_auc)
    return 

def main():
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('results'):
        os.makedirs('results')

    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, # 784=28*28, dataset is 1D, 
        return_X_y=True, # X is dataset, y is label 
        data_home='data',
        cache = True,
    )

    # Display 10 train data
    #show_dataset(X, y)

    # Shuffle data 
    X, y = shuffle_dataset(X,y)

    # Normalize from uchar to float
    X = normalize_image(X)

    # Split dataset into 'train' and 'test'
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=60000, test_size=10000)

    # log training time
    log_file = open("results/log.txt", "w")
    start_t = time.time()
    log_file.write("start time {}\n".format( time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(start_t)) ))

    #mlp = buildin_training(X_train, y_train, X_test, y_test, N_EPOCHS=10, N_BATCH=128)
    mlp = customized_training(X_train, y_train, X_test, y_test, N_EPOCHS=10, N_BATCH=128)

    end_t = time.time()
    log_file.write("end time {}\n".format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(end_t))))
    log_file.write("elapsed time {}\n".format(end_t-start_t))
    log_file.close()

    Calculate_confusion_matrix(mlp, X_train, y_train, X_test, y_test)

    Calculate_roc(mlp, X_train, y_train, "train")
    Calculate_roc(mlp, X_test, np.asarray([int(l) for l in y_test]), "test")

    #visualize_mlp_weights(mlp)

    return

if __name__ == "__main__":
    main()