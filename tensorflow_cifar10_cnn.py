"""
version: 1.2
Dataset: CIFAR10, 10 animal classes, color images 50000 training and 10000 testing images

Model: Convolutional Neural Network (CNN)
- inpuit layer:
- hidden layer: conv2d, maxpo
- output layer:
- activation func: Relu

Training:
- optimizer: adam, learning rate = ?
- loss: cross entropy

"""

import time, csv, json, os
from itertools import cycle
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np

SHOW_FIG_FLAG = False

def show_dataset(train_images, train_labels):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)

        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

    return

def plot_training_history(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(accuracy, label='accuracy')
    axs[0].plot(val_accuracy, label = 'val_accuracy')
    #axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')
    axs[0].grid(True)

    axs[1].plot(loss, label='loss')
    axs[1].plot(val_loss, label='val_loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='lower right')
    axs[1].grid(True)
  
    if SHOW_FIG_FLAG:
        plt.show()
    plt.savefig('results/accuracy_vs_epoch.png')

    with open("results/history.csv", "w") as file:
        file.write("{0},{1},{2},{3},{4}\n".format(
            "epoch", "accuracy", "val_accuracy", "loss", "val_loss"
        ))
        for i in range(len(accuracy)):
            file.write("{0},{1},{2},{3},{4}\n".format(
                i, accuracy[i], val_accuracy[i], loss[i], val_loss[i]
            ))

    return

def Calculate_confusion_matrix(model, train_images, train_labels, test_images, test_labels):
    # Get 1-shot label from sparse_catagory
    # see https://stackoverflow.com/a/64622975/9265852
    true_categories_train = train_labels
    y_pred_train = list(model.predict(train_images))
    predicted_categories_train = tf.argmax(y_pred_train, axis=1)

    cm_train = tf.math.confusion_matrix(true_categories_train, predicted_categories_train)

    true_categories_test = test_labels
    y_pred_test = list(model.predict(test_images))
    predicted_categories_test = tf.argmax(y_pred_test, axis=1)
    cm_test = tf.math.confusion_matrix(true_categories_test, predicted_categories_test)

    with open("results/ConfusionMatrix_train.csv", "w", newline = "") as file:
        spamwriter = csv.writer(file, delimiter=',')
        spamwriter.writerows(cm_train.numpy())
    
    with open("results/ConfusionMatrix_test.csv", "w", newline = "") as file:
        spamwriter = csv.writer(file, delimiter=',')
        spamwriter.writerows(cm_test.numpy())

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
    true_categories = label_binarize(label, classes=classes)

    # predicted label
    y_pred = model.predict(dataset)

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
    if not os.path.exists('results'):
        os.makedirs('results')

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    #show_dataset(train_images, train_labels)

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # define model
    model = models.Sequential()

    model.add(layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    # compile model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        metrics=['accuracy']
    )

    # log training time
    log_file = open("results/log.txt", "w")
    start_t = time.time()
    log_file.write("start time {}\n".format( time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(start_t)) ))
    
    # train model
    history = model.fit(
        train_images, train_labels, 
        epochs=2,
        validation_data=(test_images, test_labels)
    )
    
    end_t = time.time()
    log_file.write("end time {}\n".format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(end_t))))
    log_file.write("elapsed time {}\n".format(end_t-start_t))
    log_file.close()

    plot_training_history(history)
  
    Calculate_confusion_matrix(model, train_images, train_labels, test_images, test_labels)

    Calculate_roc(model, train_images, train_labels, "train")
    Calculate_roc(model, test_images, test_labels, "test")

    return


if __name__ == "__main__":
    main()
