# -*- coding: utf-8 -*-
"""
Version: v1.2
Dataset: MNIST (60000 training + 10000 testing)
Input pipline: 
  1. Download 'mnist' dataset
  2. Shuffle and split in to training (60000) and testing (10000) dataset
  3. Normalize image from uchar to float
  4. Store in cache for small (<1GB) dataset
  5. Shuffle training dataset
  6. Set batch size = 128

Model:
  * ANN: fully connected multilayer perceptrons
  * Input layer: reshape 28 x 28 2D image to 1D array
  * Hiddlen layers: 5 (128, 256, 128, 256, 128) nodes
  * Activation func: Relu
  * Output layer: 10 classes, softmax

  ## Training
  * loss func: cross entropy
  * optimizer: Adam, learning rate = 0.001
  * epochs: 10
"""

import time, csv, json, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow.python.client import device_lib
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp

print(tf.__version__)
tfds.disable_progress_bar()
tf.enable_v2_behavior()

# Hide GPU from visible devices
# If not works, see https://stackoverflow.com/questions/37660312/how-to-run-tensorflow-on-cpu
tf.config.set_visible_devices([], 'GPU')

SHOW_FIG_FLAG = False

def show_dataset(ds_train, ds_info):
  # Know more about dataset
  print('Train = {}, Test = {}'.format(
      ds_info.splits['train'].num_examples,
      ds_info.splits['test'].num_examples)
  )

  print(type(ds_train))

  # Show some dataset
  #sess = tf.compat.v1.InteractiveSession()
  plt.figure(figsize=(10, 5))
  for i, data in enumerate(ds_train.take(10)):
    #print(type(data))
    #print(len(data))
    #print(type(data[0]))
    #print(data[0].shape)
    #print(data[1])
    #print(data[1].eval())
    img = tf.reshape(data[0], [28, 28]) # 28x28x1 tensor to 28x28 image
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(img, interpolation='nearest', 
                   cmap=plt.cm.Greys, vmin=0, vmax=255)
  plt.show()
  return
  
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

def plot_training_history(history, is_print_file = True):
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

    if is_print_file == True:
        with open("results/history.csv", "w") as file:
            file.write("{0},{1},{2},{3},{4}\n".format(
                "epoch", "accuracy", "val_accuracy", "loss", "val_loss"
            ))
            for i in range(len(accuracy)):
                file.write("{0},{1},{2},{3},{4}\n".format(
                    i, accuracy[i], val_accuracy[i], loss[i], val_loss[i]
                ))
    return

def Calculate_confusion_matrix(model, ds_train, ds_test):
    # Get 1-shot label from sparse_catagory
    # see https://stackoverflow.com/a/64622975/9265852
    true_categories_train = tf.concat([y for x, y in ds_train], axis=0)
    y_pred_train = list(model.predict(ds_train))
    predicted_categories_train = tf.argmax(y_pred_train, axis=1)
    cm_train = tf.math.confusion_matrix(true_categories_train, predicted_categories_train)

    true_categories_test = tf.concat([y for x, y in ds_test], axis=0)
    y_pred_test = list(model.predict(ds_test))
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
def Calculate_roc(model, dataset, filename):
    classes = [i for i in range(0,10)]
    n_classes = len(classes)

    # train dataset 
    true_categories = tf.concat([y for x, y in dataset], axis=0)
    true_categories = label_binarize(true_categories, classes=classes)

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
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('results'):
        os.makedirs('results')

    print("Available CPUs={}, GPUs={}: ".format(
      len(tf.config.experimental.list_physical_devices('CPU')),
      len(tf.config.experimental.list_physical_devices('GPU'))
    ))
  
    # Step 1 Build input pipeline
    (ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'],
        shuffle_files=True, as_supervised=True,  with_info=True,
        data_dir='data', download=True,
    )
  
    # Plot 10 training data
    #show_dataset(ds_train, ds_info)
    
    # Prepare training dataset
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    #ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
  
    # Prepare test dataset
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #ds_test = ds_test.shuffle(ds_info.splits['test'].num_examples)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
  
    # ANN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128,activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(256,activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(128,activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(256,activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(128,activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
  
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        metrics=['accuracy'],
    )

    # log training time
    log_file = open("results/log.txt", "w")
    start_t = time.time()
    log_file.write("start time {}\n".format( time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(start_t)) ))
 
    history =  model.fit(
        ds_train,
        epochs=10,
        validation_data=ds_test,
    )

    end_t = time.time()
    log_file.write("end time {}\n".format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(end_t))))
    log_file.write("elapsed time {}\n".format(end_t-start_t))
    log_file.close()

    plot_training_history(history, True)
  
    Calculate_confusion_matrix(model, ds_train, ds_test)
    
    Calculate_roc(model, ds_train, "train")
    Calculate_roc(model, ds_test, "test")
  
    return

if __name__ == "__main__":
    main()