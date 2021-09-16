import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
# Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
# from tensorflow.python.compiler.mlcompute import mlcompute
from sklearn.preprocessing import OneHotEncoder

# Select CPU device.
# mlcompute.set_mlc_device(device_name='any') # Available options are 'cpu', 'gpu', and 'any'.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data(encoder=None):
    (x_train, Y_train), (x_test, Y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
    if encoder == 'one-hot':
        Y_train = tf.one_hot(Y_train, 10)
        Y_test = tf.one_hot(Y_test, 10)
    elif encoder == 'binary':
        Y_train = np.unpackbits(np.array(Y_train)).reshape(-1, 8)[:, 4:]
        Y_test = np.unpackbits(np.array(Y_test)).reshape(-1, 8)[:, 4:]

    return x_train, Y_train, x_test, Y_test

def network(num_hl=1, neurons=256, activation='relu', out_act='softmax', regularizer=None, dropout=0,encoder=None):
    imgs = tf.keras.Input(shape=784)
    if activation == 'lrelu':
        activation = tf.keras.layers.LeakyReLU(alpha=0.01)
    if regularizer == 'l1':
        regularizer = tf.keras.regularizers.l1(1e-4)
    elif regularizer == 'l2':
        regularizer = tf.keras.regularizers.l2(1e-4)
    elif regularizer == 'l1l2':
        regularizer = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)

    hidden_layer = tf.keras.layers.Dense(neurons, activation=activation)(imgs)
    for i in range(num_hl - 1):
        hidden_layer = tf.keras.layers.Dense(neurons, activation=activation)(hidden_layer)
        hidden_layer = tf.keras.layers.Dropout(rate=dropout)(hidden_layer)
    if encoder == 'one-hot':
        output_nodes = 10
    elif encoder == 'binary':
        output_nodes = 4
    else:
        output_nodes = 10
        out_act = 'softmax'

    if out_act == 'no function':
        y_pred = tf.keras.layers.Dense(output_nodes, kernel_regularizer=regularizer)(hidden_layer)
    else:
        y_pred = tf.keras.layers.Dense(output_nodes, activation=out_act, kernel_regularizer=regularizer)(hidden_layer)

    net = tf.keras.Model(inputs=imgs, outputs=y_pred)
    return net


def optimization(net, loss='cross-entropy', lr=0.001, momentum=0):
    if loss == 'L1 loss':
        loss_function = tf.keras.losses.MeanAbsoluteError()
    elif loss == 'L2 Loss':
        loss_function = tf.keras.losses.MeanSquaredError()
    elif loss == 'cross-entropy':
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    net.compile(
        loss=loss_function,
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum),
        # optimizer = tf.keras.optimizers.Adam(lr=lr),
        metrics=["accuracy"]
    )


def train(x_train, Y_train, net, batch_size=32, epoch=5):
    history = net.fit(x_train, Y_train, batch_size=batch_size, epochs=epoch)
    accuracy = history.history['accuracy']
    return net, accuracy[-1]


def test(x_test, Y_test, net, batch_size=32):
    loss, accuracy = net.evaluate(x_test, Y_test, batch_size=batch_size)
    return accuracy


def plot(train_accuracy, test_accuracy, num):
    plt.title('Accuracy vs Dropout rate')
    plt.xlabel('Dropout rate')
    plt.ylabel('Accuracy')
    plt.plot(num, train_accuracy, color='red', label='train')
    plt.plot(num, test_accuracy, color='blue', label='test', linestyle='dashed')
    plt.legend()
    plt.show()
    plt.close()


def plot_with_runtime(train_accuracy, test_accuracy, num, run_time):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('number of neurons in middle layer')
    ax1.set_ylabel('Accuracy')
    ax1.plot([32, 64, 128, 256, 512], train_accuracy[:5], color='red', label='train_{}'.format(num[0][0]))
    ax1.plot([32, 64, 128, 256, 512], test_accuracy[:5], color='red', label='test_{}'.format(num[0][0]),
             linestyle='dashed')
    ax1.plot([32, 64, 128, 256, 512], train_accuracy[5:], color='blue', label='train_{}'.format(num[5][0]))
    ax1.plot([32, 64, 128, 256, 512], test_accuracy[5:], color='blue', label='test_{}'.format(num[5][0]),
             linestyle='dashed')
    ax1.tick_params(axis='y')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.set_ylabel('Run time (s)')
    ax2.plot([32, 64, 128, 256, 512], run_time[:5], color='red', label='run_time_{}'.format(num[0][0]),
             linestyle='dashdot')
    ax2.plot([32, 64, 128, 256, 512], run_time[5:], color='blue', label='run_time_{}'.format(num[5][0]),
             linestyle='dashdot')
    ax2.tick_params(axis='y')
    ax2.legend()

    fig.tight_layout()
    plt.show()
    plt.close()


def plot_bar(train_accuracy, test_accuracy, cat):
    plt.title('Accuracy vs Regularizer')
    plt.xlabel('Regularizer')
    plt.ylabel('Accuracy')

    N = len(cat)
    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, train_accuracy, width, color='red', label='train')
    plt.bar(ind + width, test_accuracy, width, color='blue', label='test')
    plt.xticks(ind + width / 2, cat)
    plt.legend(loc='best')
    plt.show()


def demo():
    train_acc_list = []
    test_acc_list = []
    label = []
    neurons = [32, 64, 128, 256, 512]
    activation = ['lrelu', 'relu', 'sigmoid', 'tanh']
    out_act = ['softmax', 'sigmoid', 'relu', 'no function']
    loss = ['L1 loss', 'L2 Loss', 'cross-entropy']
    lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    batch_size = [1, 8, 16, 32, 64, 128, 256]
    total_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 12, 24]
    regularizer = ['l1', 'l2', 'l1l2']
    encoder = ['one-hot', 'binary']

    for i in encoder:
        print('Regularizer: ', i)
        ### stage 1: prepare dataset
        x_train, Y_tain, x_test, Y_test = load_data(encoder=i)
        ### stage 2: design model
        net = network()
        net.summary()
        optimization(net)

        ### stage 3: train the model
        net, train_accuracy = train(x_train, Y_tain, net)

        ### stage 4: evaluate the model
        test_accuracy = test(x_test, Y_test, net)
        label.append(i)
        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)

    plot_bar(train_acc_list, test_acc_list, label)


demo()