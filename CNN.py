import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import math
from skimage.util.shape import view_as_windows

# Visualising MNIST dataset
def show_dataset(x_ds, y_ds, dataset):
    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2
    plt.title("These are the handwritten digits from {}".format(dataset))
    for i in range(1, 11):
        digit_index = random.randrange(len(y_ds))
        while y_ds[digit_index] != i - 1:
            digit_index = random.randrange(len(y_ds))
        digit_data = x_ds[digit_index]
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(digit_data)
    plt.axis('off')
    plt.show()

# Visualising Filters
def show_filter(model, layer=1):
    fig = plt.figure(figsize=model.layers[layer].kernel_size)

    # retrieve weights from the second hidden layer
    filters, biases = model.layers[layer].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    square, ix = int(math.sqrt(filters.shape[3])), 1

    plt.title(model.layers[layer].name)
    # plot all 25 filters maps in an 5x5 squares
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            fig.add_subplot(square, square, ix)
            plt.axis('off')
            # plot filter channel
            plt.imshow(filters[:, :, 0, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    plt.axis('off')
    plt.show()

# Visualising patches with top 12 highest activation with random 5 filters in the first convolutional layer on a random image in x_test
def show_patch(model, x, num_filters=5):
    # Load the images with the required shape
    x = x.reshape((-1, 1, 28, 28, 1))
    # Random pick a image
    random_index = random.randrange(x.shape[0])
    img = x[random_index]
    # Get all patches with 12x12 filter and stride 2x2
    window_shape = (1,12,12,1)
    patches = view_as_windows(img, window_shape, step=2)
    # Redefine model to output right after the first hidden layer
    model = Model(inputs=model.inputs, outputs=model.layers[1].output)
    # Get feature map for first hidden layer
    feature_maps = model.predict(img)
    # Random pick 5 filters
    random_filter_index = [random.randrange(feature_maps.shape[3]) for i in range(num_filters)]
    # Get feature maps for 5 random filters
    for i in random_filter_index:
      feature_map = feature_maps[0,:,:,i]
      # Get array of indices of the sorted elements of a feature_map
      index = list(np.unravel_index(np.argsort(feature_map, axis=None), feature_map.shape))
      fig = plt.figure(figsize=(model.layers[1].kernel_size))
      plt.title('Top 12 patches in the highest activation in filter #%d'% i)
      # Visualize patches with the top 12 high activation
      for j in range(12):
        col = index[0][-1]
        row = index[1][-1]
        index[0] = index[0][:-1]
        index[1] = index[1][:-1]
        # Specify subplot and turn of axis
        fig.add_subplot(3, 4, j+1)
        plt.axis('off')
        # plot the patch
        plt.imshow(patches[0,row,col,0,:,:][0,:,:,0], cmap=None)
      plt.axis('off')
      plt.show()

# Building a Simple CNN Network
def my_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(25, 12, strides=(2, 2), activation='relu')(inputs)
    x = layers.Conv2D(64, 5, strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Load data, reshape and normalize the images
def preprocess_image():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)

# Plot training, validation accuracy and loss
def plot_history(history):
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    iterations = history['iterations']

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(iterations, accuracy, label='accuracy')
    axs[0].plot(iterations, val_accuracy, label='val_accuracy')
    # axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')
    axs[0].grid(True)

    axs[1].plot(iterations, loss, label='loss')
    axs[1].plot(iterations, val_loss, label='val_loss')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    fig.tight_layout()
    plt.show()

# Train the model and show the performance metrics every 100 iterations of training. After training, save the model and metrics on current directory
def custom_training(model, x_train, y_train, num_samples, batch_size=50, epochs=2):
    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Prepare the metrics.
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    # Reserve num_samples samples for validation.
    x_val = x_train[-num_samples:]
    y_val = y_train[-num_samples:]
    x_train = x_train[:-num_samples]
    y_train = y_train[:-num_samples]

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    # Training step
    @tf.function
    def train_step(model, x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
            train_acc = train_acc_metric.result()
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, logits)
        return loss_value, train_acc

    # Testing step
    @tf.function
    def test_step(model, x, y):
        val_logits = model(x, training=False)
        loss_value = loss_fn(y, val_logits)
        val_acc_metric.update_state(y, val_logits)
        return loss_value

    history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': [],
        'iterations': []}

    # Iterate over the epoch
    for epoch in range(epochs):
        print("\nEpoch %d/%d" % (epoch + 1, epochs))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value, train_acc = train_step(model, x_batch_train, y_batch_train)

            # Log every 100 batches (i.e. every 100 iterations).
            if step % 100 == 0:
                print(
                    "Training loss, accuracy (for one batch) at iteration %d: %.4f, %.4f"
                    % (step, float(loss_value), float(train_acc))
                )
                # Run a validation loop at the end of every 100 batches.
                for x_batch_val, y_batch_val in val_dataset:
                    val_loss_value = test_step(model, x_batch_val, y_batch_val)
                    val_acc = val_acc_metric.result()
                print("Validation loss, accuracy (for one batch) at iteration %d: %.4f, %.4f"
                      % (step, float(val_loss_value), float(val_acc)))
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

        iterations = len(x_train) / batch_size * (epoch + 1)
        history['iterations'].append(iterations)
        history['accuracy'].append(train_acc)
        history['loss'].append(loss_value)
        history['val_accuracy'].append(val_acc)
        history['val_loss'].append(val_loss_value)

        # Display metrics at the end of each epoch.
        # train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of training epochs
        if epoch == epochs - 1:
            train_acc_metric.reset_states()
            val_acc_metric.reset_states()

        print("Time taken: %.2fs" % (time.time() - start_time))

    model.save('my_model.h5')
    joblib.dump(history, 'history.pkl')

    return history

def start(load_model=False):
    (x_train, y_train), (x_test, y_test) = preprocess_image()

    # Show the handwritten digit image in sequence from trainset and testset
    show_dataset(x_train, y_train, 'trainset')
    show_dataset(x_test, y_test, 'testset')

    # If load_model==True, then we just load the model and history from current directory, otherwise create a new one
    if not load_model:
        model = my_model()
        show_filter(model)  # Show filter before training
    else:
        model = tf.keras.models.load_model('my_model.h5')
        history = joblib.load('history.pkl')

    print(model.summary())  # Print model architecture
    if not load_model:
        history = custom_training(model, x_train, y_train, 10000, epochs=5)
    plot_history(history)

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(lr=1e-4),
        metrics=["accuracy"],
    )

    model.evaluate(x_test, y_test, batch_size=50)

    show_filter(model)  # show filter after training
    show_patch(model, x_test)

if __name__ == "__main__":
    start(load_model=True)