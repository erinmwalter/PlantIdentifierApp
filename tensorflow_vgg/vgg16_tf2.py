import os
import inspect
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16(Model):
    def __init__(self, vgg16_npy_path=None, num_classes=1000):
        super(Vgg16, self).__init__() # init tf.keras.model
        #finds npy file
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            vgg16_npy_path = os.path.join(path, "vgg16.npy")
        #loads npy file and puts dict in self.data_dict for later
        print(f"Loading VGG weights from: {vgg16_npy_path}")
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1', allow_pickle=True).item()

        # Convolutional + Pooling layers
        self.conv_blocks = []
        layer_config = [2, 2, 3, 3, 3]  # VGG16 structure

        #in_channels = 3
        for i, num_convs in enumerate(layer_config, 1):
            block = []
            for j in range(num_convs):
                name = f"conv{i}_{j+1}"
                weights, biases = self.data_dict[name]
                conv = layers.Conv2D(
                    filters=weights.shape[3],
                    kernel_size=(3, 3),
                    padding='same',
                    activation='relu',
                    name=name
                )
                # Build the layer (so weights exist) and set pretrained weights
                conv.build((None, None, None, weights.shape[2]))
                conv.set_weights([weights, biases])
                block.append(conv)

            block.append(layers.MaxPool2D((2, 2), strides=(2, 2), name=f"pool{i}"))
            self.conv_blocks.append(block)

        # Fully connected layers (you can replace these for 71 class)
        # fc6
        w6, b6 = self.data_dict["fc6"]
        self.fc6 = layers.Dense(4096, activation='relu', name='fc6')
        self.fc6.build((None, 7 * 7 * 512))  # VGG16 fc6 input shape
        self.fc6.set_weights([w6.reshape(7 * 7 * 512, 4096), b6])

        # fc7
        w7, b7 = self.data_dict["fc7"]
        self.fc7 = layers.Dense(4096, activation='relu', name='fc7')
        self.fc7.build((None, 4096))
        self.fc7.set_weights([w7, b7])

        # fc8 (classification)
        w8, b8 = self.data_dict["fc8"]
        self.fc8 = layers.Dense(num_classes, activation='softmax', name='fc8')

        # Load weights only if same output size (e.g., 1000 for ImageNet)
        if num_classes == 1000 and w8.shape[1] == 1000:
            self.fc8.build((None, 4096))
            self.fc8.set_weights([w8, b8])
        else:
            print(f" Skipping fc8 weight load (num_classes={num_classes} != 1000)")

        print(" VGG16 model initialized with pretrained weights.")

    def call(self, inputs):
        # inputs: RGB images scaled [0,1]
        x = inputs * 255.0
        # Convert RGB to BGR and subtract mean
        r, g, b = tf.split(x, 3, axis=3)
        bgr = tf.concat(
            [b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]],
            axis=3
        )

        # Forward pass through conv blocks
        for block in self.conv_blocks:
            for layer in block:
                bgr = layer(bgr)
        x = bgr

        # Flatten and FC layers
        x = layers.Flatten()(x)
        x = self.fc6(x)
        x = self.fc7(x)
        out = self.fc8(x)
        return out
