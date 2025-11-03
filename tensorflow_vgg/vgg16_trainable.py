import tensorflow as tf
import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16(tf.keras.Model):
    """
    A trainable version of VGG16, compatible with TensorFlow 2.x
    """

    def __init__(self, vgg16_npy_path=None, num_classes=71, dropout=0.5):
        super(Vgg16, self).__init__()
        self.dropout_rate = dropout
        self.num_classes = num_classes
        
        #Load pre-trained weights if provided
        #if vgg16_npy_path is not None:
            #self.data_dict = np.load(vgg16_npy_path, allow_pickle=True, encoding='latin1').item()
        #else:
            #self.data_dict = None

        # --- Convolutional layers ---
        self.conv1_1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv1_1')
        self.conv1_2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv1_2')
        self.pool1 = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='pool1')

        self.conv2_1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2_1')
        self.conv2_2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2_2')
        self.pool2 = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='pool2')

        self.conv3_1 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='conv3_1')
        self.conv3_2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='conv3_2')
        self.conv3_3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='conv3_3')
        self.pool3 = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='pool3')

        self.conv4_1 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='conv4_1')
        self.conv4_2 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='conv4_2')
        self.conv4_3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='conv4_3')
        self.pool4 = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='pool4')

        self.conv5_1 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='conv5_1')
        self.conv5_2 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='conv5_2')
        self.conv5_3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='conv5_3')
        self.pool5 = tf.keras.layers.MaxPool2D(2, 2, padding='same', name='pool5')

        # --- Custom classifier ---
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')
        self.fc6 = tf.keras.layers.Dense(512, activation='relu', name='fc6')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fc7 = tf.keras.layers.Dense(num_classes, activation='softmax', name='fc7')

        # Initialize weights if data_dict exists
        #if self.data_dict is not None:
            #self._load_weights_from_npy()

    def call(self, inputs, training=False):
        # Scale and BGR conversion
        x = inputs * 255.0
        red, green, blue = tf.split(x, num_or_size_splits=3, axis=-1)
        x = tf.concat([blue - VGG_MEAN[0],
                       green - VGG_MEAN[1],
                       red - VGG_MEAN[2]], axis=-1)

        # --- Forward pass ---
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        x = self.global_avg_pool(x)
        x = self.fc6(x)
        x = self.dropout(x, training=training)
        x = self.fc7(x)

        return x

    def train_step(self, data):
        tf.print("Data passed to train_step:", data)
        print("DEBUG data type:", type(data))
        if isinstance(data, (list, tuple)):
             print("DEBUG data length:", len(data))
             for i, d in enumerate(data):
                   print(f"DEBUG element {i}:", type(d), getattr(d, 'shape', None))
        else:
             print("DEBUG data:", data)

        x, y = data
        ...

    def load_weights_from_npy(self, npy_path):
        """Load pre-trained weights from .npy file AFTER model is built"""
        if npy_path is None:
            print("No weights file provided")
            return
        
        self.data_dict = np.load(npy_path, encoding='latin1', allow_pickle=True).item()
        print(f"Loading weights from {npy_path}...")
        
        # Map of layer names in the .npy file
        layer_map = {
            'conv1_1': self.conv1_1,
            'conv1_2': self.conv1_2,
            'conv2_1': self.conv2_1,
            'conv2_2': self.conv2_2,
            'conv3_1': self.conv3_1,
            'conv3_2': self.conv3_2,
            'conv3_3': self.conv3_3,
            'conv4_1': self.conv4_1,
            'conv4_2': self.conv4_2,
            'conv4_3': self.conv4_3,
            'conv5_1': self.conv5_1,
            'conv5_2': self.conv5_2,
            'conv5_3': self.conv5_3,
        }
        
        loaded_count = 0
        for name, layer in layer_map.items():
            if name in self.data_dict:
                try:
                    weights = self.data_dict[name]
                    # VGG16 .npy format: weights[0] = kernel, weights[1] = bias
                    kernel = weights[0]
                    bias = weights[1]
                    layer.set_weights([kernel, bias])
                    loaded_count += 1
                    print(f"   Loaded {name}: kernel shape {kernel.shape}, bias shape {bias.shape}")
                    print(len(weights))
                    for i, w in enumerate(weights):
                        print(i, type(w), getattr(w, 'shape', None))

                except Exception as e:
                    print(f"   Failed to load {name}: {e}")
        
        print(f"\n Successfully loaded {loaded_count}/13 convolutional layers")
        print("Note: FC layers (fc6, fc7) will be trained from scratch\n")


    def get_var_count(self):
        return np.sum([np.prod(v.shape) for v in self.trainable_variables])
