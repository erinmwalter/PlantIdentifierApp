import numpy as np
import tensorflow as tf

import sys
from pathlib import Path

# Add the parent directory to Python path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
from tensorflow_vgg import vgg16_tf2, utils

#sys.path.append(str(parent_dir))
#from tensorflow_vgg import utils

img1 = utils.load_image("./test_data/tiger.jpeg") #this is where we need to connect the image from website to

batch1 = img1.reshape((1, 224, 224, 3))

# Optional: GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
# Run VGG16 on CPU
with tf.device('/CPU:0'):
    vgg = vgg16_tf2.Vgg16()
    # Forward pass
    vgg(batch1)  # runs the model eagerly
    prob = vgg(batch1).numpy()

# Print probabilities
print(prob)
utils.print_prob(prob[0], './synset.txt')
