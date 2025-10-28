import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import sys
from pathlib import Path

# Add the parent directory to Python path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
from tensorflow_vgg import vgg16, utils

#sys.path.append(str(parent_dir))
#from tensorflow_vgg import utils

img1 = utils.load_image("./test_data/tiger.jpeg") #this is where we need to connect the image from website to

batch1 = img1.reshape((1, 224, 224, 3))

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [1, 224, 224, 3])
        feed_dict = {images: batch1}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images) #will need to change this to build_for_disease_detection(images, 71)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        print(prob)
        utils.print_prob(prob[0], './synset.txt')
