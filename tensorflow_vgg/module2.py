import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# List available physical devices
print("Available CPUs:", tf.config.list_physical_devices('CPU'))
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Available XPU (Intel GPUs):", tf.config.list_physical_devices('XPU'))
