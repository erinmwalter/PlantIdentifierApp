import numpy as np
data = np.load('vgg16.npy', allow_pickle=True, encoding='latin1').item()
print(data.keys())

