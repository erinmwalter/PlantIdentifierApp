import urllib.request
import os

url = "https://github.com/machrisaa/tensorflow-vgg/releases/download/v1.0/vgg16.npy"
output_path = "vgg16.npy"

print("Downloading VGG16 weights...")
urllib.request.urlretrieve(url, output_path)
print(f"Downloaded to {output_path}")
