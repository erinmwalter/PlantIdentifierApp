import os
import numpy as np
import tensorflow as tf
print(tf.__file__)
from vgg19_trainable import Vgg16  # your TF2.x model
import utils

# ==========================================
# CONFIG
# ==========================================
DATA_DIR = r"C:\Users\abiga\Cro Disease ID\archive\data"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = r"C:\Users\abiga\Cro Disease ID\PlantIdentiferApp\tensorflow_vgg\vgg16_plant_classifier_tf2"

MAX_IMAGES_PER_CLASS = 500

# ==========================================
# DATA GENERATOR
# ==========================================
def create_file_list(data_dir, classes, max_per_class=None):
    file_paths, labels = [], []
    for idx, cls in enumerate(classes):
        class_path = os.path.join(data_dir, cls)
        count = 0
        for fname in os.listdir(class_path):
            if max_per_class and count >= max_per_class:
                break
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(class_path, fname))
                labels.append(idx)
                count += 1
    return file_paths, labels

def load_batch(file_paths, label_list, indices, num_classes):
    batch_x, batch_y = [], []
    indices = [int(i) for i in indices]
    for idx in indices:
        try:
            img = utils.load_image(file_paths[idx])
            img = img.astype(np.float32)
            batch_x.append(img)
            batch_y.append(int(label_list[idx]))
        except:
            continue
    if len(batch_x) == 0:
        return None, None
    batch_x = np.array(batch_x)
    batch_y_onehot = np.zeros((len(batch_y), num_classes), dtype=np.float32)
    for i, label in enumerate(batch_y):
        batch_y_onehot[i, label] = 1.0
    return batch_x, batch_y_onehot

# ==========================================
# MAIN
# ==========================================
def main():
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")
    
    file_paths, label_list = create_file_list(DATA_DIR, classes, MAX_IMAGES_PER_CLASS)
    label_list = list(label_list)
    
    indices = np.arange(len(file_paths))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices, val_indices = indices[:split], indices[split:]
    
    # Build model
    model = Vgg16(vgg16_npy_path=None, num_classes=num_classes, dropout=0.5)

    # Only train FC layers
    for layer in model.layers:
        if 'fc' not in layer.name:
            layer.trainable = False
    
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    # Training loop
    for epoch in range(EPOCHS):
        np.random.shuffle(train_indices)
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_indices), BATCH_SIZE):
            batch_idx = train_indices[i:i+BATCH_SIZE]
            batch_x, batch_y = load_batch(file_paths, label_list, batch_idx, num_classes)
            if batch_x is None:
                continue
            
            with tf.GradientTape() as tape:
                logits = model(batch_x, training=True)
                loss_value = loss_fn(batch_y, logits)
            
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            epoch_loss += loss_value.numpy()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        # Validation
        val_acc_total = 0
        val_batches = 0
        for i in range(0, len(val_indices), BATCH_SIZE):
            batch_idx = val_indices[i:i+BATCH_SIZE]
            batch_x, batch_y = load_batch(file_paths, label_list, batch_idx, num_classes)
            if batch_x is None:
                continue
            logits = model(batch_x, training=False)
            acc = np.mean(np.argmax(logits.numpy(), axis=1) == np.argmax(batch_y, axis=1))
            val_acc_total += acc
            val_batches += 1
        
        val_acc_avg = val_acc_total / val_batches if val_batches > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Acc: {val_acc_avg:.4f}")
    
    # Save model weights
    model.save_weights(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()

