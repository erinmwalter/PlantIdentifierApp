import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import vgg16_trainable
import utils

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = r"C:\Users\abiga\Cro Disease ID\archive\data"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = r"C:\Users\abiga\Cro Disease ID\PlantIdentiferApp\tensorflow_vgg\vgg16_plant_classifier.npy"

# Limit images per class to avoid memory issues (remove this line to use all images)
MAX_IMAGES_PER_CLASS = 500  # Adjust based on your RAM

# ==========================================
# DATA GENERATOR - LOADS BATCHES ON THE FLY
# ==========================================
def create_file_list(data_dir, classes, max_per_class=None):
    """Create list of file paths and labels WITHOUT loading images into memory"""
    file_paths = []
    labels = []
    
    for idx, cls in enumerate(classes):
        class_path = os.path.join(data_dir, cls)
        print(f"Scanning class '{cls}'...", end=" ")
        
        count = 0
        for fname in os.listdir(class_path):
            if max_per_class and count >= max_per_class:
                break
                
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, fname)
                file_paths.append(img_path)
                labels.append(idx)
                count += 1
        
        print(f"{count} images")
    
    return file_paths, labels

def load_batch(file_paths, label_list, indices, num_classes):
    """Load only the images needed for one batch"""
    batch_x = []
    batch_y = []
    
    for idx in indices:
        try:
            img = utils.load_image(file_paths[idx])
            img = img.astype(np.float32)  # Use float32 to save memory
            batch_x.append(img)
            #batch_y.append(label_list[idx])
            # Get label as plain Python int
            label_value = label_list[idx]
            if hasattr(label_value, 'numpy'):  # If it's a tensor
                label_value = label_value.numpy()
            label_value = int(label_value)  # Force to Python int
            batch_y.append(label_value)
        except Exception as e:
            # Skip corrupted images
            print(f"Warning: Could not load {file_paths[idx]}: {e}")
            continue
    
    if len(batch_x) == 0:
        return None, None
    
    batch_x = np.array(batch_x, dtype=np.float32)
    # Convert batch_y to numpy array first, then use for one-hot encoding
    #batch_y_indices = np.array(batch_y, dtype=np.int32)
    #batch_y = np.eye(num_classes, dtype=np.float32)[batch_y_indices]

    # Manually create one-hot encoding to avoid any tensor issues
    batch_y_onehot = np.zeros((len(batch_y), num_classes), dtype=np.float32)
    for i, label_idx in enumerate(batch_y):
        batch_y_onehot[i, label_idx] = 1.0
    
    
    return batch_x, batch_y_onehot


# ==========================================
# MAIN
# ==========================================
def main():
    # Step 1: Discover classes
    classes = sorted([d for d in os.listdir(DATA_DIR) 
                     if os.path.isdir(os.path.join(DATA_DIR, d))])
    num_classes = len(classes)
    print(f"\nFound {num_classes} classes: {classes}\n")
    
    # Step 2: Create file lists (NOT loading images yet!)
    print("Creating file index...")
    file_paths, labels = create_file_list(DATA_DIR, classes, MAX_IMAGES_PER_CLASS)
    print(f"\nTotal images indexed: {len(file_paths)}")
    
    if len(file_paths) < 10:
        raise ValueError("Too few images found! Check your DATA_DIR path.")
    
    # Step 3: Split train/val
    indices = np.arange(len(file_paths))
    np.random.shuffle(indices)
    
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Step 4: Build TensorFlow graph
    print("\nBuilding model...")
    images = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
    labels = tf.placeholder(tf.float32, [None, num_classes])
    train_mode = tf.placeholder(tf.bool)
    
    vgg = vgg16_trainable.Vgg16()
    vgg.build(images, train_mode)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=vgg.prob, labels=labels))
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    
    correct = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(labels, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    # Step 5: Training loop
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    steps_per_epoch = len(train_indices) // BATCH_SIZE
    
    print(f"\nStarting training ({steps_per_epoch} steps per epoch)...\n")
    
    for epoch in range(EPOCHS):
        # Shuffle training data each epoch
        np.random.shuffle(train_indices)
        
        epoch_loss = 0
        num_batches = 0
        
        # Train on batches
        for i in range(0, len(train_indices), BATCH_SIZE):
            batch_idx = train_indices[i:i+BATCH_SIZE]
            
            # Load this batch into memory
            batch_x, batch_y = load_batch(file_paths, labels, batch_idx, num_classes)
            
            if batch_x is None:
                continue
            
            _, batch_loss = sess.run([train_op, loss], 
                                    feed_dict={images: batch_x, 
                                             labels: batch_y, 
                                             train_mode: True})
            epoch_loss += batch_loss
            num_batches += 1
            
            # Print progress every 50 batches
            if num_batches % 50 == 0:
                print(f"  Epoch {epoch+1} - Batch {num_batches}/{steps_per_epoch} - Loss: {batch_loss:.4f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        # Validation (use multiple batches for better estimate)
        val_acc_total = 0
        val_loss_total = 0
        val_batches = 0
        
        for i in range(0, min(len(val_indices), 5 * BATCH_SIZE), BATCH_SIZE):
            batch_idx = val_indices[i:i+BATCH_SIZE]
            batch_x, batch_y = load_batch(file_paths, labels, batch_idx, num_classes)
            
            if batch_x is None:
                continue
            
            v_acc, v_loss = sess.run([acc, loss],
                                    feed_dict={images: batch_x, 
                                             labels: batch_y, 
                                             train_mode: False})
            val_acc_total += v_acc
            val_loss_total += v_loss
            val_batches += 1
        
        val_acc_avg = val_acc_total / val_batches if val_batches > 0 else 0
        val_loss_avg = val_loss_total / val_batches if val_batches > 0 else 0
        
        print(f"\n Epoch {epoch+1}/{EPOCHS} Complete:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Acc: {val_acc_avg:.4f} | Val Loss: {val_loss_avg:.4f}\n")
    
    # Step 6: Save model
    print("Saving model...")
    vgg.save_npy(sess, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    sess.close()
    print("\n Training complete!")

if __name__ == "__main__":
    main()