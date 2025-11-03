import os


import numpy as np
import tensorflow as tf
print(tf.__version__)
import vgg19_trainable
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
    label_list = []
    
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
                label_list.append(idx)
                count += 1
        
        print(f"{count} images")
    
    return file_paths, label_list

def load_batch(file_paths, label_list, indices, num_classes):
    """Load only the images needed for one batch"""
    batch_x = []
    batch_y = []
    
    # Convert indices to plain Python list
    indices = [int(i) for i in indices]
    
    for idx in indices:
        try:
            img = utils.load_image(file_paths[idx])
            img = img.astype(np.float32)  # Use float32 to save memory
            batch_x.append(img)
            
            # Get label - be very defensive about type conversion
            label_value = label_list[idx]
            
            # Check if it's already a Python int
            if isinstance(label_value, int):
                batch_y.append(label_value)
            # Check if it's a numpy int
            elif isinstance(label_value, (np.integer, np.int32, np.int64)):
                batch_y.append(int(label_value))
            # Check if it has a numpy() method (TensorFlow tensor)
            elif hasattr(label_value, 'numpy'):
                batch_y.append(int(label_value.numpy()))
            # Try direct conversion
            else:
                batch_y.append(int(label_value))
            
        except Exception as e:
            # Skip corrupted images 
            print(f"\nSkipped an image\n")
            continue
    
    if len(batch_x) == 0:
        return None, None
    
    batch_x = np.array(batch_x, dtype=np.float32)
    
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
    file_paths, label_list = create_file_list(DATA_DIR, classes, MAX_IMAGES_PER_CLASS)
    
    # Ensure label_list is a regular Python list, not a tensor
    label_list = list(label_list)
    
    print(f"\nTotal images indexed: {len(file_paths)}")
    print(f"Label list type: {type(label_list)}")  # Debug
    
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
    # Replace placeholders
    images = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="images")
    labels = tf.keras.Input(shape=(num_classes,), name="labels")
    #train_mode = tf.placeholder(tf.bool)
    
    vgg = vgg19_trainable.Vgg16(vgg16_npy_path=r"tensorflow_vgg\vgg16.npy")
    vgg.build(images, training =K.learning_phase())
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=vgg.prob, labels=labels))
    #train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    # Only train the fully connected layers
    train_vars = [v for v in tf.trainable_variables() if 'fc' in v.name]
    print("\nTraining the following layers only:")
    for v in train_vars:
        print("  ", v.name)
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, var_list=train_vars)

    
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
            batch_x, batch_y = load_batch(file_paths, label_list, batch_idx, num_classes)
            
            # Skip empty batches (all images failed to load)
            if batch_x is None or len(batch_x) == 0:
                print(f"  Warning: Skipping empty batch at step {i//BATCH_SIZE}")
                continue
            
            _, batch_loss = sess.run([train_op, loss], 
                                    feed_dict={images: batch_x, 
                                             labels: batch_y, 
                                             })
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
            batch_x, batch_y = load_batch(file_paths, label_list, batch_idx, num_classes)
            
            if batch_x is None:
                continue
            
            v_acc, v_loss = sess.run([acc, loss],
                                    feed_dict={images: batch_x, 
                                             labels: batch_y, 
                                             }) #took out train_mode here
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
