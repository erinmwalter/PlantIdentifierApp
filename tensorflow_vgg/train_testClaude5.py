import os
import numpy as np
import tensorflow as tf

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

# Limit images per class to avoid memory issues
MAX_IMAGES_PER_CLASS = 500

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
            img = img.astype(np.float32)
            batch_x.append(img)
            
            label_value = label_list[idx]
            if isinstance(label_value, int):
                batch_y.append(label_value)
            elif isinstance(label_value, (np.integer, np.int32, np.int64)):
                batch_y.append(int(label_value))
            elif hasattr(label_value, 'numpy'):
                batch_y.append(int(label_value.numpy()))
            else:
                batch_y.append(int(label_value))
            
        except Exception as e:
            print(f"\nSkipped an image: {e}\n")
            continue
    
    if len(batch_x) == 0:
        return None, None
    
    batch_x = np.array(batch_x, dtype=np.float32)
    
    # Create one-hot encoding
    batch_y_onehot = np.zeros((len(batch_y), num_classes), dtype=np.float32)
    for i, label_idx in enumerate(batch_y):
        batch_y_onehot[i, label_idx] = 1.0
    
    return batch_x, batch_y_onehot

# ==========================================
# TF2 TRAINING FUNCTIONS
# ==========================================
def main():
    # Step 1: Discover classes
    classes = sorted([d for d in os.listdir(DATA_DIR) 
                     if os.path.isdir(os.path.join(DATA_DIR, d))])
    num_classes = len(classes)
    print(f"\nFound {num_classes} classes: {classes}\n")
    
    # Step 2: Create file lists
    print("Creating file index...")
    file_paths, label_list = create_file_list(DATA_DIR, classes, MAX_IMAGES_PER_CLASS)
    label_list = list(label_list)
    
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
    
    # Step 4: Build model
    print("\nBuilding model...")
    model = vgg19_trainable.Vgg16(vgg16_npy_path=None)
    
    # Build the model by calling it once (initialize layers)
    dummy_input = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
    _ = model(dummy_input, training=False)

    # NOW load the pre-trained weights
    #print("Loading pre-trained weights...")
    #model.load_weights_from_npy(r"tensorflow_vgg\vgg16.npy")
    #for batch in train_dataset:
        #model.train_step(batch)
    # Only train FC layers
    trainable_vars = [v for v in model.trainable_variables if 'fc' in v.name]
    print("\nTraining the following layers only:")
    for v in trainable_vars:
        print("  ", v.name)
    
    # Step 5: Setup optimizer and loss
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    
    # Metrics
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
    
    # Step 6: Define training and validation steps
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)  # Dropout ON
            loss = loss_fn(y, predictions)
        
        # Only compute gradients for FC layers
        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        train_loss_metric.update_state(loss)
        return loss
    
    @tf.function
    def val_step(x, y):
        predictions = model(x, training=False)  # Dropout OFF
        loss = loss_fn(y, predictions)
        
        val_loss_metric.update_state(loss)
        val_acc_metric.update_state(y, predictions)
        return loss
    
    # Step 7: Training loop
    steps_per_epoch = len(train_indices) // BATCH_SIZE
    print(f"\nStarting training ({steps_per_epoch} steps per epoch)...\n")
    
    for epoch in range(EPOCHS):
        # Reset metrics
        train_loss_metric.reset_state()
        val_loss_metric.reset_state()
        val_acc_metric.reset_state()
        
        # Shuffle training data each epoch
        np.random.shuffle(train_indices)
        
        # Training
        num_batches = 0
        for i in range(0, len(train_indices), BATCH_SIZE):
            batch_idx = train_indices[i:i+BATCH_SIZE]
            
            # Load batch
            batch_x, batch_y = load_batch(file_paths, label_list, batch_idx, num_classes)
            
            if batch_x is None or len(batch_x) == 0:
                print(f"  Warning: Skipping empty batch at step {i//BATCH_SIZE}")
                continue
            
            # Train step
            loss = train_step(batch_x, batch_y)
            num_batches += 1
            
            # Print progress every 50 batches
            if num_batches % 50 == 0:
                print(f"  Epoch {epoch+1} - Batch {num_batches}/{steps_per_epoch} - Loss: {loss.numpy():.4f}")
        
        # Validation
        val_batches = 0
        for i in range(0, min(len(val_indices), 5 * BATCH_SIZE), BATCH_SIZE):
            batch_idx = val_indices[i:i+BATCH_SIZE]
            batch_x, batch_y = load_batch(file_paths, label_list, batch_idx, num_classes)
            
            if batch_x is None:
                continue
            
            val_step(batch_x, batch_y)
            val_batches += 1
        
        # Print epoch results
        print(f"\n Epoch {epoch+1}/{EPOCHS} Complete:")
        print(f"  Train Loss: {train_loss_metric.result().numpy():.4f}")
        print(f"  Val Loss: {val_loss_metric.result().numpy():.4f}")
        print(f"  Val Acc: {val_acc_metric.result().numpy():.4f}\n")
    
    # Step 8: Save model
    print("Saving model...")
    model.save_npy(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    print("\n Training complete!")

if __name__ == "__main__":
    main()
