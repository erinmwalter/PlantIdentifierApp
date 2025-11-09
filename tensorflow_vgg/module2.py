import os
import shutil
import random

def move_sample_photos(parent_folder, num_photos=50):
    # Define the output folder (next to the parent folder)
    parent_dir = os.path.dirname(parent_folder.rstrip("/\\"))
    output_folder = os.path.join(parent_dir, os.path.basename(parent_folder) + "_sampled")

    os.makedirs(output_folder, exist_ok=True)

    # Loop through each subfolder
    for root, dirs, files in os.walk(parent_folder):
        # Ignore the parent folder itself, handle only subfolders
        if root == parent_folder:
            continue

        # Filter image files (you can add more extensions if needed)
        images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        if not images:
            continue

        # Randomly select up to `num_photos` images
        selected = random.sample(images, min(num_photos, len(images)))

        # Build output subfolder path
        rel_path = os.path.relpath(root, parent_folder)
        out_subfolder = os.path.join(output_folder, rel_path)
        os.makedirs(out_subfolder, exist_ok=True)

        # Copy files
        for img in selected:
            src = os.path.join(root, img)
            dst = os.path.join(out_subfolder, img)
            shutil.move(src, dst)

        print(f"Copied {len(selected)} photos from '{root}' to '{out_subfolder}'")

    print("\nDone! Sampled photos saved to:", output_folder)

# ======= Example usage =======
# Replace with your actual main folder path, e.g.:
# main_folder = r"C:\Users\YourName\Pictures\MainPhotos"
main_folder = r"C:\Users\abiga\Cro Disease ID\archive\data"
move_sample_photos(main_folder)
