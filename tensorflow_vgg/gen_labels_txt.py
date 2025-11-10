import os

def generate_labels_txt(data_dir, output_file="class_labels.txt"):
    """
    Scans the given folder and writes all subfolder names (class names)
    into a text file, one per line.
    
    Args:
        data_dir (str): Path to the main dataset directory (e.g. 'data/train')
        output_file (str): Path to save the labels text file (default: class_labels.txt)
    """
    # Get only folders (not files)
    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    # Write to file
    with open(output_file, "w") as f:
        for name in class_names:
            f.write(name + "\n")

    print(f" Saved {len(class_names)} class names to '{output_file}'")
    print("Classes:", class_names)


# Example usage:
if __name__ == "__main__":
    # Change this path to your dataset directory
    dataset_path = r"C:\Users\abiga\Cro Disease ID\archive\data"

    generate_labels_txt(dataset_path, "plant_labels.txt")

