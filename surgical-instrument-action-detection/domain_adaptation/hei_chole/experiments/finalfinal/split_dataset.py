import os
import random
import shutil

def split_dataset(base_path, train_ratio=0.8, val_ratio=0.2, test_ratio=0.3):
    """
    Split the dataset into train, validation, and test sets.
    
    :param base_path: Base path of the dataset
    :param train_ratio: Ratio of training data (within train+val)
    :param val_ratio: Ratio of validation data (within train+val)
    :param test_ratio: Ratio of test data
    """
    # Ensure the target directories exist
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_path, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'labels', split), exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(os.path.join(base_path, 'images')) 
                   if f.endswith('.jpg')]
    
    # Shuffle the files
    random.shuffle(image_files)
    
    # Calculate split sizes
    total_files = len(image_files)
    test_count = int(total_files * test_ratio)
    trainval_count = total_files - test_count
    train_count = int(trainval_count * train_ratio)
    val_count = trainval_count - train_count
    
    # Split the files
    test_files = image_files[:test_count]
    train_files = image_files[test_count:test_count+train_count]
    val_files = image_files[test_count+train_count:]
    
    # Copy files to respective directories
    def copy_files(file_list, split):
        for filename in file_list:
            # Copy image
            shutil.copy(
                os.path.join(base_path, 'images', filename),
                os.path.join(base_path, 'images', split, filename)
            )
            
            # Copy corresponding label
            label_filename = filename.replace('.jpg', '.txt')
            shutil.copy(
                os.path.join(base_path, 'labels', label_filename),
                os.path.join(base_path, 'labels', split, label_filename)
            )
    
    # Perform the copying
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    # Print summary
    print(f"Total files: {total_files}")
    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

# Set the base path
base_path = '/data/Bartscht/m2cai16-tool-locations'

# Run the split
random.seed(42)  # for reproducibility
split_dataset(base_path)