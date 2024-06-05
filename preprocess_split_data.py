import os
import shutil
import random

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_dataset(root_dir, output_dir, train_ratio=0.8):
    # Define the classes
    classes = ['OSCC', 'with_dysplasia', 'without_dysplasia']
    
    # Create train and test directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    
    create_dir_if_not_exists(train_dir)
    create_dir_if_not_exists(test_dir)
    
    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        images = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        
        random.shuffle(images)
        
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Create class-specific directories in train and test directories
        train_class_dir = os.path.join(train_dir, cls)
        test_class_dir = os.path.join(test_dir, cls)
        
        create_dir_if_not_exists(train_class_dir)
        create_dir_if_not_exists(test_class_dir)
        
        # Move the images
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_class_dir, img)
            shutil.move(src, dst)
        
        for img in test_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(test_class_dir, img)
            shutil.move(src, dst)
        
        print(f"{cls}: {len(train_images)} images moved to train, {len(test_images)} images moved to test")

if __name__ == "__main__":
    root_dir = "/home/coe_iot_ai/Desktop/Amit/medHisPath/images/train"
    output_dir = "/home/coe_iot_ai/Desktop/Amit/medHisPath/images/split_data"
    split_dataset(root_dir, output_dir, train_ratio=0.8)
