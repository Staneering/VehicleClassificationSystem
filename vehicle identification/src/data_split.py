import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def stratified_split(annotations):
    labels = [label for _, label in annotations]
    train_data, valtest_data = train_test_split(annotations, test_size=0.3, stratify=labels, random_state=42)
    val_data, test_data = train_test_split(valtest_data, test_size=1/3, stratify=[label for _, label in valtest_data], random_state=42)
    return train_data, val_data, test_data

def save_images_to_split_folder(split_data, split_name, output_dir):
    for img_path, label in split_data:
        save_dir = Path(output_dir) / split_name / str(label)
        save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, save_dir / Path(img_path).name)
