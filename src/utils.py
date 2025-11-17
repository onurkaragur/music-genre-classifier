import os 
import shutil
import random
from pathlib import Path

#Split the GTZAN dataset by copying audio files into train/val/test folders.
def split_dataset(
        source_dir,
        output_dir,
        train_ratio = 0.7,
        val_ratio = 0.15,
        test_ratio = 0.15,
        seed = 42):
    
    random.seed(seed)

    # Convert to absolute paths if relative
    source_dir = os.path.abspath(source_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Validate source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    #Create Output Folders
    for split in ["train","val","test"]:
        split_path = os.path.join(output_dir, split)
        os.makedirs(split_path, exist_ok=True)

    genres = os.listdir(source_dir)

    for genre in genres:
        genre_path = os.path.join(source_dir, genre)
        if not os.path.isdir(genre_path):
            continue

        files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_files = files[:n_train]
        val_files = files[n_train: n_train + n_val]
        test_files = files[n_train + n_val:]

        #Create genre subfolders in output directories
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(output_dir, split, genre), exist_ok=True)

        #Copy audio files
        for f in train_files:
            shutil.copy(
                os.path.join(genre_path, f),
                os.path.join(output_dir, "train", genre, f)
            )
        
        for f in val_files:
            shutil.copy(
                os.path.join(genre_path, f),
                os.path.join(output_dir, "val", genre, f)
            )

        for f in test_files:
            shutil.copy(
                os.path.join(genre_path, f),
                os.path.join(output_dir, "test", genre, f)
            )
                
        print(f"[OK] {genre}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    source = project_root / "data" / "dataset"
    output = project_root / "data"
    
    split_dataset(str(source), str(output))    