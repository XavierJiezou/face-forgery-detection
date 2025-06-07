from glob import glob
import os
from typing import List

def get_fake_paths(root_path="data")->List[str]:
    """
    Get paths to all fake images in the specified root directory.
    
    Args:
        root_path (str): The root directory where fake images are stored.
        
    Returns:
        list: A list of paths to all fake images.
    """
    fake_paths = glob(os.path.join(root_path, "fake", "*", "512","face"))
    return fake_paths

def get_real_paths(root_path="data")->List[str]:
    """
    Get paths to all real images in the specified root directory.
    
    Args:
        root_path (str): The root directory where real images are stored.
        
    Returns:
        list: A list of paths to all real images.
    """
    real_paths = glob(os.path.join(root_path, "real", "*","face"))
    return real_paths


def check_folders(folders):
    for folder in folders:
        images = os.listdir(folder)
        assert len(images) > 0, f"{folder} has zero image. Please check the folder."

if __name__ == "__main__":
    fake_folders = get_fake_paths()
    print(fake_folders[0])
    check_folders(fake_folders)

    real_folders = get_real_paths()
    print(real_folders[0])

    check_folders(real_folders)
