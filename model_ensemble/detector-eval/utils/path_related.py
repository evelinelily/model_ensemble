
from pathlib import Path
import os

def generate_path_dict(data_dir, extension='jpg'):
    """
    Generate a dictionary, where key: file name, and value: file path
    Assuming all file names are unique in data_dir
    param:
        data_dir: directory where you want to scan the file, recursively
        extension: the file extension. this param specifies the file format
    return:
        a dictionary, where key: file name, and value: file path
    """
    assert os.path.exists(data_dir), data_dir+" does not exist"
    extension = extension.strip('.')
    files = Path(data_dir).glob('**/*.{}'.format(extension))
    files = [str(x) for x in files]
    path_dict = dict()
    for file_path in files:
        file_name = os.path.basename(file_path)
        path_dict[file_name] = file_path
    return path_dict