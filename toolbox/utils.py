import os

# create directory if it does not exist
def check_dir(dir_path):
    dir_path = dir_path.replace('//','/')
    os.makedirs(dir_path, exist_ok=True)

def get_device(t):
    if t.is_cuda:
        return t.get_device()
    return 'cpu'
