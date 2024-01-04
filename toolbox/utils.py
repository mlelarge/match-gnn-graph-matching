import os
import json
import torch

# create directory if it does not exist
def check_dir(dir_path):
    dir_path = dir_path.replace('//','/')
    os.makedirs(dir_path, exist_ok=True)

def get_device(t):
    if t.is_cuda:
        return t.get_device()
    return 'cpu'

def load_json(json_file):
    # Load the JSON file into a variable
    with open(json_file) as f:
        json_data = json.load(f)

    # Return the data as a dictionary
    return json_data

# Function to search for folders named "checkpoints" recursively and retrieve files
def find_and_retrieve_files(root):
    names = []
    for foldername, subfolders, _ in os.walk(root):
        if 'checkpoints' in subfolders:
            checkpoint_folder = os.path.join(foldername, 'checkpoints/')
            #print(f'Found a "checkpoints" folder: {checkpoint_folder}')
            
            # Retrieve files from the "checkpoints" folder
            for filename in os.listdir(checkpoint_folder):
                file_path = os.path.join(checkpoint_folder, filename)
                if os.path.isfile(file_path):
                    names.append(file_path)
                    #print(f'Retrieving file: {file_path}')
    return names

# Custom sorting key function to extract and compare val_acc values
def sort_by_val_acc(item):
    # Find the position of 'val_acc=' in the string
    val_acc_start = item.find('val_acc=')
    
    if val_acc_start != -1:
        # Extract the val_acc value (assuming it's a float)
        val_acc_str = item[val_acc_start + len('val_acc='):-5]
        try:
            val_acc = float(val_acc_str)
            #print(val_acc)
            return val_acc
        except ValueError:
            # If the 'val_acc' value is not a valid float, return a low value
            return -float('inf')
    else:
        # If 'val_acc=' is not found in the string, return a low value
        return -float('inf')
    
def get_file_creation_date(file_path):
    return os.path.getctime(file_path)

def get_device_config(path):
    config_file = os.path.join(path,'config.json')
    with open(config_file) as json_file:
        config_model = json.load(json_file)
    use_cuda = not config_model['cpu'] and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    return config_model, device