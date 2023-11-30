import os
import json

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