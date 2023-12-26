import os
import glob
import json
import re
from pyee import EventEmitter
# from super_gradients.training import models

em = EventEmitter()

def create_array(n):
   return [str(i) for i in range(n)]

def searchFolderNames(path):
    all_items = os.listdir(path)
    folder_names = [item for item in all_items if os.path.isdir(os.path.join(path, item))]
    return folder_names


def getModelClasses(checkpoints_dir, model_name, name):
    experiment_logs_files = []
    files = glob.glob(f'{checkpoints_dir}/{model_name}/{name}/*')
    for file in files:
        if file.startswith(f'{checkpoints_dir}/{model_name}/{name}\\experiment_logs'):
            experiment_logs_files.append(file)
    if len(experiment_logs_files) > 0:
        experiment_logs_file = experiment_logs_files[0]
        with open(experiment_logs_file, 'r') as f:
            lines = f.readlines()
        lines.pop(0)
        lines.pop(-1)
        json_string = ''.join(lines)
        data = json.loads(json_string)
        train_dataset_params = data['dataset_params']['valid_dataset_params'].replace("'", '"')
        # train_dataset_params = re.sub(r'\n', '', train_dataset_params)
        # train_dataset_params = json.loads(train_dataset_params)
        train_dataset_params = json.dumps(train_dataset_params, default=default)
        print(train_dataset_params)
    else:
        return []

def getModel(checkpoints_dir, model_name, name, num_classes, device):
    try:
        predict_model = models.get(
            model_name=model_name,
            checkpoint_path=f'{checkpoints_dir}/{model_name}/{name}/ckpt_best.pth',
            num_classes=num_classes
        ).to(device) if name is not None else None
        return predict_model
    except:
        return None


def delete_prev_dataset(dataset_dir):
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            os.remove(os.path.join(root, file))

def generate_dataset_folder(dataset_name):
    if not os.path.exists(f'{dataset_name}/samples'):
        os.makedirs(f'{dataset_name}/samples')
    make_folder(dataset_name, 'train')
    make_folder(dataset_name, 'valid')
    make_folder(dataset_name, 'test')

def make_folder(dataset_name, folder_name):
    if not os.path.exists(f'{dataset_name}/{folder_name}'):
        os.makedirs(f'{dataset_name}/{folder_name}')
        os.makedirs(f'{dataset_name}/{folder_name}/images')
        os.makedirs(f'{dataset_name}/{folder_name}/labels')
        print(f'{folder_name} 폴더 생성 완료')

def get_event_bus():
    return em