import os
from pyee import EventEmitter

em = EventEmitter()

def create_array(n):
   return [str(i) for i in range(n)]

def searchFolderNames(path):
    all_items = os.listdir(path)
    folder_names = [item for item in all_items if os.path.isdir(os.path.join(path, item))]
    return folder_names

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