import sys
import json
with open("./config.json") as json_file:
    _CONFIG = json.load(json_file)
sys.path.insert(0,_CONFIG['raug_full_path'])

from glob import glob
import os
from sklearn.model_selection import train_test_split
from raug.loader import get_data_loader
from aug import ImgEvalTransform
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

def _get_paths_and_labels(path, lab_names):
    imgs_path, labels = list(), list()
    lab_cnt = 0
    for lab in lab_names:    
        paths_aux = glob(os.path.join(path, lab, "*.png"))
        
        # Atualizando os labels e imgs_paths
        labels += [lab_cnt] * len(paths_aux)
        imgs_path += paths_aux
        
        lab_cnt += 1
        
    return imgs_path, labels

def get_dataloaders(dataset_path):


    labels_name = glob(os.path.join(dataset_path, "*"))
    labels_name = [l.split(os.path.sep)[-1] for l in labels_name]

    imgs_paths, labels = _get_paths_and_labels(dataset_path, labels_name)

    temp_imgs_paths, test_imgs_paths, temp_labels, test_labels = train_test_split(imgs_paths, labels, test_size=0.2, random_state=10, shuffle=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

    train_dataloader_list = []
    val_dataloader_list = []

    temp_imgs_paths = np.array(temp_imgs_paths)
    temp_labels = np.array(temp_labels)

    eval_transform = ImgEvalTransform()

    for i, (train_index, val_index) in enumerate(skf.split(temp_imgs_paths, temp_labels)):

        train_imgs_paths, val_imgs_paths = temp_imgs_paths[train_index], temp_imgs_paths[val_index]
        train_labels, val_labels = temp_labels[train_index], temp_labels[val_index]

        train_dataloader = get_data_loader(train_imgs_paths, train_labels, transform=eval_transform)
        val_dataloader = get_data_loader(val_imgs_paths, val_labels, transform=eval_transform)

        train_dataloader_list.append(train_dataloader)
        val_dataloader_list.append(val_dataloader)

        print(f"Fold {i+1}:")
        print(f'Train: {len(train_imgs_paths)}, {len(train_labels)}')
        print(f'Validacao: {len(val_imgs_paths)}, {len(val_labels)}')

    print(f'Teste: {len(test_imgs_paths)}, {len(test_labels)}')

    test_dataloader = get_data_loader(test_imgs_paths, test_labels, transform=eval_transform)

    return train_dataloader_list, test_dataloader, val_dataloader_list

