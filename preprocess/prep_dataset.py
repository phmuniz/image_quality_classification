import sys
import json
with open("./config.json") as json_file:
    _CONFIG = json.load(json_file)
sys.path.insert(0,_CONFIG['raug_full_path'])

from glob import glob
import os
from raug.loader import get_data_loader
from aug import ImgEvalTransform, ImgTrainTransform
from sklearn.model_selection import StratifiedKFold
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

def get_dataloaders(dataset_path, num_folders=5, batch_size=30):

    """
    :param dataset_path (string): caminho para a pasta que contém o dataset.
    :param num_folder (int): número de folders para validação cruzada. Default 5.
    :param batch_size (int): batch size para o dataset. Default 30.
    :return (tuple): uma tupla contendo a lista de dataloaders do treino, a lista de dataloaders da validação e o nome dos labels.
    """

    labels_name = glob(os.path.join(dataset_path, "*"))
    labels_name = [l.split(os.path.sep)[-1] for l in labels_name]

    print(f"Classes: {labels_name}\n\n")

    imgs_paths, labels = _get_paths_and_labels(dataset_path, labels_name)

    skf = StratifiedKFold(n_splits=num_folders, shuffle=True, random_state=10)

    train_dataloader_list = []
    val_dataloader_list = []

    imgs_paths = np.array(imgs_paths)
    labels = np.array(labels)

    train_transform = ImgTrainTransform()
    eval_transform = ImgEvalTransform()

    for i, (train_index, val_index) in enumerate(skf.split(imgs_paths, labels)):

        train_imgs_paths, val_imgs_paths = imgs_paths[train_index], imgs_paths[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        train_dataloader = get_data_loader(train_imgs_paths, train_labels, transform=train_transform, batch_size=batch_size)
        val_dataloader = get_data_loader(val_imgs_paths, val_labels, transform=eval_transform, batch_size=batch_size)

        train_dataloader_list.append(train_dataloader)
        val_dataloader_list.append(val_dataloader)

        print(f"Fold {i+1}:")
        print(f'Train: {len(train_imgs_paths)}, {len(train_labels)}')
        print(f'Validacao: {len(val_imgs_paths)}, {len(val_labels)}')

    return train_dataloader_list, val_dataloader_list, labels_name

