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

    train_imgs_paths, eval_imgs_paths, train_labels, eval_labels = train_test_split(imgs_paths, labels, test_size=0.3, random_state=10)

    test_imgs_paths, val_imgs_paths, test_labels, val_labels = train_test_split(eval_imgs_paths, eval_labels, test_size=0.5, random_state=10)

    print(f'Treino - imgs:{len(train_imgs_paths)}, labels:{len(train_labels)}')
    print(f'Teste - imgs:{len(test_imgs_paths)}, labels:{len(test_labels)}')
    print(f'Validacao - imgs:{len(val_imgs_paths)}, labels:{len(val_labels)}')

    # train_transform = ImgTrainTransform()
    eval_transform = ImgEvalTransform()

    train_dataloader = get_data_loader(train_imgs_paths, train_labels, transform=eval_transform)
    test_dataloader = get_data_loader(test_imgs_paths, test_labels, transform=eval_transform)
    val_dataloader = get_data_loader(val_imgs_paths, val_labels, transform=eval_transform)

    return train_dataloader, test_dataloader, val_dataloader

