import sys
import json
with open("./config.json") as json_file:
    _CONFIG = json.load(json_file)
sys.path.insert(0,_CONFIG['raug_full_path'])

_DATASET_PATH = _CONFIG['dataset_full_path']
_SAVE_FOLDER_PATH = _CONFIG['save_folder_full_path']

from prep_dataset import get_dataloaders
from torchvision import models
import torch.optim as optim
import torch.nn as nn
from raug.train import fit_model
from raug.eval import test_model
from resnet import MyResnet
import os

train_dataloader, test_dataloader, val_dataloader = get_dataloaders(_DATASET_PATH)

resnet = models.resnet50()

model = MyResnet(resnet, 2)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

fit_model(model, train_dataloader, val_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=30, epochs_early_stop=5,
        save_folder=_SAVE_FOLDER_PATH, initial_model=None, device=None, config_bot=None, model_name="CNN", resume_train=False,
        history_plot=True, val_metrics=["balanced_accuracy"])

_metric_options = {
        'save_all_path': os.path.join(_SAVE_FOLDER_PATH, "test_pred"),
        'pred_name_scores': 'predictions.csv',
}

_class_names = ['boa', 'ruim']

test_model(model=model, data_loader=test_dataloader, class_names=_class_names, metrics_options=_metric_options, metrics_to_comp='all', save_pred=True)