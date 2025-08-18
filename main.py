import sys
import json
with open("./config.json") as json_file:
    _CONFIG = json.load(json_file)
sys.path.insert(0,_CONFIG['raug_full_path'])

_DATASET_PATH = _CONFIG['dataset_full_path']
_SAVE_FOLDER_PATH = _CONFIG['save_folder_full_path']

from preprocess.prep_dataset import get_dataloaders
import torch.optim as optim
import torch.nn as nn
from raug.train import fit_model
from raug.eval import test_model
import os
from sacred import Experiment
import time
from models.models import set_model
from posprocess.agg_metrics import agg_metrics_all_folders

ex = Experiment()

@ex.config
def my_config():
	_model = 'resnet' # 'resnet' ou 'mobilenet' ou 'vggnet'
	_epochs = 30
	_epochs_early_stop = 7
	_optimizer = 'adam' # 'adam' ou 'sgd'
	_lr_init = 0.00001
	_num_folders = 5
	_batch_size = 30

@ex.automain
def main(_model, _epochs, _epochs_early_stop, _optimizer, _lr_init, _num_folders, _batch_size):
    
	dir_ex = f"{_model}_{str(time.time()).replace('.', '')}"
	os.mkdir(_SAVE_FOLDER_PATH + '/' + dir_ex)
	_save_folder = os.path.join(_SAVE_FOLDER_PATH, dir_ex)

	train_dataloader_list, val_dataloader_list, _class_names = get_dataloaders(_DATASET_PATH, num_folders=_num_folders, batch_size=_batch_size)

	loss_fn = nn.CrossEntropyLoss()
	# weight=torch.tensor([2, 1])

	_metric_options = {
		'save_all_path': os.path.join(_save_folder, "test_pred"),
		'pred_name_scores': 'predictions.csv',
	}

	for i, (train_dataloader), in enumerate(train_dataloader_list):

		model = set_model(_model)
            
		if _optimizer == 'adam':
			optimizer = optim.Adam(model.parameters(), lr=_lr_init)
		else:
			optimizer = optim.SGD(model.parameters(), lr=_lr_init)

		_save_folder_k = os.path.join(_save_folder, f"folder_{i+1}")

		_metric_options = {
			'save_all_path': os.path.join(_save_folder_k, "test_pred"),
			'pred_name_scores': 'predictions.csv',
		}

		print(f"\n\n------------- STARTING TRAIN FOLDER {i+1}---------------\n\n")

		fit_model(model, train_dataloader, val_dataloader_list[i], optimizer=optimizer, loss_fn=loss_fn, epochs=_epochs, epochs_early_stop=_epochs_early_stop,
				save_folder=_save_folder_k, initial_model=None, device=None, config_bot=None, model_name="CNN", resume_train=False,
				history_plot=True, val_metrics=["balanced_accuracy"])
		
		_checkpoint_best = os.path.join(_save_folder_k, "best-checkpoint/best-checkpoint.pth")

		test_model(model=model, data_loader=val_dataloader_list[i], class_names=_class_names, metrics_options=_metric_options, metrics_to_comp='all', save_pred=True, checkpoint_path=_checkpoint_best)



	agg_metrics_all_folders(ex_path=_save_folder, num_folders=_num_folders, class_names=_class_names)