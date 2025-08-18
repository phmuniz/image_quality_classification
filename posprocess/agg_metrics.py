import sys
import json
with open("./config.json") as json_file:
    _CONFIG = json.load(json_file)
sys.path.insert(0,_CONFIG['raug_full_path'])

import os
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
from raug.utils.classification_metrics import conf_matrix, plot_conf_matrix
import matplotlib.pyplot as plt

def agg_metrics_all_folders(ex_path, num_folders, class_names):

    """
    Esta função agrega as métricas de todas as folders de determinado experimento. Salva os resultados agregados no
    caminho passado como parâmetro.

    :param ex_path(string): caminho para a pasta onde estão salvos os resultados dos experimentos.
    :param num_folders(int): número de folders utilizado na validação cruzada.
    """

    acc, precision, recall, f1 = [], [], [], []
    agg_pred, agg_real = np.array([]), np.array([])

    for folder in range(num_folders):

        folder_path = os.path.join(ex_path, f"folder_{folder+1}")

        predictions_csv_path = os.path.join(folder_path, "test_pred/predictions.csv")

        predictions = pd.read_csv(predictions_csv_path)

        # encoder for real and predict collumns
        predictions['REAL'] = predictions['REAL'].map({'ruim': 0, 'boa': 1}).astype(int)
        predictions['PRED'] = predictions['PRED'].map({'ruim': 0, 'boa': 1}).astype(int)

        real = np.array(predictions['REAL'])
        pred = np.array(predictions['PRED'])

        agg_pred = np.concatenate((agg_pred, pred))
        agg_real = np.concatenate((agg_real, real))

        acc_folder = accuracy_score(real, pred)
        precision_folder = precision_score(real, pred)
        recall_folder = recall_score(real, pred)
        f1_folder = f1_score(real, pred)

        acc.append(acc_folder)
        precision.append(precision_folder)
        recall.append(recall_folder)
        f1.append(f1_folder)

    metrics = {
        'accuracy': [f'{np.mean(acc):.2f} +- {np.std(acc):.2f}'],
        'precision': [f'{np.mean(precision):.2f} +- {np.std(precision):.2f}'],
        'recall': [f'{np.mean(recall):.2f} +- {np.std(recall):.2f}'],
        'f1-score': [f'{np.mean(f1):.2f} +- {np.std(f1):.2f}'],
    }

    metrics_df = pd.DataFrame(metrics)

    if not os.path.exists(ex_path + '/agg_metrics'):
        os.mkdir(ex_path + '/agg_metrics')

    conf_mat = conf_matrix(agg_real, agg_pred)

    plot_conf_matrix(cm=conf_mat, class_names=class_names, normalize=True, save_path=os.path.join(ex_path, "agg_metrics/agg_conf_mat.png"))

    metrics_df.to_csv(os.path.join(ex_path, "agg_metrics/agg_metrics.csv"))