import os
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np

def agg_metrics_all_folders(ex_path, num_folders):

    acc, precision, recall, f1 = [], [], [], []

    for folder in range(num_folders):

        folder_path = os.path.join(ex_path, f"folder_{folder+1}")

        predictions_csv_path = os.path.join(folder_path, "test_pred/predictions.csv")

        predictions = pd.read_csv(predictions_csv_path)

        # encoder for real and predict collumns
        predictions['REAL'] = predictions['REAL'].map({'ruim': 0, 'boa': 1}).astype(int)
        predictions['PRED'] = predictions['PRED'].map({'ruim': 0, 'boa': 1}).astype(int)

        real = np.array(predictions['REAL'])
        pred = np.array(predictions['PRED'])

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

    os.mkdir(ex_path + '/agg_metrics')
    metrics_df.to_csv(os.path.join(ex_path, "agg_metrics/agg_metrics.csv"))