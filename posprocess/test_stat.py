from scipy.stats import friedmanchisquare
import os

def test_stat_models(best_metric_values, models_names, save_path):

    if len(best_metric_values) != len(models_names):
        print("Erro: o número de amostras e de modelos não coincidem!")
        exit(1)

    out = '-------------------TESTE ESTÍSTICO----------------------\n\n'

    out += 'Resultados do Teste de Friedman:\n'

    chi, pv = friedmanchisquare(*best_metric_values)

    out += f'- Chi2: {chi}\n'
    out += f'- p-value: {pv}\n'

    if pv > 0.05:
        out += 'O teste indica que não há diferença entre os métodos.\n'

    else:
        out += 'O teste indica que há diferença entre os métodos.\n'


    with open(os.path.join(save_path, "test_stat_log.txt"), "w") as f:
        f.write(out)

'''
Exemplo de como utilizar:

data = [[0.95, 0.98, 0.97, 0.98, 0.97], [0.95, 0.96, 0.97, 0.94, 0.96], [0.95, 0.97, 0.94, 0.96, 0.96]]
models_names = ['resnet', 'mobilenet', 'vggnet']
save_path = 'save/path'

test_stat_models(data, models_names, save_path)

'''