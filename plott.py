import matplotlib.pyplot as plt
import numpy as np
import os
import json
from lib.base.constants import Constants

cc = Constants

def plot(x, experiment_date: str, dataset: str, feature: str, lang: str, grads: bool = False):
    print(x)
    def get_data(fl):
        with open(fl) as json_file:
            data = json.load(json_file)
        return data

    rnd = get_data(os.path.join(cc.GRAPHS_PATH, experiment_date, 
                      f"{dataset}_{lang}_{feature}_task_random=full_grads={grads}_variational={str(False)}.json"))
    d = get_data(os.path.join(cc.GRAPHS_PATH, experiment_date, 
                      f"{dataset}_{lang}_{feature}_task_random={str(None)}_grads={grads}_variational={str(False)}.json"))
    var_rnd = get_data(os.path.join(cc.GRAPHS_PATH, experiment_date, 
                      f"{dataset}_{lang}_{feature}_task_random=full_grads={grads}_variational={str(True)}.json"))
    var_d = get_data(os.path.join(cc.GRAPHS_PATH, experiment_date, 
                      f"{dataset}_{lang}_{feature}_task_random={str(None)}_grads={grads}_variational={str(True)}.json"))

    print("var. ord. / data. ord.", np.array(var_d['data']['loss']) / np.array(d['data']['loss']))
    print("data. ord. / data. rand.", np.array(var_rnd['data']['loss']) / np.array(rnd['data']['loss']))

    print("data. rand. / data. ord.", np.array(rnd['data']['loss']) / np.array(d['data']['loss']))
    print("var. rnd. / var. ord.", np.array(var_rnd['data']['loss']) / np.array(var_d['data']['loss']))

    fig, axs = plt.subplots(2, 2, sharex=True)
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["figure.figsize"] = (16, 12)

    axs[0, 0].bar(x, var_d['data']['loss'], color = 'r', label = 'variational')
    axs[0, 0].bar(x, d['data']['loss'], color = 'k', label = 'data')
    axs[0, 0].set_title("Codelength on pretrained network")
    axs[0, 0].set_xticks(x)
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[1, 0].bar(x, var_rnd['data']['loss'], color = 'r', label = 'variational')
    axs[1, 0].bar(x, var_d['data']['loss'], color = 'y', label = 'ord. variational')
    axs[1, 0].bar(x, rnd['data']['loss'], color = 'k', label = 'data')
    axs[1, 0].set_title("Codelength on rand. init. network")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[0, 1].plot(x, d['data']['metrics'],  lw = 2, label = 'pretrained')
    axs[0, 1].plot(x, rnd['data']['metrics'],  lw = 2, label = 'rand. init.')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].set_title("Metrics [F1]")

    axs[1, 1].plot(x, np.abs(np.array(d['data']['metrics']) - np.array(rnd['data']['metrics'])), c = 'k', lw = 2, label = 'data')
    axs[1, 1].plot(x, np.abs(np.array(var_d['data']['metrics']) - np.array(var_rnd['data']['metrics'])), c = 'r', lw = 2, label = 'variational')
    axs[1, 1].set_title("Selectivity " + r"$\sigma(m, r, \hat{y}, y) \triangleq |r(m, \hat{y}, y) - r(m_{rand}, \hat{y}, y)|$")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    fig.tight_layout()
    # plt.show()

    plt.savefig(f'{feature}.png', bbox_inches='tight')

def main():
    x = np.arange(1, 24, 1)
    experiment_date = '2022-02-10'
    dataset = 'bert'
    feature = 'pos_tag'
    lang = 'en'
    plot(x, experiment_date, dataset, feature, lang, grads = False)

if __name__ == '__main__': main()
