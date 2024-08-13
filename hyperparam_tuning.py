import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from wavenet_torch import WaveSeekerClassifier
from sklearn.metrics import balanced_accuracy_score
import faulthandler
import csv
faulthandler.enable()


def hyperparameter_tuning(X, y, D):
    """
    """
    scores_dict = {}

    param_grid = {
        'patch_size': [(4, 4), (8, 8)],
        'n_blocks': [2, 3],
        'emb_dim': [64, 100],
        'final_hidden_size': [4, 8],
        'use_fft': [True, False],
        'use_wavelets': [True, False]
    }

    total_combinations = sum(len(values) for values in param_grid.values())
    current_combination = 0

    for param, values in param_grid.items():
        for param_value in values:
            current_combination += 1
            print(
                f"Processing combination {current_combination}/{total_combinations} for param {param} with value {param_value}")

            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1,
                                         random_state=0)
            scores = []
            for train, test in cv.split(X, y):
                base_clf = WaveSeekerClassifier(seq_L=2 ** D, res_L=2 ** D,
                                                n_channels=1, n_out=2,
                                                epochs=5)
                setattr(base_clf, param, param_value)
                base_clf.fit(X[train], y[train])
                pred = base_clf.predict(X[test])
                bal_acc = balanced_accuracy_score(y[test], pred, adjusted=True)
                scores.append(bal_acc)

            mu = np.mean(scores)
            stdev = np.std(scores, ddof=1)
            print(f"Mean balanced accuracy: {mu}, Std: {stdev}")

            if param not in scores_dict:
                scores_dict[param] = [[param_value, mu, stdev]]
            else:
                scores_dict[param].append([param_value, mu, stdev])

    return scores_dict


if __name__ == "__main__":

    # Load numpy files
    X = np.load("X_100.data.npy").astype(np.float32)
    y = np.load("y_100.data.npy").astype(int)

    print(len(X))
    print(len(y))
    print(np.sum(y == 0))
    print(np.sum(y == 1))

    # Set D value
    D = 6

    # Hyperparameter optimization
    scores_dict = hyperparameter_tuning(X, y, D)
    print(scores_dict)

    final_param = {}
    for param, values in scores_dict.items():
        tmp = np.asarray(values)
        final_param[param] = values[np.argmax(tmp[:, 1, :])][0]

    # output file as csv
    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['param', 'value', 'mean_balanced_accuracy', 'std'])
        for param, values in scores_dict.items():
            for value, mu, std in values:
                writer.writerow([param, value, mu, std])