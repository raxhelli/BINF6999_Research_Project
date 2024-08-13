import numpy as np
import csv
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import random
from complexcgr import FCGR
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, \
    RandomizedSearchCV
from wavenet_torch import WaveSeekerClassifier
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay)
from matplotlib import pyplot as plt
from pathlib import Path
import faulthandler
import torch
from wavenet_torch import WaveNetTorch
from collections import Counter
faulthandler.enable()


# Function to filter out short sequences that are less than 500 bp long
def filter_short_sequences_and_labels(sequences, labels, min_length=500,
                                      max_N_percentage=0.2):
    filtered_sequences = []

    filtered_labels = []
    y = np.asarray(labels)

    for sequence in sequences:
        N_count = sequence.count('N')
        N_percentage = N_count / len(sequence)
        if len(sequence) >= min_length and N_percentage <= max_N_percentage:
            filtered_sequences.append(sequence)
            filtered_labels.append(True)
        else:
            filtered_labels.append(False)

    y_filtered = y[filtered_labels]

    return filtered_sequences, y_filtered


# Function to generate 3000 bp non-overlapping reads from a sequence
def generate_reads(sequence, max_read_length=3000):
    sequence_length = len(sequence)

    # Initialize an empty list to store generated reads
    reads = []
    rev_reads = []

    if sequence_length <= max_read_length:
        reads.append(sequence)

        rev_sequence = Seq(sequence)
        rev_sequence.reverse_complement()
        rev_sequence = str(rev_sequence)
        rev_reads.append(rev_sequence)

    else:
        for i in range(0, sequence_length, max_read_length):
            tmp = sequence[i:i + max_read_length]
            reads.append(tmp)

            rev_sequence = Seq(tmp)
            rev_sequence.reverse_complement()
            rev_sequence = str(rev_sequence)
            rev_reads.append(rev_sequence)

    print(sequence_length, sequence_length / 3000, len(reads))

    return reads, rev_reads


# Function to replace bases not in ACGT with N
def replaced_bases(seq: str):
    """
    Replaces any bases not in AGCT with N

    :param seq: DNA sequences
    :return: DNA sequence only containing AGCT or N bases
    """
    replace_chars = set(list("BSYRKMWVHD"))  # Bases to replace with N
    seq_final = seq.upper()

    replacement = list(seq_final)

    for i, base in enumerate(seq_final):
        if base in replace_chars:
            replacement[i] = "N"

    return "".join(replacement)


def rev_comp(seqs):
    rev_seqs = []
    for seq in seqs:
        print(seq)
        tmp = str(Seq(seq).reverse_complement())
        # "".join(complement[base] for base in reversed(seqs)))
        rev_seqs.append(tmp)

    return tmp
    # reversing sequenc, loop though


def return_reads(seqs, labels):
    # Loop over each sequence to generate reads
    y = []
    mod_seqs = []
    for seq in seqs:
        mod_seqs.append(replaced_bases(seq))

    reads = []
    rev_seqs = []
    for seq in mod_seqs:
        read_list, reverse_reads = generate_reads(seq)
        reads.append(read_list)
        rev_seqs.append(reverse_reads)

    for i, entry in enumerate(reads):
        tmp = [labels[i] for _ in range(len(entry))]
        y.extend(tmp)

    fcgr = FCGR(k=6)
    X = []

    with Pool(2) as mp:
        for i, sub_seqs in enumerate(reads):
            tmp = mp.map(fcgr, sub_seqs)
            X.extend(tmp)

    fcgr = FCGR(k=6)
    X_rev = []

    for i, sub_seqs in enumerate(rev_seqs):
        for sub_seq in sub_seqs:
            tmp = fcgr(sub_seq)
            X_rev.append(tmp)

    X = np.asarray(X) + np.asarray(X_rev)

    return X, np.asarray(y)


def hyperparameter_tuning(X, y, D):
    """

    """
    # Define the parameter grid
    param_grid = {
        'patch_size': [(4, 4), (8, 8)],
        'n_blocks': [2, 3, 4],
        'emb_dim': [64, 81, 100, 324],
        'final_hidden_size': [4, 8, 16],
        'use_fft': [True, False],
        'use_wavelets': [True, False]
    }

    # loop through the dictionary
    scores_dict = {}

    for param, values in param_grid.items():

        for param_value in values:
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1,
                                         random_state=0)

            scores = []
            for train, test in cv.split(X, y):
                base_clf = WaveSeekerClassifier(H=2 ** D, W=2 ** D, n_out=2,
                                                epochs=5)
                setattr(base_clf, param, param_value)

                base_clf.fit(X[train], y[train])
                pred = base_clf.predict(X[test])
                bal_acc = balanced_accuracy_score(y[test], pred, adjusted=True)
                scores.append(bal_acc)

            mu = np.mean(scores)
            stdev = np.std(scores, ddof=1)

            if param not in scores_dict:
                scores_dict[param] = [[param_value, mu, stdev]]
            else:
                scores_dict[param].append([param_value, mu, stdev])

    return scores_dict


def plot_test_performance(scores: list[list]) -> None:
    """
    Plot boxplot from distributions of balanced accuracy and f1 score across all splits

    :param scores: list of scores across all splits
    :return: None
    """

    # Create boxplot
    plt.figure(figsize=(12, 6))
    plt.boxplot(scores, vert=True, patch_artist=True,
                labels=['Balanced Accuracy', 'Weighted F1 Score'])

    # Add labels, title, and grid
    plt.ylabel('Score')
    plt.title('Test Performance')
    plt.grid(True, axis='y')

    # Extract and label the median values on the plot
    medians = [np.median(score) for score in scores]
    for i, median in enumerate(medians, start=1):
        plt.text(i, median, f'{median:.2f}', horizontalalignment='center',
                 verticalalignment='bottom', fontweight='bold')

    # Save the plot as png file
    plt.savefig(
        str(Path.home() / '/Users/rachel/Desktop/BINF6999/Output/test_performance_distributions.png'))
    plt.close()


def plot_scores(train_loss, val_loss, val_f1, val_acc, epochs) -> None:
    """
    Plot training together with validation loss in one plot
    and validation accuracy together with validation f1 score in another plot

    Save plot to file.

    :param train_loss: Training loss 2D array
    :param val_loss: Validation loss 2D array
    :param val_f1: Validation F1 Score 2D array
    :param val_acc: Validation Accuracy 2D array
    :param epochs: Number of epochs
    :return: None
    """

    # Compute means and standard deviations
    mean_training_loss = np.nanmean(train_loss, axis=0)
    std_training_loss = np.nanstd(train_loss, axis=0)

    mean_validation_loss = np.nanmean(val_loss, axis=0)
    std_validation_loss = np.nanstd(val_loss, axis=0)

    mean_validation_accuracy = np.nanmean(val_acc, axis=0)
    std_validation_accuracy = np.nanstd(val_acc, axis=0)

    mean_validation_f1 = np.nanmean(val_f1, axis=0)
    std_validation_f1 = np.nanstd(val_f1, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # Set up subplots

    # Training and Validation Loss Plot
    axes[0].plot(range(1, epochs + 1), mean_training_loss,
                 label='Mean Training Loss +- SD', color='blue')
    axes[0].fill_between(range(1, epochs + 1),
                         mean_training_loss - std_training_loss,
                         mean_training_loss + std_training_loss,
                         color='blue', alpha=0.2)

    axes[0].plot(range(1, epochs + 1), mean_validation_loss,
                 label='Mean Validation Loss +- SD', color='orange')
    axes[0].fill_between(range(1, epochs + 1),
                         mean_validation_loss - std_validation_loss,
                         mean_validation_loss + std_validation_loss,
                         color='orange', alpha=0.2)

    axes[0].set_title('Training and Validation Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Validation Accuracy and F1 Score Plot
    axes[1].plot(range(1, epochs + 1), mean_validation_accuracy,
                 label='Mean Validation Accuracy +- SD', color='green')
    axes[1].fill_between(range(1, epochs + 1),
                         mean_validation_accuracy - std_validation_accuracy,
                         mean_validation_accuracy + std_validation_accuracy,
                         color='green', alpha=0.2)

    axes[1].plot(range(1, epochs + 1), mean_validation_f1,
                 label='Mean Validation F1 Score +- SD', color='red')
    axes[1].fill_between(range(1, epochs + 1),
                         mean_validation_f1 - std_validation_f1,
                         mean_validation_f1 + std_validation_f1,
                         color='red', alpha=0.2)

    axes[1].set_title('Validation Accuracy and F1 Score Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        str(Path.home() / '/Users/rachel/Desktop/BINF6999/output/training_performance.png'))  # Save as PNG file


def plot_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    """
    Plot a confusion matrix using sklearn ConfusionMatrixDisplay.

    :param cm: Confusion matrix
    :param labels: List of labels for the matrix
    :return: None
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")

    plt.title("Confusion Matrix")
    plt.savefig(
        str(Path.home() / '/Users/rachel/Desktop/BINF6999/output/confusion_matrix.png'))  # Save as PNG file
    plt.close()


if __name__ == "__main__":

    # Load numpy files
    X = np.load("X.data.npy").astype(np.float32)
    y = np.load("y.data.npy").astype(int)

    print(len(X))
    print(np.sum(y == 0))
    print(np.sum(y == 1))

    # Set D value
    D = 6

    # Hyperparameter optimization
    #scores_dict = hyperparameter_tuning(X, y, D)

    #final_param = {}
    #for param, values in scores_dict.items():
    #    tmp = np.asarray(values)
    #    final_param[param] = values[np.argmax(tmp[:, 1, :])][0]


    # Initialize objects/variables
    scores = []

    # Splitting X and y into testing and training set
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)

    for train_index, test_index in splitter.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train classification model
        D = 6

        # Train classification model using the best params
        clf = WaveSeekerClassifier(seq_L=2 ** D, res_L=2 ** D, n_channels=1,
                                   n_out=2, patch_size=(8, 8), n_blocks=2,
                                   emb_dim=64, final_hidden_size=8,
                                   use_fft=True, use_wavelets=True, epochs=5,
                                   activation=torch.nn.GELU)

        clf.fit(X_train, y_train)
        torch.save(clf.model.state_dict(), "model_params.pth")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # put in actual values
        model = WaveNetTorch(
            seq_L=2 ** D,
            res_L=2 ** D,
            n_channels=1,
            device=device,
            patch_size=(8,8),
            n_out=2,
            emb_dim=64,
            wavelet_names=["bior3.3", "sym4"],
            wave_dropout=0.5,
            use_fft=True,
            use_wavelets=True,
            n_blocks=2,
            final_dropout=0.5,
            final_hidden_size=32,
            use_smoe=True,
            return_probs=True,
            patch_mode="compress",
            use_kan=True,
            activation=torch.nn.GELU,
        )

        model.to("cpu")

        model.load_state_dict(torch.load("model_params.pth"))

        setattr(clf, "model", model)
        clf.model = model

        # model prediction
        y_predict = clf.predict(X_test)
        class_counts = Counter(y_predict)
        print("Class distribution in predictions:")
        for cls, count in class_counts.items():
            print(f"Class {cls}: {count} sequences")

        bal_acc = f1_score(y_test, y_predict, average="macro")
        scores.append(bal_acc)
        print("Balanced Accuracy Scores:", scores)

        f1 = f1_score(y_test, y_predict)
        print("F1 Score:", f1)

        cm = confusion_matrix(y_test, y_predict)
        print("Confusion Matrix:\n", cm)
