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
from collections import Counter


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
    X = np.load("X_train.npy").astype(np.float32)
    y = np.load("y_train.npy").astype(int)

    X_test = np.load("X_test.npy").astype(np.float32)
    y_test = np.load("y_test.npy").astype(int)

    print(len(X_test))
    #print(np.sum(y == 0))
    #print(np.sum(y == 1))

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)

    D = 6

    for train_index, test_index in splitter.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train classification model
        D = 6
        scores = []
        # Train classification model using the best params
        clf = WaveSeekerClassifier(seq_L=2 ** D, res_L=2 ** D, n_channels=1,
                                   n_out=2,
                                   patch_size=(8, 8), n_blocks=2, emb_dim=64,
                                   final_hidden_size=8, use_fft=True,
                                   use_wavelets=True, epochs=35)

        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)

        class_counts = Counter(y_predict)
        print("Class distribution in predictions:")
        for cls, count in class_counts.items():
            print(f"Class {cls}: {count} sequences")

        y_predict = clf.predict(X_test)
        bal_acc = f1_score(y_test, y_predict, average="macro")
        scores.append(bal_acc)
        print("Balanced Accuracy Scores:", scores)

        f1 = f1_score(y_test, y_predict)
        print("F1 Score:", f1)

        cm = confusion_matrix(y_test, y_predict)
        print("Confusion Matrix:\n", cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
