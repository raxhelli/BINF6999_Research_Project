# Automated Deep Learning Classification of Viral Genomes Using Frequency Chaos Game Representation (FCGR)#
## Introduction ##
This repository provides code and data for the automated classification of viral genomes using Frequency Chaos Game Representation (FCGR) and deep learning techniques. The aim is to enhance the accuracy and efficiency of viral genome classification by leveraging a non-reference-based approach.

## Content ##
### Python Scripts ###
hyperparam_tuning.py
Performs hyperparameter tuning to optimize the deep learning model's performance. Key parameters such as patch size, embedding dimension, and final hidden size are adjusted to achieve the best results.

model_training.py
Handles the training process of the deep learning model using the prepared dataset. Includes functionality for implementing stratified cross-validation to ensure robust model evaluation.

model_validation.py
Evaluates the trained model's performance on a validation dataset. Generates balanced accuracy and F1 scores, along with confusion matrices and learning curves to assess model performance and detect overfitting.

test_dataset_creation.py
Creates the test dataset used for evaluating the model on real-world sequencing data. Ensures the dataset includes a representative sample of viral sequences.

train_data_creation.py
Constructs the training dataset from raw genomic sequences. Filters and preprocesses sequences to prepare them for FCGR conversion and model training.

### Data Files ###
virus_summary.csv
Provides a summary of viral species and accession numbers used for retrieving genomic sequences.

archaea_summary.tsv
Provides a summary of archaeic species and accession numbers used for retrieving genomic sequences.

bacteria_summary.tsv
Provides a summary of baterial species and accession numbers used for retrieving genomic sequences.

eukaryote_summary.tsv
Provides a summary of eukaryotic species and accession numbers used for retrieving genomic sequences.
