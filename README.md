# Project README

This project aims to build a baseline model for Named Entity Recognition (NER) using a Bi-LSTM with a Conditional Random Field (CRF). The following steps guide you through the project setup, model training, and evaluation process.

## Table of Contents
1. Introduction
2. Project Structure
3. Requirements
4. Setup
5. Running the Code
6. Functions and Methods
7. Training and Evaluation
8. Hyperparameter Tuning
9. Saving and Loading Model Predictions
10. Fine-tuning the Model

## Introduction

This project uses a Bi-LSTM with a CRF layer to perform Named Entity Recognition (NER). The model processes tokenized sentences to identify entities like locations, organizations, and persons.

## Project Structure
In this project structure noted only files needed for reproduction, all other files serve as supplimentary data for understanding of code and model

2YEARPROJECTNLP_GROUP_ABC
* Data
    * LotR 
        * Books - Original book text as txt and book text formated for annotation
        * Labeling1st - First 100 sentances from Hobbit annotated by all members of group to agree on annotation standarts
        * Labeling2nd - Second 100 sentances from Hobbit annotated by all members of group to agree on annotation standarts
        * LabelingFinals - ~500 random sentances annotated by different members of the group
    * UniversalNER
        * test - test NER datasets
        * train - train NER datasets
* Scripts
    * Baseline.py - Baseline model
    * BaselineTest.ipynb - Implimenting baseline model for our goals 
    * Anotation.py - Converting text for annotation and back for model processing

## Requirements
* Python 3.8+
* NumPy
* PyTorch
* AllenNLP
* Torcheval
* TorchCRF

## Setup
1. Clone this repository

2. Prepare the data:
Ensure the training and test datasets are placed in the data/ directory in their respective folders as shown in project structure.

3. Set random seeds:
To ensure reproducibility, set the random seeds at the beginning of your script:
### Copy code or use ready function in BaselineTest.ipynb
>import random\
>import numpy as np\
>import torch\
>random.seed(666)\
>np.random.seed(666)\
>torch.manual_seed(666)
### Copy code or use ready function in Anotation.py
>import random\
>import numpy as np\
>import torch\
>random.seed(42)\
>np.random.seed(42)\
>torch.manual_seed(42)

# Running the code


# Functions and Methods 

## In BaselineTest.ipynb
* extractData - converts iob2 file into dictionaries with sentances 
* convertDataShape - takes dictionaries with sentances and converts them into tensor file of right shape with tokens
* saveToIob2 - saves words and their labels into iob2 format
* loadingAllData - Loads annotated data and splits it into train and test
###     Cells written not as a function, but as jupiter notebook cell
* #Small dataset - dataset ment for debugging not to load full data
* #Test train - checks that model runs without errors
* #Loading all the training data for the submission - converts data to iob2 

## In Anotation.py
* Hobbit_word_per_line - converts *Hobbit* book into word per line txt file
* LOTR_word_per_line - converts *Lord of The Rings* book into word per line txt file
* combine_text_files - combines provided txt files into one
* read_and_write_random_sentences - taxes txt file of word per line and outputs three txt files with 500 sentances in each
* read_file_and_split - reads annotated files, splits them in sentances and adds 'O' token
* compare_labels - takes multiple text files with annotated text and compares them
* process_file - gives hand annotated files same format as UniversalNER dataset
* combine_annotations - combines annotated files into words and labels lists