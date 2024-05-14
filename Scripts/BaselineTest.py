#Putting all the imports in one place for readability
import numpy as np
import torch
from torch import nn
from allennlp.modules.conditional_random_field import ConditionalRandomField as CRF
from allennlp.modules import conditional_random_field as CRFmodule
from torcheval.metrics.functional import multiclass_accuracy
from torcheval.metrics.functional import multiclass_confusion_matrix as MCM
import random
from collections import Counter
import Anotation

# Setting seeds to ensure reproducibility of results

random.seed(666)
np.random.seed(666)
torch.manual_seed(666)

#Extracts the data into 2 lists of lists, one with the tokens another with the tags


def extractData(filePath):
    """
    Returns:tuple: A tuple containing input data (list of lists of words), tags (list of lists of tags),
    and metadata (list of tuples containing newdoc_id, sent_id, and text).
    """
    wordsData = []
    tagsData = []
    currentSent = None
    with open(filePath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("# sent_id"):
                sentId = line.split("= ")[1]
            elif line.startswith("#"):
                continue
            elif line:                
                parts = line.split('\t')
                word = parts[1]
                tag = parts[2]
                if sentId != currentSent:
                    currentSent = sentId
                    wordsData.append([word])
                    tagsData.append([tag])
                else:
                    wordsData[-1].append(word)
                    tagsData[-1].append(tag)
    return wordsData, tagsData

# Example usage:
#file_path = "../Data/UniversalNER/train/en_ewt-ud-train.iob2"
#words_data, tags_data = extractData(file_path)
# for words, tags in zip(words_data, tags_data):
#     print("Words:", words)
#     print("Tags:", tags)
#     print()

def combine_annotations(files):
    words = []
    tags = []
    
    for file in files:
        read = Anotation.read_file_and_split(file)
        words.extend(read[0])
        tags.extend(read[1])
    return words, tags

annotated_files = ["..\Data\LotR\LabelingFinals\Daniil.txt", "..\Data\LotR\LabelingFinals\Tobi.txt"]

words = combine_annotations(annotated_files)[0]
tags = combine_annotations(annotated_files)[1]


#Converts the Data into a tensor for use by the model

def convertDataShape(data, vocabulary = {}, labels = [], training = True, PADDING_TOKEN = '<PAD>', START_TOKEN = '<START>', STOP_TOKEN = '<END>', UNKNOWN_TOKEN = '<UNK>'):
    """
    If training is enabled creates a vocabulary of all words in a list. Otherwise, a vocabulary should be passed.
    Does the same with the labels.
    Creates a matrix of sentences and positions, where each value indicates a word via its index in the vocabulary.
    Creates another matrix of sentences and positions, where the values indicate a label.
    '<PAD>' or another user defined token is used as padding for short sentences. Will also act as an unknown token, if not training, it is assumed to be in vocabulary.
    Returns, the vocabulary, the labels and the two matrices.
    
    Input:
    data          - (string list * string list) list - List of sentences. Each sentence is a tuple of two lists. The first is a list of words, the second a list of labels.
    vocabulary    - string : int dictionary          - Dictionary of words in the vocabulary, values are the indices. Should be provided if not training. Defaults to empty dict.
    labels        - string : int dictionary          - Dictionary of labels to classify, values are the indices. Should be provided if not training. Defaults to empty dict.
    training      - boolean                          - Boolean variable deffining whether training is taking place, if yes then a new vocabulary will be created. Defaults to yes.
    PADDING_TOKEN - string                           - Token to be used as padding. Default is provided
    START_TOKEN   - string                           - Token to be used as marker for the start of the sentence. Default is provided
    STOP_TOKEN    - string                           - Token to be used as marker for the end of the sentence. Default is provided
    UNKNOWN_TOKEN - string                           - Token to be used as the unknown token. Default is provided
    
    Output:
    Xmatrix       - 2D torch.tensor                  - 2d torch tensor containing the index of the word in the sentence in the vocabulary
    Ymatrix       - 2D torch.tensor                  - 2d torch tensor containing the index of the label in the sentence in the labels
    vocabulary    - string : int dictionary          - Dictionary of words, with indices as values, used for training.
    labels        - string : int dictionary          - Dictionary of all the labels, with indices as values, used for classification. (all the labels are expected to be present in the training data, or in other words, the label list provided should be exhaustive)
    """


    if training:
        vocabList = sorted(set(word for sentence, _ in data for word in sentence))
        
        #In order to be able to work with unknown words in the future, we turn some of the least common words into unknown words so we can train on them
        #This is done by removing them from the vocab list before creating the dictionary
        vocabCount = Counter([word for sentence, _ in data for word in sentence])
        UNKNOWN_RATIO = 5 #This should be percentage of tokens we want to turn into Unknown tokens, the least common tokens will be used
        cutoff = int(len(vocabList) / (100 / UNKNOWN_RATIO)) + 1
        removeList = vocabCount.most_common()[:-cutoff:-1]
        for i in removeList:
            vocabList.remove(i[0])

        # Adding the special tokens in the first positions after the least common have been removed and creating the dictionaries
        vocabList = [PADDING_TOKEN, START_TOKEN, STOP_TOKEN, UNKNOWN_TOKEN] + vocabList
        vocabulary = {word: i for i, word in enumerate(vocabList)}
        labelList = [PADDING_TOKEN, START_TOKEN, STOP_TOKEN] + sorted(set(label for _, sentenceLabels in data for label in sentenceLabels))
        labels = {label: i for i, label in enumerate(labelList)}
    
    # Adding two to the max len in order to accomodate the introduction of start and end tokens
    maxLen = max(len(sentence) for sentence, _ in data) + 2
    Xmatrix = np.zeros((len(data), maxLen), dtype=int)
    Ymatrix = np.zeros((len(data), maxLen), dtype=int)

    for i, (sentence, sentenceLabels) in enumerate(data):
        #Set the first token as the start token (assumes it's index is 1)
        Xmatrix[i, 0] = 1
        Ymatrix[i, 0] = 1
        #Set all the indices to the correct index, with the unknown token as default
        for j, word in enumerate(sentence):
            Xmatrix[i, j+1] = vocabulary.get(word, vocabulary[UNKNOWN_TOKEN])
        for j, label in enumerate(sentenceLabels):
            Ymatrix[i, j+1] = labels.get(label, labels[START_TOKEN])
            lastWord = j         
        # Sets the token after the last word as en end token
        Xmatrix[i, lastWord + 2] = 2
        Ymatrix[i, lastWord + 2] = 2
    
    return torch.tensor(Xmatrix, dtype=torch.long), torch.tensor(Ymatrix, dtype=torch.long), vocabulary, labels

# two first sentences of EWT training dataset so that quickdebugging can be run



trainingDebugSen = [["Where", "in", "the", "world", "is", "Iguazu", "?"], ["Iguazu", "Falls"]]
trainingDebugTags = [["O", "O", "O", "O", "O", "B-LOC", "O"], ["B-LOC", "I-LOC"]]

dataDebug, labelsDebug, vocabDebug, tagsDebug = convertDataShape(list(zip(trainingDebugSen, trainingDebugTags)))
print(dataDebug)
print(labelsDebug)
print(vocabDebug)
print(tagsDebug)

class baselineModel(torch.nn.Module):
    def __init__(self, nWords, tags, dimEmbed, dimHidden, constraints):
        super().__init__()
        self.dimEmbed = dimEmbed
        self.dimHidden = dimHidden
        self.vocabSize = nWords
        self.tagSetSize = len(tags)

        self.embed = nn.Embedding(nWords, dimEmbed)
        self.LSTM = nn.LSTM(dimEmbed, dimHidden, bidirectional=True)
        self.linear = nn.Linear(dimHidden * 2, self.tagSetSize)
        

        # Initialize the CRF layer
        self.CRF = CRF(self.tagSetSize, constraints = constraints, include_start_end_transitions=True)

    def forwardTrain(self, inputData, labels):
        # Embedding and LSTM layers
        wordVectors = self.embed(inputData)
        lstmOut, _ = self.LSTM(wordVectors)
        
        # Linear layer
        emissions = self.linear(lstmOut)
        
        # CRF layer to compute the log likelihood loss
        log_likelihood = self.CRF(emissions, labels)
        
        # The loss is the negative log-likelihood
        loss = -log_likelihood
        return loss
        
    def forwardPred(self, inputData):
        # Embedding and LSTM layers
        wordVectors = self.embed(inputData)
        lstmOut, _ = self.LSTM(wordVectors)
        
        # Linear layer
        emissions = self.linear(lstmOut)
        
        # Decode the best path
        best_paths = self.CRF.viterbi_tags(emissions)
        
        # Extract the predicted tags from the paths
        predictions = [path for path, score in best_paths]
        return predictions



def saveToIob2(words, labels, outputFilePath):
    """
    Save words and their corresponding labels in IOB2 format.

    Args:
    words (list): List of lists containing words.
    labels (list): List of lists containing labels.
    output_file (str): Path to the output IOB2 file.
    """
    with open(outputFilePath, 'w', encoding='utf-8') as file:
        for i in range(len(words)):
            for j in range(len(words[i])):
                line = f"{j+1}\t{words[i][j]}\t{labels[i][j]}\n"
                file.write(line)
            file.write('\n')


# two first sentences of EWT training dataset so that quickdebugging can be run

tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]

trainingDebugSen = [["Where", "in", "the", "world", "is", "Iguazu", "?"], ["Iguazu", "Falls"]]
trainingDebugTags = [["O", "O", "O", "O", "O", "B-LOC", "O"], ["B-LOC", "I-LOC"]]

dataDebug, labelsDebug, vocabDebug, tagsDebug = convertDataShape(list(zip(trainingDebugSen, trainingDebugTags)))

#Quick traininig script on the debug dataset

DIM_EMBEDDING = 100
LSTM_HIDDEN = 50
LEARNING_RATE = 0.01
EPOCHS = 5

random.seed(666)
np.random.seed(666)
torch.manual_seed(666)

constraint_type = None

model = baselineModel(len(vocabDebug), tagsDebug, DIM_EMBEDDING, LSTM_HIDDEN, constraint_type)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    
    optimizer.zero_grad()
    loss = model.forwardTrain(dataDebug, labelsDebug)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item()}")


#Getting predicitons and checking accuracy


with torch.no_grad():
    predictsDebug = model.forwardPred(dataDebug)

confMat = MCM(torch.flatten(torch.tensor(predictsDebug, dtype=torch.long)), torch.flatten(labelsDebug), num_classes = len(tagsDebug))

acc = torch.trace(confMat[1:,1:])/torch.sum(confMat[1:,1:]) #Taking away the first collumn and first row, because those correspond to the padding token and we don't care
acc

# Loading all the training data sets

filePathTrain = "../Data/UniversalNER/train/"
wordsData = []
tagsData = []
datasets = ["da_ddt", "en_ewt", "hr_set", "pt_bosque", "sk_snk", "sr_set", "sv_talbanken", "zh_gsdsimp", "zh_gsd"]

for i in datasets:
    wordsDataTemp, tagsDataTemp = extractData(filePathTrain + i + "-ud-train.iob2")
    wordsData += wordsDataTemp
    tagsData += tagsDataTemp

trainData, trainLabels, vocab, labels = convertDataShape(list(zip(wordsData, tagsData)))