"""
This file implements the baseline model´for group abc's project in 
named entity recognition for the subject of Natural Language 
Processing and Deep Learning. 

The baseline model is a BiLSTM model with a CRF layer on top.
Inspiration has been drawn from https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
acessed 21/03/2024.
"""

import numpy as np
import torch
import TorchCRF
from torch import nn

class baselineModel(torch.nn.Module):

    def __init__(self, nWords, tags, dimEmbed, dimHidden):
        super().__init__()
        self.dimEmbed = dimEmbed
        self.dimHidden = dimHiddenconda 
        self.vocabSize = nWords
        self.tagSetSize = len(tags)
        self.tagSet = tags

        self.embed = nn.Embedding(nWords, dimEmbed)
        self.LSTM = nn.LSTM(dimEmbed, dimHidden, batch_first = True, bidirectional = True)
        self.linear = nn.Linear(dimHidden * 2, self.tagSetSize) 
        
        self.CRF = TorchCRF.CRF(self.tagSetSize, batch_first = True)


        
    def forward(self, inputData):
        wordVectors = self.embed(inputData)
        out, _ = self.LSTM(wordVectors.view((inputData.shape[0], inputData.shape[1], self.dimEmbed)))
        backwardOut = out[:,0, - (self.dimHidden):].squeeze()
        forwardOut = out[:,-1,:self.dimHidden].squeeze()
        out = torch.cat((backwardOut, forwardOut), dim = 1)
        out = self.output(out)
        logProbs = nn.functional.log_softmax(out, dim=1)
        return logProbs
        
        



