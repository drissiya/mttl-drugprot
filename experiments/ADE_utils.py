import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import os


def list_of_sentence(data_path_ade):
    sentences = []
    Class = []
    sentences_class = []
    with open(os.path.join(data_path_ade, 'DRUGaeV1.rel')) as f:
        for line in f:
            pubmed_id, text, AdverseEffect, beginAE, endAE, drug, beginDrug, endDrug = line.strip().split('|')
            #if text not in sentences:
            seq = [text, 'P']
            sentences_class.append(seq)
    with open(os.path.join(data_path_ade, 'ADE-NEG.txt')) as f:
        for i, lines in enumerate(f):
            line = lines.strip().split(' ')
            text = ' '.join(line[2:])
            #if text not in sentences:
            seq = [text, 'N']
            sentences_class.append(seq)


    for line in sentences_class:
        sentences.append(word_tokenize(line[0]))
        Class.append(line[1])
    
    X_train, X_test, y_train, y_test = train_test_split(sentences, Class,
        train_size=0.8,
        test_size=0.2,
        random_state=42)
    
    return X_train, X_test, y_train, y_test



