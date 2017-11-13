# -*- coding: utf-8 -*-

#%%  CARGAR DATOS Y PAQUTERIAS

import pandas as pd
#import funcs  ##Funciones quese utilizan

train = pd.read_csv("data/train.csv")

##% FUNCIONES


#%% 
import spacy
import string
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from gensim.models import Phrases
from gensim.models import AuthorTopicModel
from gensim.corpora import Dictionary

nlp = spacy.load('en')

# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_text(docs):
    texts = []
    punctuations = string.punctuation
    for doc in docs:
        doc = nlp(doc)
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)

def bigramas(docs):
    bigram = Phrases(docs, min_count=20)
    doc = docs
    for idx in range(len(doc)):
        for token in bigram[doc[idx]]:
            if '_' in token:
                doc[idx].append(token)
    return doc

def filtrar_extremos(docs,max_freq=0.5,min_wordcount=2,n_top=3):
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)
    dictionary.filter_n_most_frequent(n_top)
    _ = dictionary[0]

    return dictionary

#%%

train_cleaned = cleanup_text(train['text'])
train_c = list(train_cleaned)
for s in range(0,len(train_c)):
    train_c[s] = train_c[s].split()

docs = bigramas(train_c)
dictionary = filtrar_extremos(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]
id_borrar = [i for i in range(0,len(corpus)) if len(corpus[i]) == 0]
#%%
corpus, dictionary, author2doc

#%%
#BORRAR textos VACIOS
train = train.drop(train.index[id_borrar])
train = train.reset_index(drop=True)
docs = list(train['text'])
train_cleaned = cleanup_text(train['text'])
train_c = list(train_cleaned)
for s in range(0,len(train_c)):
    train_c[s] = train_c[s].split()
docs = bigramas(train_c)
dictionary = filtrar_extremos(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]