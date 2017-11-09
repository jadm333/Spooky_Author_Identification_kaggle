# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:10:18 2017

@author: PEPE
"""
import spacy
from gensim.models import Phrases
from gensim.corpora import Dictionary


def entidades(docs,nlp):
    processed_docs = []
    for doc in nlp.pipe(docs, n_threads=4, batch_size=100):
        ents = doc.ents  

        #doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        doc = [token.lemma_ for token in doc]

        doc.extend([str(entity) for entity in ents if len(entity) > 1])
        
        processed_docs.append(doc)

    return processed_docs

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

def preprocesamiento(df):
    #Sustituir simbolos (no necesario)
    #df['text'] = df['text'].str.replace('@', '')
    nlp = spacy.load('en')
    #stopwords(no necesario)
    #nlp.vocab["y"].is_stop = True

    docs = list(df['text'])

    docs = entidades(docs,nlp)

    docs = bigramas(docs)

    corpus = [dictionary.doc2bow(doc) for doc in docs]

    #AUTHOR2DOC
    author2doc = {}
    for aut in df.author.unique():
        author2doc[aut] = []
        
    for index, row in df.iterrows():
        author2doc[row['screen_name']].append(index)


    return corpus, dictionary, author2doc




for doc in nlp.pipe(docs, n_threads=4, batch_size=100):
    ents = doc.ents  
    #doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    doc = [token.lemma_ for token in doc]

    doc.extend([str(entity) for entity in ents if len(entity) > 1])
        
    processed_docs.append(doc)
