#!/root/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
@author: PEPE
uso:
    modelo.py f_ini f_fin path
"""
#%%

#%% Paqueterias
import sys
import datetime
import re
import os
import pandas as pd
import glob
import preprocessor as p
import numpy as np
import spacy
from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.models import AuthorTopicModel
from gensim.models import LdaModel
import pickle
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import subprocess
import json
from shutil import copyfile
# Leer Archivos scv

#%%
def archivos_csv(f_ini=(datetime.datetime.today()-datetime.timedelta(days=7)).strftime('%Y-%m-%d'),f_fin=datetime.datetime.today().strftime('%Y-%m-%d'),path="Archivos_csv/"):
    
    allFiles = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.NUMBER)
    for file_ in allFiles:
        df = pd.read_csv(file_, header=0, parse_dates=True, infer_datetime_format=True, index_col=0)
        df['screen_name'] = os.path.splitext(os.path.basename(file_))[0]
        df = df.loc[df['RT_temp'] == 0]
        del df['id_tweet']
        del df['id_twitter']
        del df['created_at']
        del df['in_reply_to_user_id']
        del df['in_reply_to_status_id']
        del df['in_reply_to_screen_name']
        del df['retweet_count']
        del df['favorite_count']
        del df['longitude']
        del df['latitude']
        del df['retweeted']
        del df['creation_date']
        del df['modification_date']
        del df['RT_temp']
        del df['is_retweeted']
        df = df.loc[df['created_at_datetime'] > f_ini]
        df = df.loc[df['created_at_datetime'] < f_fin]
        df['text'] = df['text'].apply(p.clean)
        df['text'].replace('', np.nan, inplace=True)
        df.dropna(subset=['text'], inplace=True)
        df = df.drop_duplicates(subset="text", keep='last')
        list_.append(df)
    df = pd.concat(list_, ignore_index=True)
    del allFiles
    del file_
    del frame
    del list_
    del path
    return df, f_ini, f_fin

def entidades(docs,nlp):
    processed_docs = []
    for doc in nlp.pipe(docs, n_threads=4, batch_size=100):
        ents = doc.ents  

        doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

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
    df['text'] = df['text'].str.replace('@', '')
    df['text'] = df['text'].str.replace('#', '')
    df['text'] = df['text'].str.replace('!', '')
    df['text'] = df['text'].str.replace('¿', '')
    df['text'] = df['text'].str.replace('?', '')
    df['text'] = df['text'].str.replace("'", '')
    df['text'] = df['text'].str.replace('"', '')
    df['text'] = df['text'].str.replace('%', '')
    df['text'] = df['text'].str.replace('+', '')
    df['text'] = df['text'].str.replace('=', '')
    df['text'] = df['text'].str.replace('`', '')
    df['text'] = df['text'].str.replace('~', '')
    df['text'] = df['text'].str.replace('|', '')
    df['text'] = df['text'].str.replace(';', '')
    df['text'] = df['text'].str.replace('.', '')
    df['text'] = df['text'].str.replace('á', '')
    df['text'] = df['text'].str.replace('à', '')
    df['text'] = df['text'].str.replace('é', '')
    df['text'] = df['text'].str.replace('.', '')
    df['text'] = df['text'].str.replace('‘', '')
    df['text'] = df['text'].str.replace('“', '')
    df['text'] = df['text'].str.replace('”', '')
    df['text'] = df['text'].str.replace('¡', '')
    df['text'] = df['text'].str.replace('º', '')
    df['text'] = df['text'].str.replace('ª', '')
    df['text'] = df['text'].str.replace('▶️', '')
    df['text'] = df['text'].str.replace('ㅤ', '')
    df['text'] = df['text'].str.replace('.', '')
    

    nlp = spacy.load('es')

    nlp.vocab["y"].is_stop = True
    nlp.vocab["a"].is_stop = True
    nlp.vocab["ante"].is_stop = True
    nlp.vocab["a"].is_stop = True
    nlp.vocab["bajo"].is_stop = True
    nlp.vocab["con"].is_stop = True
    nlp.vocab["de"].is_stop = True
    nlp.vocab["desde"].is_stop = True
    nlp.vocab["durante"].is_stop = True
    nlp.vocab["en"].is_stop = True
    nlp.vocab["entre"].is_stop = True
    nlp.vocab["excepto"].is_stop = True
    nlp.vocab["hacia"].is_stop = True
    nlp.vocab["hasta"].is_stop = True
    nlp.vocab["mediante"].is_stop = True
    nlp.vocab["para"].is_stop = True
    nlp.vocab["por"].is_stop = True
    nlp.vocab["salvo"].is_stop = True
    nlp.vocab["según"].is_stop = True
    nlp.vocab["sin"].is_stop = True
    nlp.vocab["sobre"].is_stop = True
    nlp.vocab["tras"].is_stop = True
    nlp.vocab["y"].is_stop = True
    nlp.vocab["e"].is_stop = True
    nlp.vocab["ni"].is_stop = True
    nlp.vocab["o"].is_stop = True
    nlp.vocab["u"].is_stop = True
    nlp.vocab["que"].is_stop = True
    nlp.vocab["si"].is_stop = True
    nlp.vocab["como"].is_stop = True
    nlp.vocab["donde"].is_stop = True
    nlp.vocab["quien"].is_stop = True
    nlp.vocab["cual"].is_stop = True
    nlp.vocab["cuyo"].is_stop = True
    nlp.vocab["cuanto"].is_stop = True
    nlp.vocab["el"].is_stop = True
    nlp.vocab["lalos"].is_stop = True
    nlp.vocab["las"].is_stop = True

    docs = list(df['text'])

    docs = entidades(docs,nlp)

    docs = bigramas(docs)

    dictionary = filtrar_extremos(docs)

    corpus = [dictionary.doc2bow(doc) for doc in docs]

    #BORRAR TWEETS VACIOS
    id_borrar = [i for i in range(0,len(corpus)) if len(corpus[i]) == 0]
    df = df.drop(df.index[id_borrar])
    df = df.reset_index(drop=True)
    docs = list(df['text'])
    docs = entidades(docs,nlp)
    docs = bigramas(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    #AUTHOR2DOC
    author2doc = {}
    for aut in df.screen_name.unique():
        author2doc[aut] = []
        
    for index, row in df.iterrows():
        author2doc[row['screen_name']].append(index)


    return corpus, dictionary, author2doc

def folder():
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    i = []
    for f in all_subdirs:
        n = re.findall('\d+', f)
        if len(n) > 0:
            i.append(int(n[0]))
    return str(max(i) + 1)


#%%


##Correr script de R (actualiza los csv's)

subprocess.call(['Rscript --vanilla extraer_tweets.R'], shell=True)


####Fecha inicio y fin
#%%
if len(sys.argv)>1:
    if len(sys.argv)>2:
        if len(sys.argv)>3:
            df, f_ini, f_fin = archivos_csv(sys.argv[1],sys.argv[2],sys.argv[3])
        else:
            df, f_ini, f_fin = archivos_csv(sys.argv[1],sys.argv[2])
    else:
        df, f_ini, f_fin = archivos_csv(sys.argv[1])
else:
    df, f_ini, f_fin = archivos_csv()
#%%

corpus, dictionary, author2doc = preprocesamiento(df)
print('# de autores: %d' % len(author2doc))
print('# tokens unicos: %d' % len(dictionary))
print('# de documentos: %d' % len(corpus))

print('Corriendo modelo')
model = AuthorTopicModel(corpus=corpus, num_topics=100, id2word=dictionary.id2token, author2doc=author2doc, chunksize=2000, passes=55, eval_every=0, iterations=10000000,gamma_threshold=1e-11)

f =  'modelo' + folder()
os.makedirs(f)
os.makedirs(f+'/LDA')

model.save(f+'/model.atmodel')

print("MODELO TERMINADO Y GUARDADO")
#  LDA
print('Corriendo LDA')
ldamodel = LdaModel(corpus=corpus, num_topics=100, id2word=dictionary)

#  SALVAR LDA

pickle.dump(ldamodel, open(f+"/LDA/ldamodel.p", "wb"))
pickle.dump(corpus, open(f+"/LDA/corpus.p", "wb"))
pickle.dump(dictionary, open(f+"/LDA/dictionary.p", "wb"))

print('Terminado')
###Copiar notebooks
#
#copyfile('aut_topic_model.ipynb', f+'/aut_topic_model.ipynb')
#copyfile('LDA.ipynb', f+'/LDA/LDA.ipynb')

##AUT-topic
with open('aut_topic_model.ipynb') as file:
    nb = nbformat.read(file, as_version=4)
    
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

ep.preprocess(nb, {'metadata': {'path': f+'/'}})


with open(f+'/aut_topic_model.ipynb', 'wt') as file:
    nbformat.write(nb, file)

##LDA
with open('LDA.ipynb') as file:
    nb = nbformat.read(file, as_version=4)
    
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

ep.preprocess(nb, {'metadata': {'path': f+'/LDA/'}})


with open(f+'/LDA/LDA.ipynb', 'wt') as file:
    nbformat.write(nb, file)

#%% Info

data = {
   'f_ini' : f_ini,
   'f_fin' : f_fin,
   'n_aut' : len(author2doc),
   'n_tok' : len(dictionary),
   'n_doc' : len(corpus)
}


with open(f+'/info.json', 'w') as outfile:
    json.dump(data, outfile)
    
    
#%%Mover archivos a www

copyfile(f+'/grafica.html','/var/www/politicaenlinea.com/public_html/temas/grafica.html')
copyfile(f+'/LDA/lda.html','/var/www/politicaenlinea.com/public_html/temas/lda.html')
copyfile(f+'/info.json','/var/www/politicaenlinea.com/public_html/temas/info.json')

