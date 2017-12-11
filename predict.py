# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:33:01 2017

@author: jadm
"""
#%%
from gensim.models import AuthorTopicModel
model = AuthorTopicModel.load('modelo1/model.atmodel')

#%%
top_topics = model.top_topics(model.corpus)
#%%

for i in range(len(top_topics)):
    print(top_topics[i][1])
    
    