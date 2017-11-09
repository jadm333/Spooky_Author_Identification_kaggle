# -*- coding: utf-8 -*-

#%%  CARGAR DATOS Y PAQUTERIAS

import pandas as pd
import funcs  ##Funciones quese utilizan

train = pd.read_csv("data/train.csv")

##% FUNCIONES


#%% 

corpus, dictionary, author2doc = funcs.preprocesamiento(train)
    
