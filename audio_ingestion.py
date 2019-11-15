#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Importings base libraries
import os;
import pandas as pd; 
import librosa;
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Input Variables and Setup
training_data_path   = os.path.join(".","dados","TREINAMENTO")
validation_data_path = os.path.join(".","dados","VALIDACAO")


# In[10]:


def get_files_from(path):
    from os.path import isfile, join
    from os import listdir
    onlyfiles = [join(path,f) for f in listdir(path) if (isfile(join(path, f)) and f.endswith(".wav")) ]
    return onlyfiles

def load_wav_file(file, intervalo_segundos = 2):
    audio_data, sampling_rate = librosa.load(file, None)
    duracao = audio_data.shape[0]/sampling_rate
    frames_por_audio = sampling_rate * intervalo_segundos
    dados_por_amostra = list()
    for i, ini in enumerate(range(0, audio_data.shape[0], frames_por_audio)):
        dados_por_amostra.append(pd.Series(audio_data[ini:(ini+frames_por_audio)]))
    
    return(dados_por_amostra)

def get_labels(file):
    base = os.path.basename(file)
    without_extension = os.path.splitext(base)[0]
    return without_extension

def get_sampling_rate(file):
    return librosa.load(file, None)[1]

def join_onda_with_label(waves, labels, sampling_rate, duracao_minima = 2.0, duracao_maxima = 2.0):
    labs = pd.Series(list(labels))
    ondas = pd.DataFrame(waves)
    duracoes = [ w.shape[0]/sr for w in waves]
    ondas.insert(loc  = 0, column = 'label', value = labs)
    ondas.insert(loc  = 1, column = 'duracao', value = duracoes)
    ondas.insert(loc = 2, column = 'sr', value = sampling_rate )
    ondas = ondas[ (ondas['duracao'] >= duracao_minima) & (ondas['duracao'] <= duracao_maxima) ]
    return ondas

def get_audio_and_sr_from_df(df, row_index, sample_rate_col_index = 2, first_data_col_index = 3, asNumpyArray = True ):
    data = df.iloc[row_index, first_data_col_index: ].astype('float64')
    if(asNumpyArray):
        data = data.values
    sample_rate = df.iloc[row_index, sample_rate_col_index]
    return(data, sample_rate)


# In[7]:


f = get_files_from(training_data_path)


# In[ ]:


df = pd.DataFrame()
processed = 0
to_be = len(f)
for file in f:
    print(f"{file}: Processing {processed} of {to_be}.")
    wave = load_wav_file(file)
    sr = get_sampling_rate(file)
    labels = get_labels(file)
    temp_df = join_onda_with_label(wave, labels, sr)
    df.append(temp_df)
    processed += 1


# In[ ]:


df.to_pickle("training_data.pickle")

