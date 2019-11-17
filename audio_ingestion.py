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


# In[ ]:


# Loading Functions - Those functions are helpers to ingest the data
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

def join_onda_with_label(waves, labels, sampling_rate, duracao_minima = 2.0, duracao_maxima = 2.0):
    labs = pd.Series(list(labels))
    ondas = pd.DataFrame(waves)
    duracoes = [ w.shape[0]/sr for w in waves]
    ondas.insert(loc  = 0, column = 'label', value = labs)
    ondas.insert(loc  = 1, column = 'duracao', value = duracoes)
    ondas.insert(loc = 2, column = 'sr', value = sampling_rate )
    ondas.insert(loc = 3, column = "original_file", value = labels)
    ondas = ondas[ (ondas['duracao'] >= duracao_minima) & (ondas['duracao'] <= duracao_maxima) ]
    return ondas


# In[46]:


f = get_files_from(training_data_path)
lista_df = list()
processed = 0
to_be = len(f)
for file in f:
    print(f"{file}: Processing {processed} of {to_be}.")
    wave = load_wav_file(file)
    sr = get_sampling_rate(file)
    labels = get_labels(file)
    temp_df = join_onda_with_label(wave, labels, sr)
    lista_df.append(temp_df)
    processed += 1


# In[47]:



df = pd.concat(lista_df, ignore_index=True)
df.to_pickle("training_data.pickle")


# In[1]:


# Verificando data frame
print(f"Shape: {df.shape} \n"+
      f"Número de Audios: {df.shape[0]} \n"+
      f"Número de Audios/4: {df.shape[0]/4}\n" +
      f"Numero De Arquivos Lidos = {len(df.original_file.unique())}\n")


# In[115]:


# Verificando quais grupos = df.groupby("original_file").sr.count().sort_values(0)
grupo_problemas = grupos[grupos < 4] 
grupo_problemas


# In[ ]:


# Esses arquivos não apresentaram 4 grupos. Hipótese: O arquivo tinha pouco menos de 2 segundos.
# Processá-los manualmente e adicionar ao DF.
# original_file 
# xxxm    3
# dhcd    3
# haa6    3
# cdbc    3
# xbnb    3
# cdbd    3
# 7x6a    3
# ca7m    3
# 6add    3
# mdn6    3
# Name: sr, dtype: int64


# In[120]:


f = get_files_from(validation_data_path)
lista_df_validation = list()
processed = 0
to_be = len(f)
for file in f:
    print(f"{file}: Processing {processed} of {to_be}.")
    wave = load_wav_file(file)
    sr = get_sampling_rate(file)
    labels = get_labels(file)
    temp_df = join_onda_with_label(wave, labels, sr)
    lista_df_validation.append(temp_df)
    processed += 1


# In[122]:


df_val = pd.concat(lista_df, ignore_index=True)
df_val.to_pickle("validation.pickle")


# In[44]:


# Working with data - Those functions are helpers to work with the final DataFrame

def get_audio_and_sampling_rate_from_df(df, row_index, asNumpyArray = True ):
    data, sampling_rate, label = get_data_sample_rate_and_legend_from_df(df, row_index, asNumpyArray)
    return(data, sampling_rate)

def get_data_sampling_rate_and_legend_from_df(df, row_index, asNumpyArray = True):
    data = df.iloc[ row_index , df.columns.get_loc(0): ].astype('float64')
    if(asNumpyArray):
        data = data.values
    sample_rate = df.loc[: , "sr"].iloc[row_index]
    label = df.loc[ : , 'label'].iloc[row_index]
    return(data, sample_rate, label)
    
def show_data(df, row):
    # Retrieve information from DF
    audio_data, sampling_rate, label = get_data_sample_rate_and_legend_from_df(df, row)
    
    # Print some stats and display the sound
    print(f"{label}({librosa.get_duration(audio_data, sr=sampling_rate)} sec)")
    ipd.display(ipd.Audio(audio_data, rate=sampling_rate))
    
    print("\n")
    # Make plots
    X = librosa.stft(audio_data)
    Xdb = librosa.amplitude_to_db(abs(X))

    plt.figure(figsize=(8, 16), dpi= 80, facecolor='w', edgecolor='k')

    plt.subplot(3, 1, 1)
    plt.title("Wave")
    librosa.display.waveplot(audio_data, sr=sampling_rate, x_axis="time")

    plt.subplot(3, 1, 2)
    plt.title("MEL")
    librosa.display.specshow(Xdb, sr=sampling_rate, x_axis="time", y_axis="mel")

    plt.subplot(3, 1, 3)
    plt.title("HZ")
    librosa.display.specshow(Xdb, sr=sampling_rate, x_axis="time", y_axis="hz")

    print("Audio")
    ipd.Audio(audio_data, rate = sampling_rate)

