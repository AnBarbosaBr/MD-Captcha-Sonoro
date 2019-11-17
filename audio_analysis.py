#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importings base libraries
import os;
import pandas as pd; 
import librosa;
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Will read from the pickle file - use ingestion when ready
training_data_path   = os.path.join(".","dados","training_data.pickle")
validation_data_path = os.path.join(".","dados","validation_data.pickle")


# In[4]:


# Will read only part of the data
training = pd.read_pickle(training_data_path).sample(1000)


# In[5]:


training.shape


# In[6]:


[col for col in training.columns]


# In[29]:


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


# In[15]:


data, sampling_rate, label = get_data_sample_rate_and_legend_from_df(training, 3)


# In[16]:


print(f"{label}({librosa.get_duration(data, sr=sampling_rate)} sec)")
ipd.Audio(data, rate = sampling_rate)


# In[51]:





# In[53]:


show_data(training, 1)


# In[ ]:




