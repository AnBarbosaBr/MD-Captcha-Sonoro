#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importings base libraries
import os;
import pandas as pd; 
import numpy as np;
import librosa;
import IPython.display as ipd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Models
import sklearn.model_selection # train_test_split
import sklearn.discriminant_analysis # LinearDiscriminantAnalysis
import sklearn.naive_bayes  # GaussianNB


seed = 42


# In[2]:


# SETUP


# In[3]:


## Ingestion Functions
def _process_wave_file(wave_file, labels_list, filename_list, duration_list, sr_list, data_list, interval_time = 2):
    ''' This function will append to the lists with the data from the wave file. 
    It does not have a return'''
    # Get data from the wave file:
    audio_data, sampling_rate = librosa.load(wave_file, None)
    original_filename = os.path.basename(wave_file)
    original_filename = os.path.splitext(original_filename)[0]

    # Calculate Some Attributes
    labels = list(original_filename)[0:4] # each label contains 4 letters
    frames_per_audio = sampling_rate * interval_time
    
    
    # Separate the Wave File in interval_time sections.
    rows_processed = 0
    
    for i, ini in enumerate(range(0, audio_data.shape[0], frames_per_audio)):
            
            # Calculate attributes
            this_audio = pd.Series(audio_data[ini:(ini+frames_per_audio)])
            this_duration = this_audio.shape[0]/sampling_rate
            # Update the lists with this section data.
            rows_processed += 1
            filename_list.append(original_filename)
            duration_list.append(this_duration)
            sr_list.append(sampling_rate)
            data_list.append(this_audio)
            
            
   
    # If we process more intervals than those predicted by our original_filename,
    # We label as "?"
    while(len(labels) < rows_processed):
        #print(f"adding ? to {original_filename}")
        labels.append("?")  
    
    # Update the labels list.
    labels_list.extend(labels)

def _load_wavs_from_dir(directory, verbose=False):
    # Using those imports only on this function
    from os.path import isfile, join
    from os import listdir
    
    # Reading wave files from the directory
    wave_files = [join(directory , f) for f in listdir(directory) if (isfile(join(directory, f)) and f.endswith(".wav")) ]
    
    # Creating lists that will store the data
    labels_list = list()
    filename_list = list() 
    duration_list = list()
    sr_list = list() 
    data_list = list()
    
    # Auxiliar variables
    processed = 1;   # For Verbose output
    to_be_processed = len(wave_files) # For Verbose output
    
    for file in wave_files:
        if(verbose): print(f"{file}: Processing {processed} of {to_be_processed}.")
        _process_wave_file(file, labels_list, filename_list, duration_list, sr_list, data_list)
        processed += 1
    # After process all the files, create the DataFrame
    if(verbose): print("Creating DataFrame")
    df = pd.DataFrame(data_list)
    if(verbose): print("Inserting Labels...")
    df.insert(loc  = 0, column = 'label', value = labels_list)
    if(verbose): print("Inserting Duração...")
    df.insert(loc  = 1, column = 'duracao', value = duration_list)
    if(verbose): print("Inserting Sampling Rates(sr)...")
    df.insert(loc = 2, column = 'sr', value = sr_list )
    if(verbose): print("Inserting Original Filename...")
    df.insert(loc = 3, column = "original_file", value = filename_list)
    if(verbose): print("DataFrame Created. Returning")
    return(df)


# In[4]:


## Pipeline Functions 
def load_data(data_directory, output_pickle_file = None, reuse_if_exists=True):
    if(output_pickle_file):
        output_extension = os.path.splitext(output_pickle_file)[1]
        if (output_extension != ".pickle"):
            raise("Output must be a file ended with .pickle") 
    
        if( reuse_if_exists and os.path.isfile(output_pickle_file) ):
            # If the user wants to reuse existing pickle file and it exists
            return pd.read_pickle(output_pickle_file)
        
        else:
            # If the user do not wan´t to use existing file, or if it does not exists
            df = _load_wavs_from_dir(data_directory)
            df.to_pickle(output_pickle_file)
            return df
    return(_load_wavs_from_dir(data_directory))

def preprocess_data(df):
    ''' Filtre, remova nulls, e transforme os dados nessa etapa'''
    return df.fillna(0, inplace=False)

def extract_features(df):
    features = df.iloc[ : , df.columns.get_loc(0): ]
    return features

def extract_labels(df):
    labels = df.loc[ : , "label"]
    
    label_encoder = sklearn.preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    return labels


# In[5]:


## Main functions
def process_data(training_data, validation_data, algorithm):
    # Preprocess - Filter and Imputing 
    train_data = preprocess_data(training_data)
    test_data  = preprocess_data(validation_data)
    
    # Extracting information
    x_train = extract_features(train_data)
    y_train = extract_labels(train_data)
    
    x_test = extract_features(test_data)
    y_test = extract_labels(test_data)
    
    # Fit model
    algorithm.fit(x_train, y_train)
    
    # Predict
    predict_train = algorithm.predict(x_train)
    predict_test  = algorithm.predict(x_test)
    
    return(predict_train, predict_test)
    
    
    
    
def process_folder(training_folder, validation_folder, algorithm):
    ## Ler os dados
    training_data = load_data(training_folder)
    validation_data = load_data(validation_folder)
    
    return process_data(training_data, validation_data, algorithm)
    


# In[6]:


# RUNNING THE MODEL
## Inputs
train_path = ".\\dados\\treinar\\"
test_path  = ".\\dados\\validar\\"

## Input Saving -> Will be used to avoid having to reload all data
train_pickle = ".\\dados\\saving_treinamento.pickle"
test_pickle = ".\\dados\\saving_teste.pickle"

## Algorithms
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
lda2 = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()


# In[7]:


get_ipython().run_cell_magic('time', '', '# Testando ler os dados\ntraining_data = load_data(train_path, train_pickle)\nvalidation_data = load_data(test_path, test_pickle)')


# In[8]:


get_ipython().run_cell_magic('time', '', 'train_predict, test_predict = process_data(training_data, validation_data, lda)')


# In[9]:


get_ipython().run_cell_magic('time', '', '# Testando ler os diretorios\n\n\ntrain_predict_from_folder, test_predict_from_folder = process_folder(train_path, test_path, lda2)')

