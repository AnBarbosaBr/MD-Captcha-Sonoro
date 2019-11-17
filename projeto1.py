#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[8]:


# SETUP


# In[9]:


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


# In[10]:


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
    prepared = df[df.label != "?"] # Remove rows with unknown labels
    return prepared.fillna(0, inplace=False)

def extract_features(df):
    features = df.iloc[ : , df.columns.get_loc(0): ]
    return features

def extract_labels(df):
    labels = df.loc[ : , "label"]
    
    label_encoder = sklearn.preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    return labels

def score_classifier(df, y_pred):
    prepared_data = preprocess_data(df)
    y_real = extract_labels(prepared_data)
    
    print("Confusion Matrix:")
    print(sklearn.metrics.confusion_matrix(y_true=y_real, y_pred = y_pred))
    
    print("\n Other Observations:")
    
    trues = y_real == y_pred
    hits = sum(trues)
    total = len(y_pred)
    print(f"It got right: {hits} from {total} letters: {100*hits/total :.2f}%")
    word_hits = prepared_data[trues].original_file.value_counts()
    unique_words = len(df.original_file.unique())
    print(f"It received {unique_words} words. ")
    print(f"It got right 4 letters of: {sum(word_hits == 4)} words.\n" +
          f"It got right 3 letters of: {sum(word_hits == 3)} words.\n" +
          f"It got right 2 letters of: {sum(word_hits == 2)} words.\n" +
          f"It got right 1 letters of: {sum(word_hits == 1)} words.\n" +
          f"It got right 0 letters of: {unique_words - len(word_hits)} words.\n")
    
    print("Those are the words and hit count:")
    print(word_hits)
    print("Those are the letters:")
    print(prepared_data[trues].label.value_counts())
    


# In[11]:


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
    
    score_classifier(validation_data, predict_test)
    
    return(predict_train, predict_test)
    
def process_folder(training_folder, validation_folder, algorithm):
    ## Ler os dados
    training_data = load_data(training_folder, verbose=True)
    validation_data = load_data(validation_folder, verbose=True)
    
    return process_data(training_data, validation_data, algorithm)
    


# In[12]:


# RUNNING THE MODEL
## Inputs
train_path = ".\\dados\\TREINAMENTO\\"
test_path  = ".\\dados\\VALIDACAO\\"

## Input Saving -> Will be used to avoid having to reload all data
train_pickle = ".\\dados\\treina_1752.pickle"
test_pickle = ".\\dados\\valida_1752.pickle"

## Algorithms
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
nb = sklearn.naive_bayes.GaussianNB()


# In[13]:


get_ipython().run_cell_magic('time', '', '# Testando ler os dados\ntraining_data = load_data(train_path, train_pickle)\nvalidation_data = load_data(test_path, test_pickle)')


# In[17]:


get_ipython().run_cell_magic('time', '', 'train_predict_lda, test_predict_lda = process_data(training_data, validation_data, lda)')


# In[18]:


get_ipython().run_cell_magic('time', '', 'score_classifier(validation_data, test_predict_lda)')


# In[19]:


get_ipython().run_cell_magic('time', '', 'score_classifier(training_data, train_predict_lda)')


# In[14]:


get_ipython().run_cell_magic('time', '', 'train_predict_nb, test_predict_nb = process_data(training_data, validation_data, nb)')


# In[15]:


get_ipython().run_cell_magic('time', '', 'score_classifier(validation_data, test_predict_nb)')


# In[16]:


get_ipython().run_cell_magic('time', '', 'score_classifier(training_data, train_predict_nb)')

