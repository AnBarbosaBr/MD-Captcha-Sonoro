
"""
IMPORTANTE: Ao classificar os resultados finais, esse script gera um 
csv na localização do "output_csv"

Esses são os "PARAMETROS" do script:
            * train_path e test_path indicam os diretórios
        onde estão os dados para o treinamento e 
        validação do modelo.
            * train_pickle e test_pickle são a localização
        de um arquivo auxiliar contendo os dados já proces-
        sados pelo script. 
            Se for None, a variável é ignorada pelo script,
        se tiver algum valor, e o arquivo existir, o script
        irá ler o conteúdo dessa variável ao invés de proces-
        sar os dados dos diretórios. Se o arquivo não existir,
        o script processará os dados dos diretórios e salvará
        no arquivo pickle.
            * output_csv: Se definido, é o arquivo que será gerado pelo 
        classificador final, contendo as labels dos arquivos lidos do 
        diretório de validação e as validações. Se for None,
        é ignorado.
"""

"""
===============================================================================
                        IMPORTING LIBRARIES
===============================================================================
"""
# Importings base libraries
import os;
import pandas as pd; 
import numpy as np;
import librosa;

import sklearn.ensemble
import sklearn.model_selection 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler, Normalizer,
                                   StandardScaler)


"""
===============================================================================
                        DEFINING DATA PATHS
===============================================================================
""" 
# Input
train_path = ".\\TREINAMENTO\\"
test_path  = ".\\VALIDACAO\\"
train_pickle = None 
test_pickle = None  
# Output
output_path = None # ".\\projeto_md.csv"

"""
===============================================================================
                                SETUP
===============================================================================
"""
# String constants
SAMPLE_RATE = "sr"
LABEL = "label"
DURACAO = "duracao"
ARQUIVO_ORIGINAL = "original_file"
PREDICAO = "predicao"
seed = 42
np.random.seed(seed)

# In[3]:
"""
===============================================================================
                    DEFINING INGESTION FUNCIONS
===============================================================================
"""
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
    df.insert(loc  = 0, column = LABEL, value = labels_list)
    if(verbose): print("Inserting Duração...")
    df.insert(loc  = 1, column = DURACAO, value = duration_list)
    if(verbose): print("Inserting Sampling Rates(sr)...")
    df.insert(loc = 2, column = SAMPLE_RATE, value = sr_list )
    if(verbose): print("Inserting Original Filename...")
    df.insert(loc = 3, column = ARQUIVO_ORIGINAL, value = filename_list)
    if(verbose): print("DataFrame Created. Returning")
    return(df)
    
    
# In[4]:    
"""
===============================================================================
                    DEFINING PIPELINE FUNCIONS
===============================================================================
"""            
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


def old_extract_features(df):
    """Função usada na primeira parte do projeto"""
    return df.iloc[ : , df.columns.get_loc(0): ]

def extract_features(df):
    """Função que extraí as features baseada nos dados do audio"""
    features_df = df.iloc[ : , df.columns.get_loc(0): ]
    sample_rate = df.loc[ : , SAMPLE_RATE]
    
    mfcc_dict = dict()
    centroid_dict = dict()
    bandwidth_dict = dict()
    flatness_dict = dict()
    rolloff_dict = dict()
    zero_crossing_dict = dict()
    
    for index, row in features_df.iterrows():
        audio_data = row.values.astype('float64')
        sr = sample_rate[index]
        mfcc_dict[index] = np.mean(
                                librosa.feature.mfcc(y = audio_data,
                                                     sr=sr, 
                                                     n_mfcc=20).T,  # Transposta do mfcc
                                axis=0)
        centroid_dict[index] = np.mean(
                                librosa.feature.spectral_centroid(y = audio_data,
                                                                 n_fft=4096, hop_length=1024,
                                                                 sr = sr).T,
                                axis = 0)
        bandwidth_dict[index] = np.mean(
                                 librosa.feature.spectral_bandwidth(y = audio_data, 
                                                                   sr = sr).T,
                                axis = 0)
        flatness_dict[index]  = np.mean(
                                 librosa.feature.spectral_flatness(y = audio_data).T,
                                 axis = 0)
        rolloff_dict[index]    = np.mean(
                                    librosa.feature.spectral_rolloff(y = audio_data).T,
                                    axis = 0)
        
        zero_crossing_dict[index] = np.sum(
                                        librosa.feature.zero_crossing_rate(y = audio_data).T,
                                        axis = 0)
    mfcc_df     = pd.DataFrame.from_dict(mfcc_dict,orient = "index")
    mfcc_df.columns = ["mfcc_"+ str(col)for col in mfcc_df.columns]
    
    centroid_df = pd.DataFrame.from_dict(centroid_dict, orient = "index", columns=["centroid"])
    
    bandwidth_df = pd.DataFrame.from_dict(bandwidth_dict, orient = "index", columns=["bandwidth"])
    
    flatness_df = pd.DataFrame.from_dict(flatness_dict, orient = "index", columns=["flatness"])
    
    rolloff_df = pd.DataFrame.from_dict(rolloff_dict, orient = "index", columns=["rolloff"])
    
    zero_crossing_df = pd.DataFrame.from_dict(zero_crossing_dict, orient = "index", columns=["zero_crossing"])

    return_df = pd.concat([zero_crossing_df, rolloff_df, flatness_df, bandwidth_df, centroid_df, mfcc_df], axis = 1)
    return return_df
        

def extract_labels(df):
    labels = df.loc[ : , LABEL]
    
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(["a", "b", "c", "d", "h", "m", "n", "x", "6", "7","?"])
    labels = label_encoder.transform(labels)
    return labels

    
# In[5]:   
"""
===============================================================================
                    DEFINING MAIN FUNCIONS
===============================================================================
""" 
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
           
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(["a", "b", "c", "d", "h", "m", "n", "x", "6", "7","?"])

    train_data[PREDICAO] = label_encoder.inverse_transform(predict_train)
    test_data[PREDICAO] = label_encoder.inverse_transform(predict_test)
    return(train_data, test_data)
    
def process_folder(training_folder, validation_folder, algorithm):
    ## Ler os dados
    training_data = load_data(training_folder)
    validation_data = load_data(validation_folder)
    
    return process_data(training_data, validation_data, algorithm)




# In[6]:
"""
===============================================================================
                    DEFINING MODEL FUNCIONS
===============================================================================
""" 
## Essas funções servem para avaliar o projeto
def score_classifier(df, y_pred):
    """Recebe um dataframe completo(com dados e labels) e o compara com o y_pred"""
    prepared_data = preprocess_data(df)
    y_real = extract_labels(prepared_data)
    
    print("Matriz de Confusão:")
    print(sklearn.metrics.confusion_matrix(y_true=y_real, y_pred = y_pred))
    
    print("\n Outras Observações:")
    
    trues = y_real == y_pred
    hits = sum(trues)
    total = len(y_pred)
    print(f"It got right: {hits} from {total} letters: {100*hits/total :.2f}%")
    word_hits = prepared_data[trues].original_file.value_counts()
    unique_words = len(df.original_file.unique())
    print(f"It received {unique_words} words. ")
    print(f"It got right 4 letters of: {sum(word_hits == 4)} words. ({100*sum(word_hits==4)/unique_words:.2f}%)\n" +
          f"It got right 3 letters of: {sum(word_hits == 3)} words.\n" +
          f"It got right 2 letters of: {sum(word_hits == 2)} words.\n" +
          f"It got right 1 letters of: {sum(word_hits == 1)} words.\n" +
          f"It got right 0 letters of: {unique_words - len(word_hits)} words ({100*(unique_words - len(word_hits))/unique_words :.2f}%).\n")

    print("Palavras - Número de letras Acertadas:")
    print(word_hits)

    print("Resumo - Letras:")
    letters_count = prepared_data.label.value_counts()
    letters_right = prepared_data[trues].label.value_counts()
    letters_df = pd.DataFrame({"correct": letters_right, "total": letters_count})
    letters_df['accuracy'] = 100*letters_df['correct']/letters_df['total']
    print(letters_df.sort_values('accuracy'))
 
    return f1_score(y_true = y_real, y_pred = y_pred, average="macro")
  
def fit_predict_score(x_train, y_train, x_test, validation_data, algorithm):
    # Fit model
    print("Treinando o modelo...")
    algorithm.fit(x_train, y_train)

    # Predict
    print("Classificando a validação...")
    predict_test  = algorithm.predict(x_test)

    print("Pontuando a validação...")
    return (score_classifier(validation_data, predict_test))
    



# In[8]:
"""
===============================================================================
                        LOADING DATA
===============================================================================
""" 
print("Loading Data...")
training_data = load_data(train_path, train_pickle)
validation_data = load_data(test_path, test_pickle)


# In[9]:
"""
===============================================================================
                        RUNNING MODEL (LDA)
===============================================================================
"""
print("Running LDA...")
## Classifier 
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

### Predict and scoring the validation
train_predict_lda, test_predict_lda = process_data(training_data, validation_data, lda)


# In[10]:
"""
===============================================================================
                        RUNNING MODEL (NB)
===============================================================================
"""
print("Running NB...")
## Classifier 
nb = sklearn.naive_bayes.GaussianNB()
## NB
### Predict and scoring the validation
train_predict_nb, test_predict_nb = process_data(training_data, validation_data, nb)


# In[11]:
"""
===============================================================================
                        RUNNING MODEL (RANDOM FOREST)
===============================================================================
"""
print("Running Optimal RF...")
## RF
### Predict and scoring the validation
rf = RandomForestClassifier(random_state = seed, n_jobs = -1,
                                max_depth = 20,
                                max_features = 0.3,
                                n_estimators = 900)

train_predict_rf, test_predict_rf = process_data(training_data, validation_data, rf)

print("\n---- ---- ---- RESULTADOS ---- ---- ----\n")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    print(test_predict_rf[[LABEL, ARQUIVO_ORIGINAL, PREDICAO]])

if(output_path):
    test_predict_rf[[LABEL, ARQUIVO_ORIGINAL, PREDICAO]].to_csv(output_path)
