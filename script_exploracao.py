#!/usr/bin/env python3
# coding: utf-8

"""

IMPORTANTE: 
    1) Ao classificar os resultados finais, esse script gera um 
csv com as classificações obtidas, se o parametro output_csv for
diferente de None.
    2) Rodar o processo de descobrimento(DISCOVERY_PROCESS) é 
lento, demorou várias horas para rodar num processador AMD A8-7600
com 3.10GHz e com 12GB de RAM.

Esses são os "PARAMETROS" do script:
            * DISCOVERY_PROCESS é uma booleana que indica
        se você deseja executar todo o processo de des-
        coberta que o aluno realizou. Caso seja False,
        o script só irá realizar o procedimento de clas-
        sificação.
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
            * output_csv: Se defilnido, é o arquivo que será gerado pelo 
        classificador final, contendo as labels dos arquivos lidos do 
        diretório de validação e as validações. Se for None,
        é ignorado.


Conteúdo do Script - o número da linha é aproximado:
    - As 10 linhas após a importação das bibliotecas
        Onde estão os "parâmetros" do script
        
    - Até a linha 567
      Preparação do ambiente, definição de variáveis,
    funções e classes auxiliares que foram usadas no
    processo de descoberta e análise da base.

    - A partir da linha 567
        Se a variável DISCOVERY_PROCESS for True,
    então o procedimento de análise será repetido.
    É um processo bem longo, 

    - Por volta da linha 980 
        Acaba o processo de descoberta e começa o
    de classificação, feito com o melhor classifi-
    cador que consegui obter.
"""
# Preparacao ---------------------

"""
===============================================================================
                        IMPORTING LIBRARIES
===============================================================================
"""
import os
import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
                        PARAMETROS
===============================================================================
"""
## Opções
DISCOVERY_PROCESS = False
#DISCOVERY_PROCESS = True
## Inputs
train_path = ".\\TREINAMENTO\\"
test_path  = ".\\VALIDACAO\\"
train_pickle = None 
test_pickle = None  
## Output
output_csv = None # ".\\projeto_andre_11001814.csv"

"""
===============================================================================
                        CONSTANTES
===============================================================================
"""
seed = 42
np.random.seed(seed)
SAMPLE_RATE = "sr"
LABEL = "label"
DURACAO = "duracao"
ARQUIVO_ORIGINAL = "original_file"
PREDICAO = "predicao"

"""
===============================================================================
                        CLASSES AUXILIARES
===============================================================================
"""
## Classes usadas para exploração dos dados
"""
Essas classes foram criadas com o intuito de serem usadas dentro de
uma Pipeline. Não conseguimos implementar o pipeline usando as funções
do sciki-learn, mas usamos as classes nas análises como forma de controlar 
as transformações que foram efetuadas nos dados. 
"""
class BaseAbstractExtractor(object):
    def __init__(self):
        self.params = dict()
        
    def get_params(self, deep=None):
        return self.params
    
    def set_params(self, **kwargs):
        self.params.update(kwargs)
        
    def fit(self, x, y=None):
        pass
    def transform(self, x,y=None):
        pass
    def fit_transform(self, x, y=None):
        self.fit(x,y)
        return self.transform(x,y)

class PlainDataExtractor(BaseAbstractExtractor):
    
    def __init__(self):
        self.params = dict()
        
    def get_params(self, deep=None):
        return self.params
    
    def set_params(self, **kwargs):
        self.params.update(kwargs)
        
    def fit(self, x, y=None):
        pass
    def transform(self, x,y=None):
        return x.iloc[ : , x.columns.get_loc(0): ]
    
    def fit_transform(self, x, y=None):
        self.fit(x,y)
        return self.transform(x,y)

class BasicExtractor(BaseAbstractExtractor):
    def __init__(self, n_fft=2048, hop_length=512):
        self.params = {'n_fft' : n_fft,
                    'hop_length' : hop_length
        }
    
        
        self.zero_crossing_dict = dict()
        self.centroid_dict = dict()
        self.bandwidth_dict = dict()
        self.flatness_dict = dict()
        self.rolloff_dict = dict()
        self.zero_crossing_dict = dict()
    
    def get_params(self, deep=None):
        return self.params
    
    def set_params(self, **kwargs):
        self.params.update(kwargs)
        
    def fit(self, x,y=None):
        features_df = x.iloc[ : , x.columns.get_loc(0): ]
        sample_rate = x.loc[ : , SAMPLE_RATE]

        for index, row in features_df.iterrows():
            audio_data = row.values.astype('float64')
            sr = sample_rate[index]
            self.centroid_dict[index] = np.mean(
                                    librosa.feature.spectral_centroid(
                                        y = audio_data,
                                        sr = sr,
                                        n_fft = self.params['n_fft'],
                                        hop_length = self.params['hop_length']
                                    ).T,
                                    axis = 0)
            self.bandwidth_dict[index] = np.mean(
                                    librosa.feature.spectral_bandwidth(y = audio_data, 
                                                                    sr = sr).T,
                                    axis = 0)
            self.flatness_dict[index]  = np.mean(
                                    librosa.feature.spectral_flatness(y = audio_data).T,
                                    axis = 0)
            self.rolloff_dict[index]    = np.mean(
                                        librosa.feature.spectral_rolloff(y = audio_data).T,
                                        axis = 0)

            self.zero_crossing_dict[index] = np.sum(
                                            librosa.feature.zero_crossing_rate(y = audio_data).T,
                                            axis = 0)

    def transform(self, x,y=None):
        results = list()
        results.append(pd.DataFrame.from_dict(self.centroid_dict, orient = "index", columns=["centroid"]))
        results.append(pd.DataFrame.from_dict(self.bandwidth_dict, orient = "index", columns=["bandwidth"]))
        results.append(pd.DataFrame.from_dict(self.flatness_dict, orient = "index", columns=["flatness"]))
        results.append(pd.DataFrame.from_dict(self.rolloff_dict, orient = "index", columns=["rolloff"]))
        results.append(pd.DataFrame.from_dict(self.zero_crossing_dict, orient = "index", columns=["zero_crossing"]))
        return pd.concat(results, axis=1)

class MFCCExtractor(BaseAbstractExtractor):
    
    def __init__(self, n_mfcc=20):
        self.mfcc_dict = dict()
        self.params = {
            "n_mfcc" : n_mfcc
        }
    
    def get_params(self, deep=None):
        return self.params
    
    def set_params(self, **kwargs):
        self.params.update(kwargs)
    
    def fit(self, x, y=None):
        features_df = x.iloc[ : , x.columns.get_loc(0): ]
        sample_rate = x.loc[ : , SAMPLE_RATE]
        for index, row in features_df.iterrows():
            audio_data = row.values.astype('float64')
            sr = sample_rate[index]
            self.mfcc_dict[index] = np.mean(
                                    librosa.feature.mfcc(y = audio_data,
                                                        sr=sr, 
                                                        n_mfcc=self.params['n_mfcc']
                                                        ).T,  # Transposta do mfcc
                                    axis=0)
    def transform(self, x,y=None):            
        mfcc_df = pd.DataFrame.from_dict(
                            self.mfcc_dict,
                            orient = "index")
        mfcc_df.columns = ["mfcc_"+ str(col)for col in mfcc_df.columns]
        return mfcc_df

class TonnetzExtractor(BaseAbstractExtractor):
    def __init__(self):
        self.tonnetz_dict = dict()
        
    def fit(self, x, y=None):
        features_df = x.iloc[ : , x.columns.get_loc(0): ]
        sample_rate = x.loc[ : , SAMPLE_RATE]
        for index, row in features_df.iterrows():
            audio_data = row.values.astype('float64')
            sr = sample_rate[index]
            self.tonnetz_dict[index]   = np.mean(
                                        librosa.feature.tonnetz(y = audio_data, sr = sr).T,
                                        axis = 0)

    def transform(self, x, y = None):
        tonnetz_df = pd.DataFrame.from_dict(self.tonnetz_dict, orient = "index")
        tonnetz_df.columns = ["tonnetz_"+ str(col)for col in tonnetz_df.columns]
        return tonnetz_df

    def fit_transform(self, x,y = None):
        self.fit(x,y)
        return self.transform(x,y)

class MultiFeatureExtractor(BaseAbstractExtractor):
    '''Parametro extractors é uma lista de extratores, cujos outputs
    serão concatenados.'''
    def __init__(self, extractors = BaseAbstractExtractor()):
        self.params = {'extractors' : extractors}
    
    
    def get_params(self, deep=None):
        return self.params
    
    def set_params(self, **kwargs):
        self.params.update(kwargs)
    
    
    def fit(self, x, y=None):
        extractors = self.params['extractors']
        total_extractors = len(extractors)
        print(f"Starting feature fitting with {total_extractors} exctractors.")
        for i, extractor in enumerate(extractors):
            print(f"Starting fitting {i}")
            extractor.fit(x,y)
            print(f"Finished fitting {i}")
    
    
    
    def transform(self, x, y = None):
        extractors = self.params['extractors']
        total_extractors = len(extractors)
        
        results = list()
        print(f"Starting feature transformation with {total_extractors} exctractors.")
        for i, extractor in enumerate(extractors):
            print(f"Starting transformation {i}")
            results.append(extractor.transform(x,y))
            print(f"Finished transformation {i}")
        return pd.concat(results, axis = 1)
    
    def fit_transform(self, x, y = None):
        print("Starting fit_transform.")
        self.fit(x,y)
        return self.transform(x, y)



"""
===============================================================================
                    DEFINING INGESTION FUNCIONS
===============================================================================
"""
def _process_wave_file(wave_file, labels_list, filename_list, duration_list, sr_list, data_list, interval_time = 2):
    """Essa função não tem retorno. Ela altera as listas que entram."""
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
    """Retorna um DataFrame com os audios carregados letra a letra"""
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
    processed = 1   # For Verbose output
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

    
"""
===============================================================================
                    DEFINING PIPELINE FUNCIONS
===============================================================================
"""     
def load_data(data_directory, pickle_file = None, reuse_if_exists=True):
    """Tenta ler o pickle_file. Se não existir, lê os arquivos do diretório
    e salva o pickle file."""
    if(pickle_file):
        output_extension = os.path.splitext(pickle_file)[1]
        if (output_extension != ".pickle"):
            raise("Output must be a file ended with .pickle") 
    
        if( reuse_if_exists and os.path.isfile(pickle_file) ):
            # If the user wants to reuse existing pickle file and it exists
            return pd.read_pickle(pickle_file)
        
        else:
            # If the user do not wan´t to use existing file, or if it does not exists
            df = _load_wavs_from_dir(data_directory)
            df.to_pickle(pickle_file)
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


    # In[13]:


def extract_labels(df):
    """Faz o LabelEncoding das classes."""
    labels = df.loc[ : , LABEL]
    
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(["a", "b", "c", "d", "h", "m", "n", "x", "6", "7","?"])
    labels = label_encoder.transform(labels)
    return labels
  
"""
===============================================================================
                    DEFINING SCORING FUNCIONS
===============================================================================
""" 
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

   
"""
===============================================================================
                    DEFINING MAIN FUNCIONS
===============================================================================
""" 
def process_data(training_data, validation_data, algorithm):
    # Preprocess - Filter and Imputing 
    print("Carregando dados de treinamento...")
    train_data = preprocess_data(training_data)
    
    print("Carregando dados de validação...")
    test_data  = preprocess_data(validation_data)
    
    # Extracting information
    print("Extraindo atributos e classes do treinamento...")
    x_train = extract_features(train_data)
    y_train = extract_labels(train_data)
    
    print("Extraindo atributos e classes da validação...")
    x_test = extract_features(test_data) 
    # y_test = extract_labels(test_data)
    
    
    # Fit model
    print("Treinando o modelo...")
    algorithm.fit(x_train, y_train)
    

    # Predict
    print("Classificando o treinamento...")
    predict_train  = algorithm.predict(x_train)
    print("Classificando a validação...")
    predict_test  = algorithm.predict(x_test)
    
    print("Pontuando a validação...")
    score_classifier(validation_data, predict_test)
        
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(["a", "b", "c", "d", "h", "m", "n", "x", "6", "7","?"])

    train_data[PREDICAO] = label_encoder.inverse_transform(predict_train)
    test_data[PREDICAO] = label_encoder.inverse_transform(predict_test)
    return(train_data, test_data)
    
def process_folder(training_folder, validation_folder, algorithm):
    """Wrapper para as funcoes necessarias para rodar o algorithm no projeto"""
    ## Ler os dados
    training_data = load_data(training_folder)
    validation_data = load_data(validation_folder)
    
    return process_data(training_data, validation_data, algorithm)
    
"""
===============================================================================
                    INICIO DO PROCESSO DE DESCOBERTA
===============================================================================
""" 
if DISCOVERY_PROCESS:
    
    print("Iniciando processo de descoberta.")
    ## Loading Data
    ## Algoritmos da Parte 1
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    nb = sklearn.naive_bayes.GaussianNB()

    ## Loading
    print("Carregando dados.")
    training_data = load_data(train_path, train_pickle)
    validation_data = load_data(test_path, test_pickle)

    ## Pre Processamento
    # Preprocess Data - Filter and Imputing 
    # This cell takes a long time to run
    print("Preprocessando dados de treinamento...")
    train_data = preprocess_data(training_data)

    print("Preprocessando dados de validação...")
    test_data = preprocess_data(validation_data)

    # Extracting information using the extract features
    print("Extraindo atributos e classes do treinamento...")
    x_train = extract_features(train_data)
    y_train = extract_labels(train_data)

    print("Extraindo atributos e classes da validação...")
    x_test = extract_features(test_data)
    y_test = extract_labels(test_data)

    print("Salvando os dados puros...")
    # Raw data - used to search for the best features set
    x_train_raw = train_data
    x_test_raw = test_data

    print("Calculando atributos antigos...")
    # Old Features:
    x_train_old = old_extract_features(train_data)
    x_test_old = old_extract_features(test_data)

    print("Leiutra de dados - Completa.")
    print("Formatos da base de treinamento")
    print(f"X: {x_train.shape}")
    print(f"Y: {y_train.shape}")

    print("Formatos da base de teste")
    print(f"X: {x_test.shape}")
    print(f"Y: {y_test.shape}")


    # # Comparacao com a Parte 1
    print("Comparando os antigos atributos contra os novos.")

    print("\n --- LDA --- \n")
    print("\nAtributos Antigos\n")
    fit_predict_score(x_train_old, y_train, x_test_old, validation_data, lda)
    print("\nAtributos Novos\n")
    fit_predict_score(x_train, y_train, x_test, validation_data, lda)

    print("\n --- NB --- \n")
    print("\nAtributos Antigos\n")
    fit_predict_score(x_train_old, y_train, x_test_old, validation_data, nb)
    print("\nAtributos Novos\n")
    fit_predict_score(x_train, y_train, x_test, validation_data, nb)

    # RF
    print("\n --- The Optimal Random Forest Classifier --- \n")
    rf = sklearn.ensemble.RandomForestClassifier(
                                random_state = seed, n_jobs = -1,
                                max_depth = 20,
                                max_features = 0.3,
                                n_estimators = 900)

    print("\nAtributos antigos\n")
    fit_predict_score(x_train_old, y_train, x_test_old, validation_data, rf)
    print("\nAtributos Novos\n")
    fit_predict_score(x_train, y_train, x_test, validation_data, rf)


    # # Improving results
    print("\n"+"-"*80+"\n")
    print("\tMelhorando os Resultados\n")
    
    # ## Usando RF com parametros ótimos
    print("\n1a busca por parâmetros.")
    print("Potênciais Parâmetros:")
    params_rf = {"n_estimators" : [100, 500, 1000],
                "max_depth" : [25, 40, 50, 60, 75, 100], 
                "min_samples_split" : [3, 5, 15, 20]}
    print(params_rf)
    print("Iniciando busca...")
    rf_search = GridSearchCV(RandomForestClassifier(n_jobs = 3, 
                                                    random_state = seed),
                            params_rf, 
                            cv = 5)

    rf_search.fit(x_train, y_train)
    print(f"\nMelhores Parâmetros: {rf_search.best_params_}")

    print("Plotando gráficos dos parâmetros.")
    cv_df = pd.DataFrame(rf_search.cv_results_)
    cv_df = cv_df[["param_max_depth","param_n_estimators","param_min_samples_split","mean_test_score"]]
    cv_df.groupby("param_max_depth").mean().plot()
    cv_df.groupby("param_n_estimators").mean().plot()
    cv_df.groupby("param_min_samples_split").mean().plot()

    
    print("Olhando as importâncias calculadas pela RF")
    print(pd.DataFrame(rf_search.best_estimator_.feature_importances_, index = x_test.columns))

    print("\nIniciando segunda busca por parametros, baseada nos resultados da primeira.")
    print("Potênciais Parâmetros:")
    params_rf = {"n_estimators" : [900, 1000, 1100, 1200],
                "max_depth" : [40, 50, 60], 
                "min_samples_split" : [2, 3]}
    print(params_rf)

    print("\nIniciando Busca")
    rf_search = GridSearchCV(RandomForestClassifier(n_jobs = 3, 
                                                    random_state = seed),
                            params_rf, 
                            cv = 5)

    rf_search.fit(x_train, y_train)
    print(f"\nMelhores Parâmetros: {rf_search.best_params_}")
    
    print("Plotando gráficos dos parâmetros.")
    cv_df = pd.DataFrame(rf_search.cv_results_)
    cv_df = cv_df[["param_max_depth","param_n_estimators","param_min_samples_split","mean_test_score"]]
    cv_df.groupby("param_max_depth").mean().plot()
    cv_df.groupby("param_n_estimators").mean().plot()
    cv_df.groupby("param_min_samples_split").mean().plot()

    print("Listando importância dos atributos conforme a RF")
    # Looking at feature importances
    print(pd.DataFrame(rf_search.best_estimator_.feature_importances_, index = x_test.columns))


    # ## Explorando Atributos
    print("Explorando diferentes conjuntos de atributos.")
    print("Conjuntos explorados:")
    
    metodos_de_extracao = {'basico_mfcc': MultiFeatureExtractor([BasicExtractor(n_fft=2048, hop_length=512),MFCCExtractor()]),
                        'basico4096_mfcc': MultiFeatureExtractor([BasicExtractor(n_fft=4096, hop_length=1024),MFCCExtractor()]),
                        'basico1024_mfcc': MultiFeatureExtractor([BasicExtractor(n_fft=1024, hop_length=256),MFCCExtractor()]),
                        'basico_mfcc10': MultiFeatureExtractor([BasicExtractor(),MFCCExtractor(10)]),
                        'basico_mfcc40': MultiFeatureExtractor([BasicExtractor(),MFCCExtractor(40)]),
                        'basico_mfcc80': MultiFeatureExtractor([BasicExtractor(),MFCCExtractor(80)]),
                        'tonnetz_mfcc': MultiFeatureExtractor([BasicExtractor(),MFCCExtractor(), TonnetzExtractor()]),
                        'tonnetz_mfcc40': MultiFeatureExtractor([BasicExtractor(),MFCCExtractor(40), TonnetzExtractor()]),
                        'tonnetz_mfcc80': MultiFeatureExtractor([BasicExtractor(),MFCCExtractor(80), TonnetzExtractor()]),
                        '4096tonnetz_mfcc': MultiFeatureExtractor([BasicExtractor(n_fft=4096, hop_length=1024),MFCCExtractor(), TonnetzExtractor()]),
                        '4096tonnetz_mfcc40': MultiFeatureExtractor([BasicExtractor(n_fft=4096, hop_length=1024),MFCCExtractor(40), TonnetzExtractor()]),
                        }

    print(metodos_de_extracao)

    print("Iremos usar o primeiro conjunto para fazer uma segunda busca por melhores parâmetros para a RF")
    
    params_rf = {"n_estimators" : [800, 1000, 1200, 1400, 1600],
                "max_depth" : [5, 10, 20, 40, 50], 
                "max_features": [0.3, 5, 10, "sqrt","log2", None]}

    print(params_rf)

    grid_searcher = GridSearchCV(
                        RandomForestClassifier(
                            random_state = seed, 
                            n_jobs = 3,
                            min_samples_split = 2,
                        ), 
                        params_rf, 
                        cv = 5, 
                        verbose=3)


    # ## Melhorando os Metaparâmetros da Floresta Aleatória
    parametros = dict() 
    cv_results = dict() 
    estimators = dict()


    # Testes Sem Reescala:
    x = x_train_raw
    y = y_train
    x_validacao = x_test_raw
    y_validacao = y_test
    validacao  = validation_data

    chave = "basico_mfcc"
    metodo = metodos_de_extracao[chave]

    print(f"Analisando Método {chave}.")
    print("Transformando Treino.")
    metodo.fit(x)
    features = metodo.transform(x)
    encoded_labels = y

    print(f"Encontrando melhor modelo para {chave}.")
    grid_searcher.fit(features, encoded_labels)

    print(f"Modelo para {chave} encontrado.")
    parametros[chave] = grid_searcher.best_params_
    cv_results[chave] = grid_searcher.cv_results_     
    estimators[chave] = grid_searcher.best_estimator_
    
    print(f"Analisando parametros encontrados com os atributos - {chave}")
    print(pd.DataFrame(cv_results['basico_mfcc'])[["mean_test_score", 'param_max_depth', 'param_max_features','param_n_estimators']])

    cv_df = pd.DataFrame(cv_results['basico_mfcc'])
    cv_df = cv_df[['param_max_depth', 'param_n_estimators', 'param_max_features',
        'mean_test_score']]
    cv_df.groupby("param_max_features").mean().plot()
    cv_df.groupby("param_n_estimators").mean().plot()
    cv_df.groupby("param_max_depth").mean().plot()

    print(cv_df.groupby("param_max_features").mean())
    
    chave = 'basico4096_mfcc'
    print(f"Avaliando outro conjunto de atributos: {chave}")
    metodo = metodos_de_extracao[chave]


    print("Terceira busca por metaparametros:")
    params_rf = {"n_estimators" : [600, 800, 900],
                "max_depth" : [18, 20, 22], 
                "max_features": [8, 10, 12]}

    print(params_rf)
    grid_searcher = GridSearchCV(RandomForestClassifier(
                                            n_jobs=3, 
                                            random_state = seed,
                                            min_samples_split = 2), 
                                params_rf, 
                                cv = 5, 
                                verbose=3)

    print(f"Transformando os dados({chave})")
    features = metodo.fit_transform(x)
    encoded_labels = y
        
    print(f"Encontrando melhor modelo para {chave}.")
    grid_searcher.fit(features, encoded_labels)
    parametros[chave] = grid_searcher.best_params_
    cv_results[chave] = grid_searcher.cv_results_     
    estimators[chave] = grid_searcher.best_estimator_

    print("Modelo encontrado. Plotando resultados:")
    cv_df2 = pd.DataFrame(cv_results["basico4096_mfcc"])
    cv_df2 = cv_df2[['param_max_depth', 'param_n_estimators', 'param_max_features',
        'mean_test_score']]

    cv_df2.groupby("param_max_features").mean().plot()
    cv_df2.groupby("param_n_estimators").mean().plot()
    cv_df2.groupby("param_max_depth").mean().plot()


    print("Antes de analisar outros conjuntos de atributos, verificar os efeitos da normalização:")
    chave = 'basico4096_mfcc'
    metodo = metodos_de_extracao[chave]
    algoritmo = grid_searcher.best_estimator_

    # Features não escaladas
    features4096 = metodo.fit_transform(x)

    print("Escalando features... MinMaxScaler")
    scaler = MinMaxScaler()
    scaler.fit(features4096)
    features_scaled = scaler.transform(features4096)
    encoded_labels = y

    scores = dict();
    chave = "basico4096_MinMaxScaler"
    print(f"Encontrando melhor modelo para {chave}.")
    scores[chave] = sklearn.model_selection.cross_val_score(algoritmo, features_scaled, encoded_labels, cv = 5)
    print(f"{chave} ({np.mean(scores[chave])}): {scores[chave]}")

    chave = "basico4096_StandardScaler"    
    scaler = StandardScaler()
    scaler.fit(features4096)
    print("Escalando features... StandardScaler")
    features_scaled = scaler.transform(features4096)
    encoded_labels = y
    print(f"Encontrando melhor modelo para {chave}.")
    scores[chave] = sklearn.model_selection.cross_val_score(algoritmo, features_scaled, encoded_labels, cv = 5)
    print(f"{chave} ({np.mean(scores[chave])}): {scores[chave]}")

    chave = "basico4096_NoScaler"    
    features_scaled = features4096
    encoded_labels = y
    print("Escalando features... Identidade (sem normalizacao)")
    print(f"Encontrando melhor modelo para {chave}.")
    scores[chave] = sklearn.model_selection.cross_val_score(algoritmo, features_scaled, encoded_labels, cv = 5)
    print(f"{chave} ({np.mean(scores[chave])}): {scores[chave]}")

    for chave, score in scores.items():
        print(f"{chave}: {100*np.mean(score):.4f}\n") 

    # Avaliando efeito da escala em outros algoritmos
    print("Verificando se em outros algoritmos há algum efeito significante usando o StandardScaler")
    print("Transformando X")
    features_scaled = scaler.transform(features4096)
    features_unscaled = features4096
    algoritmo = LinearDiscriminantAnalysis()
    print("Calculando efeitos no LDA")
    score_scaled = 100*np.mean(sklearn.model_selection.cross_val_score(algoritmo, features_scaled, encoded_labels, cv = 5))
    score_unscaled = 100*np.mean(sklearn.model_selection.cross_val_score(algoritmo, features_unscaled, encoded_labels, cv = 5))
                            
    print(f"LDA-Scaled: {score_scaled:.4f}")
    print(f"LDA-Not: {score_unscaled:.4f}")

    algoritmo = GaussianNB()
    print("Calculando efeitos no NB")
    score_scaled = 100*np.mean(sklearn.model_selection.cross_val_score(algoritmo, features_scaled, encoded_labels, cv = 5))
    score_unscaled = 100*np.mean(sklearn.model_selection.cross_val_score(algoritmo, features_unscaled, encoded_labels, cv = 5))
                
    print(f"NB-Scaled: {score_scaled:.4f}")
    print(f"NB-Not: {score_unscaled:.4f}")

    
    print("Testando Subconjuntos de atributos - sem normalização pois ela não se mostrou muito util/consistente")
    algoritmo = grid_searcher.best_estimator_
    scores_metodos = dict()
    for metodo in ['basico_mfcc','basico4096_mfcc','basico1024_mfcc', 'basico_mfcc10', 'basico_mfcc40', 'basico_mfcc80', 'tonnetz_mfcc', 'tonnetz_mfcc40', 'tonnetz_mfcc80']:
        print("\n\tAnalisando "+metodo)
        print("\t - Transformando os atributos")
        metodos_de_extracao[metodo].fit(x)
        features = metodos_de_extracao[metodo].transform(x)
        print("\t - Pontuando transformação")
        scores_metodos[metodo] = 100*np.mean(sklearn.model_selection.cross_val_score(algoritmo, features, encoded_labels, cv = 5))
    print("\n")

    print("Resultados da análise:")
    melhor = 0
    melhor_chave = ""
    for chave, valor in scores_metodos.items():
        print(f"{chave}: {valor :.2f}%")
        if(valor > melhor):
            melhor = valor
            melhor_chave = chave

    print("Calculando atributos usando o melhor método...")
    features = metodos_de_extracao[melhor_chave].fit_transform(x)
    
    print("Avaliando Comite de Votação: RF, NB, LDA")
    rf = RandomForestClassifier(random_state = seed, n_jobs = -1,
                                max_depth = 20,
                                max_features = 0.3,
                                n_estimators = 900)
    nb = GaussianNB()
    lda = LinearDiscriminantAnalysis()


    voting = VotingClassifier([('rf',rf),('nb', nb), ('lda',lda)], 
                            weights=[2,1,1], voting="hard")
    weights = list()
    weights.append([2,1,1])
    weights.append([3,2,1])
    weights.append([3,1,2])
    weights.append([4,2,2])
    weights.append([3,2,2])
    weights.append([1,1,1])
    param_grid = dict(weights = weights, voting= ["soft","hard"])

    print("Busca por parâmetros para o comitê: ")
    print(param_grid)
    print("\n")

    vote_grid = GridSearchCV(VotingClassifier([('rf',rf),('nb', nb), ('lda',lda)]),
                        param_grid = param_grid,
                        cv = 5)
    print("Realizando a busca...")
    vote_grid.fit(features, encoded_labels)
    print("\n --- \nResultados do comitê:")
    print(vote_grid.best_params_)
    votacao_cv = pd.DataFrame(vote_grid.cv_results_)
    votacao_cv = votacao_cv[votacao_cv.params == {'voting': 'hard', 'weights': [2, 1, 1]}]
    print(votacao_cv[["mean_test_score","std_test_score"]])

    print("Comparando com a floresta aleatória...")
    results_rf = sklearn.model_selection.cross_val_score(rf, features, encoded_labels, cv = 5)
    print(f"Floresta AleatóriaMean {100*np.mean(results_rf)} +- {100*np.std(results_rf)}")

    print("Fim da etapa de estudos. Limpando memória.")
    
    algoritmo = None
    best = None
    best_estimator = None
    encoded_labels = None
    estimators = None
    
    features4096 = None
    features_scaled = None
    features_unscaled = None
    grid_search = None
    grid_searcher = None
    nb = None
    score = None
    score_classifier = None
    score_scaled = None
    score_unscaled = None
    scores = None
    scores_metodos = None
    test_data = None
    training_data = None
    validacao = None
    validation_data = None
    validation_red = None
    x = None
    x_test_raw = None
    x_test_red = None
    x_train_raw = None
    x_train_red = None
    x_transformado = None
    x_validacao = None
    y = None
    y_test = None
    y_test_red = None
    y_train = None
    y_train_red = None
    y_validacao = None
    prepared_x = None
    prepared_x_dados = None
    prepared_x_mfcc = None
    preprocess_data = None
    features = None
    train_data = None
    test_data = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    x_train_raw = None
    x_test_raw = None
    x_train_old = None
    x_test_old = None
    print("Fim do processo de descoberta. Iniciando classificação.")




"""
===============================================================================
                FIM DO PROCESSO DE DESCOBERTA - CLASSIFICANDO
===============================================================================
""" 

print("Analise sendo realizada com o melhor classificador obtido.") 

    # ## Criando árvore com os melhores parametros
rf = RandomForestClassifier(random_state = seed, n_jobs = -1,
                                max_depth = 20,
                                max_features = 0.3,
                                n_estimators = 900)

resultado_treino, resultado_teste = process_folder(train_path, test_path, rf)
print("\n---- ---- ---- RESULTADOS ---- ---- ----\n")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(resultado_teste[[LABEL, ARQUIVO_ORIGINAL, PREDICAO]])

resultado_teste[[LABEL, ARQUIVO_ORIGINAL, PREDICAO]].to_csv(output_csv)




with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(resultado_treino[[LABEL, ARQUIVO_ORIGINAL, PREDICAO]])

if(output_csv):
    resultado_treino[[LABEL, ARQUIVO_ORIGINAL, PREDICAO]].to_csv(output_csv)
