
# Importings base libraries
import os;
import pandas as pd; 
import librosa;
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt


get_ipython().run_line_magic('matplotlib', 'inline')


# Input Variables and Setup
training_data_path   = os.path.join(".","dados","TREINAMENTO")
validation_data_path = os.path.join(".","dados","VALIDACAO")



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
    
def get_sampling_rate(file):
    return librosa.load(file, None)[1]

def get_labels(file):
    base = os.path.basename(file)
    without_extension = os.path.splitext(base)[0]
    return without_extension

def get_audio_and_sampling_rate_from_df(df, row_index, asNumpyArray = True ):
    data, sampling_rate, label = get_data_sampling_rate_and_legend_from_df(df, row_index, asNumpyArray)
    return(data, sampling_rate)

def get_data_sampling_rate_and_legend_from_df(df, row_index, asNumpyArray = True):
    data = df.iloc[ row_index , df.columns.get_loc(0): ].astype('float64')
    if(asNumpyArray):
        data = data.values
    sample_rate = df.loc[: , "sr"].iloc[row_index]
    label = df.loc[ : , 'label'].iloc[row_index]
    return(data, sample_rate, label)
    
    
    
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
    
    
    
def show_data(df, row):
    # Retrieve information from DF
    audio_data, sampling_rate, label = get_data_sampling_rate_and_legend_from_df(df, row)
    
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


# Reading files
f = get_files_from(training_data_path)



# Exploring the data, first file for example
example = f[0]

wave = load_wav_file(example)
sr = get_sampling_rate(example)
labels = get_labels(example)


# First File DataFrame (66ah)
df = join_onda_with_label(wave, labels, sr)


# First row data (the number 6)
data, sampling_rate = get_audio_and_sampling_rate_from_df(df, 1)


# Confirm audio duration 
librosa.get_duration(data, sr=sampling_rate)


# Retrieving audio from example
ipd.Audio(data, rate=sampling_rate)



# Plotting data (exploratory analysis)
X = librosa.stft(data)
Xdb = librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(8, 16), dpi= 80, facecolor='w', edgecolor='k')

plt.subplot(3, 1, 1)
plt.title("Wave")
librosa.display.waveplot(data, sr=sampling_rate, x_axis="time")

plt.subplot(3, 1, 2)
plt.title("MEL")
librosa.display.specshow(Xdb, sr=sampling_rate, x_axis="time", y_axis="mel")
