
def extract_features(df):
    features_df = df.iloc[ : , df.columns.get_loc(0): ]
    sample_rate = df.loc[ : , 'sr']
    
    mfcc_dict = dict()
    centroid_dict = dict()
    bandwidth_dict = dict()
    flatness_dict = dict()
    rolloff_dict = dict()
    tonnetz_dict = dict()
    zero_crossing_dict = dict()
    
    for index, row in features_df.iterrows():
        #print(f"Analysing {index} line.")
        audio_data = row.values.astype('float64')
        sr = sample_rate[index]
        mfcc_dict[index] = np.mean(
                                librosa.feature.mfcc(y = audio_data,
                                                     sr=sr, 
                                                     n_mfcc=20).T,  # Transposta do mfcc
                                axis=0)
        centroid_dict[index] = np.mean(
                                librosa.feature.spectral_centroid(y = audio_data,
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
        
        tonnetz_dict[index]   = np.mean(
                                    librosa.feature.tonnetz(y = audio_data, sr = sr).T,
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
    
    tonnetz_df = pd.DataFrame.from_dict(tonnetz_dict, orient = "index")
    tonnetz_df.columns = ["tonnetz_"+ str(col)for col in tonnetz_df.columns]
    
    zero_crossing_df = pd.DataFrame.from_dict(zero_crossing_dict, orient = "index", columns=["zero_crossing"])

    return_df = pd.concat([zero_crossing_df, rolloff_df, flatness_df, bandwidth_df, centroid_df, mfcc_df, tonnetz_df], axis = 1)
    return return_df

