# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 08:01:51 2024

@author: andry
"""

# Importando as Bibliotecas

import numpy as np
import pandas as pd
import random
import keras

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.applications import VGG16
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


#===============================================================================

def get_VGG16_model_Keras(input_shape=(100,100,3)) :

    VGGmodel = Sequential()
    baseModel = VGG16(
		input_shape=input_shape,
		weights='imagenet',
		include_top=False,
	)
    baseModel.trainable = False
    VGGmodel.add(baseModel)
    VGGmodel.add(Flatten())
    VGGmodel.add(Dense(512,activation = 'relu'))
    VGGmodel.add(Dropout(0.3))
    VGGmodel.add(Dense(10,activation = 'softmax'))
    
    return(VGGmodel)


#===============================================================================

def BlockSplit(dataframe, seed, n_songs):


  random.seed(seed)
  n_block = round(dataframe.shape[0] / n_songs)
  songs_per_genre = round(n_songs/10)
  n_test = round(songs_per_genre * 0.1)

  #used_list = []
  range_list = []
  block_lists = []

  for i in range(10):
    range_list.append(range(i*songs_per_genre, (i+1)*songs_per_genre))

  for k in range(10):
    test_list = []
    definitive_test_list = []

    for i in range(10):
      r = random.sample(range_list[i], n_test)
      # print(range_list[i])
      range_list[i] = [element for element in range_list[i] if element not in r]

      for j in r:
        test_list.append(j)

    test_list.sort()
    # print(test_list)
    for i in test_list:
      for j in range(n_block):
        definitive_test_list.append(round((i*n_block)+j))

    # print(definitive_test_list)
    block_lists.append(definitive_test_list)

  return block_lists

#===============================================================================

def trainVGG16(df, n_songs):

    all_performances = []
    # all_predictions  = []

    for seed in range(0, 10):#2):

        print("############################")
        print(" * Running for seed = ", seed)
        print("############################")
		# ----------------------------
		# Set the seed using keras.utils.set_random_seed. This will set:
		# 1) `numpy` seed
		# 2) `tensorflow` random seed
		# 3) `python` random seed
		# ----------------------------
        keras.utils.set_random_seed(seed)
        test_list = BlockSplit(dataframe = df, seed = seed, n_songs=n_songs)
        
        #Resetting the predictions
        conf_matrix = np.zeros((10,10))

# ----------------------------
# Running for each fold
# ----------------------------

        temp_acc = []
        split = 0

        for test in test_list:

            print("=============================")
            print(f" * Running for split = {split} | seed = {seed}")
            print("============================")            

      		# ----------------------------
      		# Defining the prediction model (CNN)
      		# ----------------------------
      
            print(" - defining VGG16 model")
            model = get_VGG16_model_Keras(input_shape=(100,100,3))
            model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])
      
      		# ----------------------------
      		# Traninig the algorithm
      		# ----------------------------
      
      		# Callbacks
            early_stopper = EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=1)
            csv_logger    = CSVLogger(f"C:/Users/andry/OneDrive/Documentos/GitHub/MGR-IC/ft2/5s/results/vgg16Logs/log_history_vgg16_seed_{seed}.csv", separator=",", append=False)
      
            print(" - training VGG16")
            X = np.array(df['spectogram'].iloc[~df.index.isin(test)].tolist())
            y = np.array(df['Classe'].iloc[~df.index.isin(test)].tolist())
            X_test = np.array(df['spectogram'].iloc[df.index.isin(test)].tolist())
            y_test = np.array(df['Classe'].iloc[df.index.isin(test)].tolist())
            # print(X.shape)
            # print(y.shape)
      
            # history  = model.fit(X, y, epochs=20, validation_split=0.1, batch_size=32, callbacks=[early_stopper, csv_logger])
            model.fit(X, y, epochs=20, validation_split=0.1, batch_size=64, callbacks=[early_stopper, csv_logger])

      		# ----------------------------
      		# Evaluating predictions
      		# ----------------------------
            print(" - Evaluating VGG16")
            predictions = model.predict(X_test)
            rounded_predictions = np.argmax(predictions, axis=1)
      
      		# ----------------------------
      		# adding predictions to a data frame
      		# ----------------------------
      		# preds   = pd.DataFrame(rounded_predictions, index = df_testing.index)
      		# preds   = preds.rename(columns={0: 'predictions'})
      		# df_pred = pd.concat([df_testing, preds], axis = 1) # by column
      
      		# ----------------------------
      		# evaluating with scikit learn metrics
      		# ----------------------------
      		# acc = accuracy_score(test_labels, rounded_predictions)
            bac = balanced_accuracy_score(np.argmax(y_test, axis=1), rounded_predictions)
      		# f1s = f1_score(test_labels, rounded_predictions)
      		# print("acc = ", acc)
            print("bac = ", bac)
      		# print("f1c = ", f1s)
            temp_acc.append(bac)
            
            #-------------------------------
            # getting the confusion matrix
            #-------------------------------
            conf_matrix += confusion_matrix(np.argmax(y_test, axis=1), rounded_predictions)
            
            split += 1
            
        print("----------------------------")
        all_performances.append([np.mean(temp_acc), seed, conf_matrix])
		# all_predictions.append(df_pred)

	# ---------------------------------------------------------
	# Binding all predictions and performances
	# ---------------------------------------------------------
	# pred_results = pd.concat(all_predictions, axis = 0) # by row
	# pred_results[['algo']] = "VGG16"

    perf_results = pd.DataFrame(all_performances, columns=["bac", "seed", "conf_matrix"])
    return perf_results


#================================================================================

if __name__ == "__main__":

    df = pd.read_pickle('cnn_spectrograms.pkl')

    perf_results = trainVGG16(df=df, n_songs = 100)
    	
    perf_results.to_pickle('performances_vgg16.pkl', index = False)
    print("Done!")


