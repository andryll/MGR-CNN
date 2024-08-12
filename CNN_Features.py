# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 07:15:25 2024

@author: andry
"""

# Importando as Bibliotecas
import numpy as np
import librosa
import os
import pandas as pd
from keras.utils import to_categorical

import librosa.display
import matplotlib.pyplot as plt
import io
from PIL import Image

#--------------------------------------------------------------
#--------------------------------------------------------------

def splitSongs (songList, duration, sr=44100):

  # Converta o tamanho da janela de segundos para amostras
  window_size_samples = int(duration * sr)

  # Inicialize uma lista para armazenar os segmentos
  segmentedList = []

  # Pega cada música da lista
  for y in songList:

    # Calcule o número total de segmentos
    num_segments = len(y[0]) // window_size_samples

    # Divida o áudio em segmentos de 5 segundos e adcione-os na lista
    for i in range(num_segments):
        start = i * window_size_samples
        end = (i + 1) * window_size_samples
        segment = (y[0][start:end], y[1], y[2])
        segmentedList.append(segment)

  # Retorna a nova lista
  return segmentedList

#--------------------------------------------------------------
#--------------------------------------------------------------


def readSongs (genre, numSongs, sr=44100, duration = 30):

  # Declarando listas iniciais
  songs = []
  genrelist = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
                'metal', 'pop', 'reggae', 'rock']

  # Se a escolha de gênero não for 'all', substitui a lista pelo gênero escolhido
  if genre != 'all':
    genrelist = [genre]

  #Percorre todos os gêneros da lista
  for g in genrelist:
    #Pega o caminho para a pasta do gênero escolhido
    dir_path = os.path.join('songs', g)
    #Lista os arquivos da pasta e os embaralha
    files = os.listdir(dir_path)
    files.sort

    # Até o número de musicas desejado ser alcançado, lê os arquivos de áudio com o librosa
    for i in range(numSongs):
      songs.append(librosa.load(os.path.join(dir_path, files[i]), sr=sr, mono = True, duration = 30))
      # Adciona o gênero como uma variável da tupla
      songs[-1] = songs[-1] + (g,)

  max_len = max(len(song[0]) for song in songs)

  # Garante que todas as músicas terão o mesmo tamanho da maior
  resized_songs = []
  for song in songs:
      # Verifica se a música precisa ser redimensionada
      if len(song[0]) < max_len:
          # Adiciona zeros à direita para igualar o tamanho
          padded_audio = librosa.util.pad_center(data = song[0], size = max_len, axis = 0)
          resized_songs.append((padded_audio, song[1], song[2]))
      else:
          resized_songs.append(song)

  new_songs = splitSongs (resized_songs, sr=sr, duration = duration)

  return new_songs

#--------------------------------------------------------------
#--------------------------------------------------------------

def featureExtraction (songs, sr=44100):

  colunas = ['spectogram', 'Classe']

  df = pd.DataFrame(columns=colunas)

  for y in songs:

    S = librosa.feature.melspectrogram(y=y[0], sr=y[1])
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Criar uma figura do matplotlib para o espectrograma
    dpi=100
    size = 254
    figsize = (size/dpi, size/dpi)
    plt.figure(figsize=figsize, dpi=dpi)
    librosa.display.specshow(S_dB, sr=y[1])
    plt.tight_layout()

    # Salvar o espectrograma em um buffer de memória
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    img_array = np.array(img)/255

    #Adcionando ao DF

    df.loc[len(df)] = [img_array, y[2]]

  return df

#--------------------------------------------------------------
#--------------------------------------------------------------

if __name__ == "__main__":
    #Reading all musics and splitting
    songs = readSongs ('all', 100, 44100, 5)
    print("Songs readed!")
    #Getting the spectrograms
    df = featureExtraction (songs, sr=44100)
    #Converting the classes to numerical values
    df['Classe'] = pd.factorize(df['Classe'])[0]
    #Converting the classes to categorical values
    df['Classe'] = df['Classe'].apply(lambda x: to_categorical(x, num_classes=10))

    print(df['spectogram'][0].shape)

    #Saving the data
    df.to_pickle('cnn_spectrograms.pkl')

    print("Done!")

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
