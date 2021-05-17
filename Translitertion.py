import keras
import numpy as np
from tensorflow.keras import optimizers
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten, Embedding, BatchNormalization, Activation, concatenate, dot
from keras.layers import LSTM,RNN,GRU,SimpleRNN
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow as tf
import math
import random
import wandb
from wandb.keras import WandbCallback
from io import BytesIO
import os

test_path = 'te.translit.sampled.test.tsv'
val_path = 'te.translit.sampled.dev.tsv'
train_path = 'te.translit.sampled.train.tsv'
def get_words_list():
  char_dict_english = {' ': 0}
  char_dict_telugu = {' ': 0}
  num_dict_telugu = {0: ' '}
  fd = open(train_path, 'r')
  num1 = 1
  num2 = 1
  tel_words = []
  eng_words = []
  max_tel_len = 0
  max_eng_len = 0
  while True:
      string = fd.readline()
      if not string:
          break
      else:
        l = string.split()
        tel_word = l[0]
        tel_word = '\t'+tel_word+'\t'
        max_tel_len = max(len(tel_word),max_tel_len)
        eng_word = l[1]
        eng_word = '\t'+eng_word+'\t'
        max_eng_len = max(len(eng_word),max_eng_len)
        tel_words.append(tel_word)
        eng_words.append(eng_word)
        for char in tel_word:
          if char not in char_dict_telugu:
            char_dict_telugu[char] = num1
            num_dict_telugu[num1] = char
            num1+=1
        for char in eng_word:
          if char not in char_dict_english:
            char_dict_english[char] = num2
            num2+=1
  fd.close()
  L = [tel_words,eng_words,char_dict_english,char_dict_telugu,num_dict_telugu,max_tel_len,max_eng_len]
  fd = open(val_path, 'r')
  num1 = 1
  num2 = 1
  tel_words = []
  eng_words = []
  while True:
      string = fd.readline()
      if not string:
          break
      else:
        l = string.split()
        tel_word = l[0]
        eng_word = l[1]
        tel_word = '\t'+tel_word+'\t'
        eng_word = '\t'+eng_word+'\t'
        tel_words.append(tel_word)
        eng_words.append(eng_word)
  fd.close()
  L.extend([eng_words,tel_words])
  return L
List = get_words_list()
[tel_words_train,eng_words_train,char_dict_english,char_dict_telugu,num_dict_telugu,max_tel_len,max_eng_len,eng_words_valid,tel_words_valid] = List

def do_run(config):
  test_path = '/content/drive/MyDrive/dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.test.tsv'
  val_path = '/content/drive/MyDrive/dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.dev.tsv'
  train_path = '/content/drive/MyDrive/dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.train.tsv'
  tel_words = tel_words_train
  eng_words = eng_words_train
  tel_vocab_length = len(char_dict_telugu)
  eng_vocab_length = len(char_dict_english)
  tel_word_list = []
  for word in tel_words:
    i = 0
    L = np.zeros((max_tel_len,tel_vocab_length))
    for char in word:
      L[i][char_dict_telugu[char]] = 1
      i+=1
    while i < max_tel_len:
      L[i][char_dict_telugu[" "]] = 1
      i+=1
    tel_word_list.append(L)
  tel_word_list = np.array(tel_word_list)
  [decoder_model,encoder_model,model] = get_model(config,eng_vocab_length,tel_vocab_length,len(eng_words))
  output_list = []
  one_hot = []
  for word in eng_words:
    i = 0
    L = np.zeros(max_eng_len)
    for char in word:
      L[i] = char_dict_english[char]
      i+=1
    while i < max_eng_len:
      L[i] = char_dict_english[" "]
      i+=1
    one_hot.append(L)
  one_hot = np.array(one_hot)
  word_embed = one_hot
  for word in tel_words:
    i = 0
    L = np.zeros((max_tel_len,tel_vocab_length))
    for char in word[1:]:
      L[i][char_dict_telugu[char]] = 1
      i+=1
    while i < max_tel_len:
      L[i][char_dict_telugu[" "]] = 1
      i+=1
    output_list.append(L)
  output_list = np.array(output_list)
  model.compile(loss = 'categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
  h = model.fit([word_embed, tel_word_list],output_list,batch_size=32,epochs=20,shuffle = True,callbacks = [WandbCallback()])
  details = [char_dict_telugu,char_dict_english,max_tel_len,max_eng_len,tel_vocab_length,eng_vocab_length,num_dict_telugu,config]
  acc = get_acc(model,val_path,config,model2,details,[decoder_model,encoder_model,model])
  wandb.log({'val_word_accuracy': acc})
  return
  
def get_model(config,eng_vocab_length,tel_vocab_length,num_words):
  inputs = keras.Input(shape=(None,config.input_size))
  decoder_inputs = Input(shape=(None, tel_vocab_length))
  encoder_inputs = inputs
  enc_layers = config.encoder_layers
  dec_layers = config.decoder_layers
  outputs = encoder_inputs
  encoder_states = []
  if config.cell == 'LSTM':
    outputs, h, c = LSTM(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout)(outputs)
    encoder_states+=[h,c]
  s = len(encoder_states)
  output1 = outputs
  outputs = decoder_inputs 
  if config.cell == 'LSTM':
    outputs, h, c = LSTM(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout)(outputs,initial_state = [encoder_states[min(2*i,s-2)],encoder_states[min(2*i+1,s-1)]])
  decoder_combined_context = outputs
  dropout = Dropout(rate=config.dropout)
  decoder_outputs = dropout(decoder_combined_context)
  decoder_dense = Dense(tel_vocab_length, activation='softmax')
  decoder_outputs = decoder_dense(decoder_outputs)
  model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
  return model
