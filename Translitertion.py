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
  test_path = 'te.translit.sampled.test.tsv'
  val_path = 'te.translit.sampled.dev.tsv'
  train_path = 'te.translit.sampled.train.tsv'
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

def get_acc(model,val_path,config,model2,details,list_L):
  [char_dict_telugu,char_dict_english,max_tel_len,max_eng_len,tel_vocab_length,eng_vocab_length,num_dict_telugu,config] = details
  tel_words = tel_words_valid
  eng_words = eng_words_valid
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
  word_embed = model2.predict(one_hot)
  one_hot2 = []
  for word in tel_words:
    i = 0
    L = np.zeros(max_tel_len)
    for char in word:
      L[i] = char_dict_telugu[char]
      i+=1
    while i < max_tel_len:
      L[i] = char_dict_telugu[" "]
      i+=1
    one_hot2.append(L)
  one_hot2 = np.array(one_hot2)
  output = predict(model,word_embed,tel_word_list,num_dict_telugu,eng_words,config,list_L)
  i = 0
  #print(output)
  for j in range(len(output)):
    L = output[j]
    lis = [tel_word_list[i][0]]
    lis.extend(list (L))
    lis.pop()
    L = np.array(lis)
    output[j] = L
    i+=1
  [max_acc,max_output] =  calc_acc(config.beam_size,output,one_hot2)
  return max_acc

def predict(model,word_embed,tel_word_list,num_dict_telugu,eng_words,config,list_L):
  [decoder_model,encoder_model,model3] = list_L
  output_list = np.zeros((tel_word_list.shape[0],tel_word_list.shape[1],tel_word_list.shape[2]))
  [output1,encoder_states] = encoder_model.predict(np.array(word_embed))
  target_seq = np.zeros((tel_word_list.shape[0],1,tel_word_list.shape[2]))
  for i in range(tel_word_list.shape[1]):
    [decoder_outputs,decoder_states] = decoder_model.predict([target_seq,output1] + encoder_states)
    encoder_states = decoder_states
    target_seq[:,0] = decoder_outputs[:,0]
    output_list[:,i] = decoder_outputs[:,0]
  s = ""
  for i in range(len(output_list[1])):
    s = s+(num_dict_telugu[np.argmax(output_list[1][i])])
  return output_list



def get_model(config,eng_vocab_length,tel_vocab_length,num_words):
  inputs = keras.Input(shape=(None,config.input_size))
  decoder_inputs = Input(shape=(None, tel_vocab_length))
  encoder_inputs = inputs
  enc_layers = config.encoder_layers
  dec_layers = config.decoder_layers
  outputs = encoder_inputs
  encoder_states = []
  gru_enc = []
  lstm_enc = []
  rnn_enc = []
  gru_dec = []
  lstm_dec = []
  rnn_dec = []
  for i in range(enc_layers):
    lstm_enc.append(LSTM(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout))
    gru_enc.append(GRU(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout))
    rnn_enc.append(SimpleRNN(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout))
  for i in range(dec_layers):
    lstm_dec.append(LSTM(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout))
    gru_dec.append(GRU(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout))
    rnn_dec.append(SimpleRNN(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout))
  for i in range(enc_layers):
    if config.cell == 'LSTM':
      outputs, h, c = (lstm_enc[i])(outputs)
      encoder_states+=[h,c]
    if config.cell == 'GRU':
      outputs, h = (gru_enc[i])(outputs)
      encoder_states+=[h]
    if config.cell == 'RNN':
      outputs, h = (rnn_enc[i])(outputs)
      encoder_states+=[h]
  s = len(encoder_states)
  output1 = outputs
  outputs = decoder_inputs 
  enc_op = []
  if (enc_layers <= dec_layers) :
    for i in range(dec_layers):
      if config.cell == 'LSTM':
        enc_op+= [encoder_states[min(2*i,s-2)],encoder_states[min(2*i+1,s-1)]]
      else:
        enc_op+=[encoder_states[min(i,s-1)]]
  else:
    for i in range(dec_layers):
      if config.cell == 'LSTM':
        enc_op+=[encoder_states[min(2*(enc_layers-dec_layers+i),s-2)],encoder_states[min(2*(enc_layers-dec_layers+i)+1,s-1)]]
      else:
        enc_op+=[encoder_states[min(enc_layers-dec_layers+i,s-1)]]
  if (enc_layers <= dec_layers) :
    for i in range(dec_layers):
      if config.cell == 'LSTM':
        outputs, h, c = (lstm_dec[i])(outputs,initial_state = [encoder_states[min(2*i,s-2)],encoder_states[min(2*i+1,s-1)]])
      if config.cell == 'GRU':
        outputs, h = (gru_dec[i])(outputs,initial_state = encoder_states[min(i,s-1)])
      if config.cell == 'RNN':
        outputs, h = (rnn_dec[i])(outputs,initial_state = encoder_states[min(i,s-1)])
  else:
    for i in range(dec_layers):
      if config.cell == 'LSTM':
        outputs, h, c = (lstm_dec[i])(outputs,initial_state = [encoder_states[min(2*(enc_layers-dec_layers+i),s-2)],encoder_states[min(2*(enc_layers-dec_layers+i)+1,s-1)]])
      if config.cell == 'GRU':
        outputs, h = (gru_dec[i])(outputs,initial_state = encoder_states[min(enc_layers-dec_layers+i,s-1)])
      if config.cell == 'RNN':
        outputs, h = (rnn_dec[i])(outputs,initial_state = encoder_states[min(enc_layers-dec_layers+i,s-1)])
  bn = BatchNormalization(momentum=0.6)
  if config.attention == 1:
    attention = dot([outputs, output1], axes=[2, 2])
    attention = Activation('softmax')(attention)
    context = dot([attention, output1], axes=[2,1])
    context = bn(context)
    decoder_combined_context = concatenate([context, outputs])
  else:
    decoder_combined_context = outputs
  dropout = Dropout(rate=config.dropout)
  decoder_outputs = dropout(decoder_combined_context)
  decoder_dense = Dense(tel_vocab_length, activation='softmax')
  decoder_outputs = decoder_dense(decoder_outputs)
  model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
  encoder_model = keras.Model(encoder_inputs,[output1,enc_op])
  decoder_ip_states = []
  for i in range(dec_layers):
    if config.cell == 'LSTM':
      decoder_ip_states += [keras.Input(shape=(config.hidden_size,)),keras.Input(shape=(config.hidden_size,))] 
    else:
      decoder_ip_states += [keras.Input(shape=(config.hidden_size,))] 
  output1 = keras.Input(shape = (None,config.hidden_size))
  outputs = decoder_inputs
  decoder_states = []
  for i in range(dec_layers):
    if config.cell == 'LSTM':
      outputs, h, c = (lstm_dec[i]) (outputs,initial_state = [decoder_ip_states[min(2*i,s-2)],decoder_ip_states[min(2*i+1,s-1)]])
      decoder_states+=[h,c]
    if config.cell == 'GRU':
      outputs, h = (gru_dec[i])(outputs,initial_state = decoder_ip_states[min(i,s-1)])
      decoder_states+=[h]
    if config.cell == 'RNN':
      outputs, h = (rnn_dec[i])(outputs,initial_state = decoder_ip_states[min(i,s-1)])
      decoder_states+=[h]
  decoder_combined_context = outputs
  decoder_outputs = dropout(decoder_combined_context)
  decoder_outputs = decoder_dense(decoder_outputs)
  decoder_model = keras.Model([decoder_inputs,output1] + decoder_ip_states, [decoder_outputs,decoder_states])
  return [decoder_model,encoder_model,model]
