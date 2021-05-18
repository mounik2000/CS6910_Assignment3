import keras
import numpy as np
from tensorflow.keras import optimizers
from matplotlib.font_manager import FontProperties
from keras.layers import Dense, Dropout, Flatten, Embedding, BatchNormalization, Activation, concatenate, dot
from keras.layers import LSTM,RNN,GRU,SimpleRNN
from keras.models import Sequential,Input,Model
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow as tf
import math
import plotly.graph_objs as go
import random
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from wandb.keras import WandbCallback
from io import BytesIO
from matplotlib import font_manager as fm, rcParams
import os
import pandas as pd
import matplotlib

def do_run(config):
  test_path = '/content/drive/MyDrive/te.translit.sampled.test.tsv'
  val_path = '/content/drive/MyDrive/te.translit.sampled.dev.tsv'
  train_path = '/content/drive/MyDrive/te.translit.sampled.train.tsv'
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
  model2 = Sequential()
  model2.add(Embedding(eng_vocab_length,config.input_size,input_length=max_eng_len))
  model2.compile('rmsprop','categorical_crossentropy')
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
  h = model.fit([word_embed, tel_word_list],output_list,batch_size=32,epochs=1,shuffle = True,callbacks = [WandbCallback()])
  details = [char_dict_telugu,char_dict_english,max_tel_len,max_eng_len,tel_vocab_length,eng_vocab_length,num_dict_telugu,config]
  acc = get_acc(model,val_path,config,model2,details,[decoder_model,encoder_model,model])
  wandb.log({'test_accuracy': acc})
  return

def get_acc(model,val_path,config,model2,details,list_L):
  [char_dict_telugu,char_dict_english,max_tel_len,max_eng_len,tel_vocab_length,eng_vocab_length,num_dict_telugu,config] = details
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
  output = predict(model,word_embed,tel_word_list,num_dict_telugu,eng_words,config,list_L,char_dict_english,tel_words)
  i = 0
  for j in range(len(output)):
    L = output[j]
    lis = [tel_word_list[i][0]]
    lis.extend(list (L))
    lis.pop()
    L = np.array(lis)
    output[j] = L
    i+=1
  [max_acc,max_output] =  calc_acc(config.beam_size,output,one_hot2)
  conf_output = np.array(max_output).reshape((-1,))
  conf_true = np.array(one_hot2).reshape((-1,))
  L = []
  for i in range(tel_word_list.shape[2]):
    L.append(num_dict_telugu[i])
  to_dump = []
  column_values = ['English word', 'Telugu true word', 'Telugu predicted word']
  tel_preds = []
  cols = []
  for i in range(len(max_output)):
    s1 = eng_words[i].strip()
    s2 = tel_words[i].strip()
    L = [s1,s2]
    s = ""
    for j in max_output[i]:
      s+=num_dict_telugu[j]
    s = s.strip()
    L.append(s)
    cols.append(L)
  cols = np.array(cols)
  df = pd.DataFrame(data = cols,columns = column_values)
  op = df.to_markdown(tablefmt="grid")
  op2 = df.to_csv()
  f = open("predictions_attention.md", "w")
  f.write(op)
  f.close()
  f = open("predictions_attention.csv", "w", encoding='utf-8')
  f.write(op2)
  f.close()
  return max_acc

def predict(model,word_embed,tel_word_list,num_dict_telugu,eng_words,config,list_L,char_dict_english,tel_words):
  [decoder_model,encoder_model,model3] = list_L
  output_list = np.zeros((tel_word_list.shape[0],tel_word_list.shape[1],tel_word_list.shape[2]))
  [output1,encoder_states] = encoder_model.predict(np.array(word_embed))
  target_seq = np.zeros((tel_word_list.shape[0],1,tel_word_list.shape[2]))
  for i in range(tel_word_list.shape[1]):
    [decoder_outputs,decoder_states,L] = decoder_model.predict([target_seq,output1] + encoder_states)
    encoder_states = decoder_states
    target_seq[:,0] = decoder_outputs[:,0]
    output_list[:,i] = decoder_outputs[:,0]
  return output_list

def calc_acc(beam_size,output,one_hot2):
  max_acc = 0.0
  max_output = []
  for beam in range(beam_size):
    Result = []
    for i in range(len(output)):
      seq = beam_search_decoder(output[i], beam_size)
      Result.append(seq[beam][0])
    acc = 0.0
    for i in range(one_hot2.shape[0]):
      acc2 = 0.0
      c = 0
      for j in range(one_hot2.shape[1]):
        p = Result[i][j]
        if (p == one_hot2[i][j] and p > 0) :
          acc2+= 1
        if (not (one_hot2[i][j] == 0)):
          c+=1
      acc2/=c
      acc+=acc2
    acc /= one_hot2.shape[0]
    if max_acc <= acc:
      max_acc = acc
      max_output = Result
  print("acc = "+str(max_acc))
  max_acc2 = 0.0
  max_output2 = []
  for beam in range(beam_size):
    Result = []
    for i in range(len(output)):
      seq = beam_search_decoder(output[i], beam_size)
      Result.append(seq[beam][0])
    acc = 0.0
    for i in range(one_hot2.shape[0]):
      c = 0
      for j in range(one_hot2.shape[1]):
        p = Result[i][j]
        if (p == one_hot2[i][j]) :
          c+=1
      if (c == one_hot2.shape[1]):
        acc+=1
    acc /= one_hot2.shape[0]
    if max_acc2 <= acc:
      max_acc2 = acc
      max_output2 = Result
  print("word acc = "+str(max_acc2))
  return [max_acc2,max_output2]


# Directly taken from https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
def beam_search_decoder(data, k):
	sequences = [[list(), 0.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				if (row[j]<=0):
					candidate = [seq + [j], score +50]
				else:
					candidate = [seq + [j], score - math.log(row[j])]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
	return sequences

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
  L = []
  if config.attention == 1:
    attention = dot([outputs, output1], axes=[2, 2])
    attention = Activation('softmax')(attention)
    L = attention
    context = dot([attention, output1], axes=[2,1])
    context = bn(context)
    decoder_combined_context = concatenate([context, outputs])
  else:
    decoder_combined_context = outputs
  decoder_outputs = dropout(decoder_combined_context)
  decoder_outputs = decoder_dense(decoder_outputs)
  decoder_model = keras.Model([decoder_inputs,output1] + decoder_ip_states, [decoder_outputs,decoder_states,L])
  return [decoder_model,encoder_model,model]

sweep_config = {
    'method' : 'grid',
    'metric': {
      'name': 'test_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'attention' : {
            'values' : [1]
        },
        'input_size' : {
            'values' : [128]
        },
        'hidden_size' : {
            'values' : [256]
        },
        'cell' : {
            'values' : ['LSTM']
        },
        'decoder_layers' : {
            'values' : [2]
        },
        'encoder_layers' : {
            'values' : [3]
        },
        'dropout' : {
            'values' : [0.2]
        },
        'beam_size' : {
            'values' : [2]
        }
    },
}

sweep_id = wandb.sweep(sweep_config, entity="mounik2000", project="A3 TestTime with Attention")

def sweep_train():
  config_defaults = {
        'dropout' : 0.2,
        'beam_size': 1,
        'encoder_layers': 1,
        'cell': 'RNN',
        'decoder_layers' : 1,
        'input_size' : 64,
        'attention' : 0,
        'hidden_size': 128
  }
  run = wandb.init(config = config_defaults)
  config = wandb.config
  do_run(config)

wandb.agent(sweep_id, sweep_train)