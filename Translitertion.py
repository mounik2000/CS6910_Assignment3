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

#Paths for training

test_path = 'te.translit.sampled.test.tsv'
val_path = 'te.translit.sampled.dev.tsv'
train_path = 'te.translit.sampled.train.tsv'

#To get all the words as lists and the dictionaries mapping letters to numbers

def get_words_list():
  #add space in the encoding maps
  char_dict_english = {' ': 0}
  char_dict_telugu = {' ': 0}
  num_dict_telugu = {0: ' '}
  #read from file
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
        #enclose the word in between tabs so that the start and end of the word is known
        tel_word = '\t'+tel_word+'\t'
        #maximum length of output word
        max_tel_len = max(len(tel_word),max_tel_len)
        eng_word = l[1]
        #enclose the word in between tabs so that the start and end of the word is known
        eng_word = '\t'+eng_word+'\t'
        max_eng_len = max(len(eng_word),max_eng_len)
        tel_words.append(tel_word)
        eng_words.append(eng_word)
        #update the dictionaries
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
  #reapeat got validation
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

#get the train, validation word lists
[tel_words_train,eng_words_train,char_dict_english,char_dict_telugu,num_dict_telugu,max_tel_len,max_eng_len,eng_words_valid,tel_words_valid] = List


def do_run(config):
  tel_words = tel_words_train
  eng_words = eng_words_train
  tel_vocab_length = len(char_dict_telugu)
  eng_vocab_length = len(char_dict_english)
  tel_word_list = []
  #get the output true array with shape (no of words, max length of word, vocab encoding for the word)
  # each row is a 2D vector in which each letter is represented by one hot encoding
  for word in tel_words:
    i = 0
    L = np.zeros((max_tel_len,tel_vocab_length))
    for char in word:
      L[i][char_dict_telugu[char]] = 1
      i+=1
    #If word is completed fill spaces at the end
    while i < max_tel_len:
      L[i][char_dict_telugu[" "]] = 1
      i+=1
    tel_word_list.append(L)
  tel_word_list = np.array(tel_word_list)
  #get training and inference models
  [decoder_model,encoder_model,model] = get_model(config,eng_vocab_length,tel_vocab_length,len(eng_words))
  output_list = []
  #FOr embedding
  model2 = Sequential()
  model2.add(Embedding(eng_vocab_length,config.input_size,input_length=max_eng_len))
  model2.compile('rmsprop','categorical_crossentropy')
  one_hot = []
  #get the input array with shape (no of words, max length of word), each row contains encoding of letter
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
  #get the embedding
  word_embed = model2.predict(one_hot)
  for word in tel_words:
    i = 0
    L = np.zeros((max_tel_len,tel_vocab_length))
    for char in word[1:]:
      L[i][char_dict_telugu[char]] = 1
      i+=1
    #If word is completed fill spaces at the end
    while i < max_tel_len:
      L[i][char_dict_telugu[" "]] = 1
      i+=1
    output_list.append(L)
  output_list = np.array(output_list)
  #compile and fit model
  model.compile(loss = 'categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
  h = model.fit([word_embed, tel_word_list],output_list,batch_size=32,epochs=20,shuffle = True,callbacks = [WandbCallback()])
  details = [char_dict_telugu,char_dict_english,max_tel_len,max_eng_len,tel_vocab_length,eng_vocab_length,num_dict_telugu,config]
  #get validation word accuracy using beam search and log it
  acc = get_acc(model,val_path,config,model2,details,[decoder_model,encoder_model,model])
  wandb.log({'val_word_accuracy': acc})
  return


#validation "word" accuracy calculation manually from inference model
def get_acc(model,val_path,config,model2,details,list_L):
  [char_dict_telugu,char_dict_english,max_tel_len,max_eng_len,tel_vocab_length,eng_vocab_length,num_dict_telugu,config] = details
  tel_word_list = []
  tel_words = tel_words_valid
  eng_words = eng_words_valid
  #get tel_list of words, eng_list, embedding of english list similar to that of training data
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
  #Here we are predicting ouput and calculating accuracy instead of training
  output = predict(model,word_embed,tel_word_list,num_dict_telugu,eng_words,config,list_L)
  i = 0
  for j in range(len(output)):
    L = output[j]
    #As prediction output is only from second character, we add first character
    lis = [tel_word_list[i][0]]
    lis.extend(list (L))
    lis.pop()
    L = np.array(lis)
    output[j] = L
    i+=1
  #use beam search and calculate accuracy between true and predicted words
  [max_acc,max_output] =  calc_acc(config.beam_size,output,one_hot2)
  return max_acc

#predict the model
def predict(model,word_embed,tel_word_list,num_dict_telugu,eng_words,config,list_L):
  [decoder_model,encoder_model,model3] = list_L
  output_list = np.zeros((tel_word_list.shape[0],tel_word_list.shape[1],tel_word_list.shape[2]))
  #get output states
  [output1,encoder_states] = encoder_model.predict(np.array(word_embed))
  #start with empty target sequence and predict the character in the next timestep everytime
  target_seq = np.zeros((tel_word_list.shape[0],1,tel_word_list.shape[2]))
  for i in range(tel_word_list.shape[1]):
    [decoder_outputs,decoder_states] = decoder_model.predict([target_seq,output1] + encoder_states)
    #update states and predictions at that timestep
    encoder_states = decoder_states
    target_seq[:,0] = decoder_outputs[:,0]
    output_list[:,i] = decoder_outputs[:,0]
  s = ""
  for i in range(len(output_list[1])):
    s = s+(num_dict_telugu[np.argmax(output_list[1][i])])
  return output_list



#Calculate word accuracy
def calc_acc(beam_size,output,one_hot2):
  max_acc2 = 0.0
  max_output2 = []
  for beam in range(beam_size):
    Result = []
    #compute accuracy for the outputs with top beam scores
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
	for row in data:
		all_candidates = list()
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				if (row[j]<=0):
					candidate = [seq + [j], score +50]
				else:
					candidate = [seq + [j], score - math.log(row[j])]
				all_candidates.append(candidate)
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		sequences = ordered[:k]
	return sequences


#Return the model with inference model also
def get_model(config,eng_vocab_length,tel_vocab_length,num_words):
  inputs = keras.Input(shape=(None,config.input_size))
  #inputs to encoder, decoder
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
  #add lstm,gru,rnn layers so that we can use one of them appropriately
  for i in range(enc_layers):
    lstm_enc.append(LSTM(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout))
    gru_enc.append(GRU(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout))
    rnn_enc.append(SimpleRNN(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout))
  for i in range(dec_layers):
    lstm_dec.append(LSTM(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout))
    gru_dec.append(GRU(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout))
    rnn_dec.append(SimpleRNN(config.hidden_size, return_state=True, return_sequences=True,dropout = config.dropout))
  #multilayer encoder
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
  #for inference model
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
  #multilayer decoder using the previously computed encoder states as initial states.
  #The below implementation uses as many encoder states as possible instead of final states of final encoder so that internal states also get good priorities
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

  #adding attention and batch normalization
  bn = BatchNormalization(momentum=0.6)
  if config.attention == 1:
    attention = dot([outputs, output1], axes=[2, 2])
    attention = Activation('softmax')(attention)
    context = dot([attention, output1], axes=[2,1])
    context = bn(context)
    decoder_combined_context = concatenate([context, outputs])
  else:
    decoder_combined_context = outputs
  #dropout and dense layers
  dropout = Dropout(rate=config.dropout)
  decoder_outputs = dropout(decoder_combined_context)
  decoder_dense = Dense(tel_vocab_length, activation='softmax')
  decoder_outputs = decoder_dense(decoder_outputs)
  #training model
  model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
  #for inference encoder model
  encoder_model = keras.Model(encoder_inputs,[output1,enc_op])
  decoder_ip_states = []
  #decoder inference model input states 
  for i in range(dec_layers):
    if config.cell == 'LSTM':
      decoder_ip_states += [keras.Input(shape=(config.hidden_size,)),keras.Input(shape=(config.hidden_size,))] 
    else:
      decoder_ip_states += [keras.Input(shape=(config.hidden_size,))] 
  output1 = keras.Input(shape = (None,config.hidden_size))
  outputs = decoder_inputs
  #below all are for inference decoder model outputs
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
  
  if config.attention == 1:
    attention = dot([outputs, output1], axes=[2, 2])
    attention = Activation('softmax')(attention)
    context = dot([attention, output1], axes=[2,1])
    context = bn(context)
    decoder_combined_context = concatenate([context, outputs])
  else:
    decoder_combined_context = outputs
  decoder_outputs = dropout(decoder_combined_context)
  decoder_outputs = decoder_dense(decoder_outputs)
  #getting decoder inference model
  decoder_model = keras.Model([decoder_inputs,output1] + decoder_ip_states, [decoder_outputs,decoder_states])
  #returning all models
  return [decoder_model,encoder_model,model]

#sweep-config

sweep_config = {
    'method' : 'random',
    'metric': {
      'name': 'val_word_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'attention' : {
            'values' : [0,1]
        },
        'input_size' : {
            'values' : [8,16,32,64,128]
        },
        'hidden_size' : {
            'values' : [32,64,128,256]
        },
        'cell' : {
            'values' : ['LSTM','GRU','RNN']
        },
        'decoder_layers' : {
            'values' : [1,2,3]
        },
        'encoder_layers' : {
            'values' : [1,2,3]
        },
        'dropout' : {
            'values' : [0.2,0.3,0.4]
        },
        'beam_size' : {
            'values' : [1,2,3]
        }
    },
}

#get sweep id
sweep_id = wandb.sweep(sweep_config, entity="mounik2000", project="A3 Training")


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

#run the agent
wandb.agent(sweep_id, sweep_train)
