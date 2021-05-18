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

import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np

from IPython.core.display import display, HTML

#display characters as cmaps, used for visualizations (Red CMAP for visualization)
def format_chars(chars,numbers,character,number):
    numbers = np.array(numbers).astype(float)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.Reds
    colors = cmap(norm(numbers))
    hexcolor = [mcolors.to_hex(c) for c in colors]
    letter = lambda x: "<span style='color:{};'>{}</span>".format(x[1],x[0])
    #letter = lambda (c,l): "<span style='color:{};'>{}</span>".format(l,c)
    text = " ".join(list(map(letter, zip(chars,hexcolor))))
    text = "<div style='font-size:14pt;font-weight:italic;'> " + '<pre>'+"Output Charater "+str(number)+": "+character+', Input Visualization: ' + text + "</div>"
    display(HTML(text))
    return colors

#training similar to that of transliteration.py

def do_run(config):
  test_path = 'te.translit.sampled.test.tsv'
  val_path = 'te.translit.sampled.dev.tsv'
  train_path = 'te.translit.sampled.train.tsv'
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
  h = model.fit([word_embed, tel_word_list],output_list,batch_size=32,epochs=20,shuffle = True,callbacks = [WandbCallback()])
  details = [char_dict_telugu,char_dict_english,max_tel_len,max_eng_len,tel_vocab_length,eng_vocab_length,num_dict_telugu,config]
  acc = get_acc(model,val_path,config,model2,details,[decoder_model,encoder_model,model])
  wandb.log({'test_accuracy': acc})
  return


#drawing confusion matrix. Same code copied from Assignment 1
def draw_confusion_matrix(y_pred, y_true, classes) :
  conf_matrix = confusion_matrix(y_pred, y_true, labels=range(len(classes)))
  #getting diagonal elements
  conf_diagonal_matrix = np.eye(len(conf_matrix)) * conf_matrix
  np.fill_diagonal(conf_matrix, 0)
  conf_matrix = conf_matrix.astype('float')
  n_confused = np.sum(conf_matrix)
  conf_matrix[conf_matrix == 0] = np.nan
  #giving red shades to non diagonal elements
  conf_matrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': classes, 'y': classes, 'z': conf_matrix, 'hoverongaps':False, 'hovertemplate': 'Predicted %{y}<br>Instead of %{x}<br>On %{z} examples<extra></extra>'})
  conf_diagonal_matrix = conf_diagonal_matrix.astype('float')
  n_right = np.sum(conf_diagonal_matrix)
  conf_diagonal_matrix[conf_diagonal_matrix == 0] = np.nan
  #giving green shade to diagonal elements
  conf_diagonal_matrix = go.Heatmap({'coloraxis': 'coloraxis2', 'x': classes, 'y': classes, 'z': conf_diagonal_matrix,'hoverongaps':False, 'hovertemplate': 'Predicted %{y} just right<br>On %{z} examples<extra></extra>'})
  fig = go.Figure((conf_diagonal_matrix, conf_matrix))
  transparent = 'rgba(0, 0, 0, 0)'
  n_total = n_right + n_confused
  fig.update_layout({'coloraxis1': {'colorscale': [[0, transparent], [0, 'rgba(180, 0, 0, 0.05)'], [1, f'rgba(180, 0, 0, {max(0.2, (n_confused/n_total) ** 0.5)})']], 'showscale': False}})
  fig.update_layout({'coloraxis2': {'colorscale': [[0, transparent], [0, f'rgba(0, 180, 0, {min(0.8, (n_right/n_total) ** 2)})'], [1, 'rgba(0, 180, 0, 1)']], 'showscale': False}})
  xaxis = {'title':{'text':'y_true'}, 'showticklabels':False}
  yaxis = {'title':{'text':'y_pred'}, 'showticklabels':False}
  fig.update_layout(title={'text':'Confusion matrix', 'x':0.5}, paper_bgcolor=transparent, plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)
  wandb.log({'Heat Map Confusion Matrix': wandb.data_types.Plotly(fig)})
  return 0


#calculation of accuracy -> similar to transliteration with test path given as an argument as we need test accuracy
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
  #from here we print confusion matrices
  conf_output = np.array(max_output).reshape((-1,))
  conf_true = np.array(one_hot2).reshape((-1,))
  L = []
  for i in range(tel_word_list.shape[2]):
    L.append(num_dict_telugu[i])
  q = draw_confusion_matrix(conf_output, conf_true,L)
  wandb.log({"Confusion Matrix" : wandb.plot.confusion_matrix(preds=conf_output,y_true=conf_true,class_names=L)})
  print(max_acc)
  #write all predictions to a csv and markdown file
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
  f = open("predictions.md", "w")
  f.write(op)
  f.close()
  f = open("predictions.csv", "w", encoding='utf-8')
  f.write(op2)
  f.close()
  return max_acc


#predict the model
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
  #This is the function to draw heatmaps and visualizations
  get_heatmap(model,word_embed,tel_word_list,num_dict_telugu,eng_words,config,list_L,char_dict_english,output_list,tel_words)
  return output_list

#plottindg attention heatmaps using matplotlib (GREY CMAP for heatmaps)
def plot_attention(attention,predicted_sentence,sentence,it):
  fname='Gidugu.ttf'
  myfont=fm.FontProperties(fname=fname)
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='gray')
  fontdict = {'fontsize': 14,'fontweight' : 'bold' }
  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict,fontproperties = myfont)
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  plt.savefig('Heatmap of Input '+str(it+1)+'.png')


def get_heatmap(model,word_embed,tel_word_list,num_dict_telugu,eng_words,config,list_L,char_dict_english,output_list,tel_words):
  if config.attention == 0:
    return	
  #if attention is 1, take some random samples and plot attention heatmaps
  [decoder_model,encoder_model,model3] = list_L
  samples = [100,200,300,400,600,1200,1600,1800,2000,2400]
  samples.extend([624,690,835,1068,1559,1733,1933,2227,2330,2627,2678,2676,2751,2915,3134,3352,3366,3598,3612,3820,4013,4557,4794,5547])
  for it in samples:
    print("English Word :",eng_words[it].strip())
    print("Telugu Actual :",tel_words[it].strip())
    s = ""
    for i in range(len(output_list[it])):
      s = s+(num_dict_telugu[np.argmax(output_list[it][i])])
    [output1,encoder_states] = encoder_model.predict(np.array([word_embed[it]]))
    list_p = np.zeros(word_embed.shape[1])
    list_p[0] = 1.0
    L = []
    for i in range(tel_word_list.shape[1]):
      target_seq = np.zeros((1,1,tel_word_list.shape[2]))
      #get attention weights
      [decoder_outputs,decoder_states,l] = decoder_model.predict([target_seq,output1] + encoder_states)
      encoder_states = decoder_states
      #remove initial tab characters
      l = list(l[0][0])
      m = l[0]
      l.pop(0)
      l.pop(0)
      l.append(m)
      l.append(0)
      target_seq[:,0] = decoder_outputs[:,0]
      L.append(l)
    L.pop()
    L = np.array(L)
    lis = []
    true = eng_words[it]
    true_list = []
    true_list.extend(list(true)[1:])
    #add the end term
    true_list[len(true_list)-1] = 'end'
    pred_list = []
    s = s.strip()
    print("Telugu Predicted :",s)
    pred_list.extend(list(s))
    #add end term
    pred_list.append('end')
    #crop the attention weights matrix till end terms only
    lis = np.array(L[:len(pred_list),:len(true_list)])
    #plot heatmaps
    plot_attention(lis,pred_list,true_list,it)
    #plot visualizations
    for i in range(len(pred_list)):
      format_chars(true_list,lis[i],pred_list[i],i+1)


#function computing accuracy. It returns word accuracy but also prints the character accuracy
#Implementation similar to that of transliteration.py

def calc_acc(beam_size,output,one_hot2):
  #for character accuracy computation
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
  #calculate word accuracy for each beam output
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


#same function as in transliteration.py for same purpose to return the models
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
  #We also add attention outputs for plotting the heatmaps
  decoder_model = keras.Model([decoder_inputs,output1] + decoder_ip_states, [decoder_outputs,decoder_states,L])
  return [decoder_model,encoder_model,model]


#setting sweeps for the best models

sweep_config = {
    'method' : 'grid',
    'metric': {
      'name': 'test_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'attention' : {
            'values' : [0,1]
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

#get a sweep id

sweep_id = wandb.sweep(sweep_config, entity="mounik2000", project="A3 TestTime check")

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

#run an agent for the sweep
wandb.agent(sweep_id, sweep_train)
