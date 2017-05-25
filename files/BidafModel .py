import keras
import json
import numpy as np
from keras.models import Model
from keras.layers import Bidirectional, LSTM, Dense, Embedding, add, dot, Input, Activation
from keras.utils import to_categorical

context_train = json.load(open('data/squad/context_train_useful.json'))
print('Context train read')
query_train = json.load(open('data/squad/query_train.json'))
print('Query train read')
shared = json.load(open('data/squad/shared.json'))
print('shared  read')

word_emb_size = 100
num_classes = T = len(context_train['context_idx'][0])
J = len(query_train['question_idx'][0])

y_train_start, y_train_end = [], [] 
for x, y in zip(query_train['answer_start'], query_train['answer_end']):
    y_train_start.append(x[0])
    y_train_end.append(y[0])

y_train_start_encoded = to_categorical(y_train_start, num_classes)
y_train_end_encoded = to_categorical(y_train_end, num_classes)

emb_mat = np.array([shared['word2vec'][word] if word in shared['word2vec']
                        else np.random.multivariate_normal(np.zeros(word_emb_size), np.eye(word_emb_size))
                        for word in shared['all_words']])


model1_input = Input(shape = (T,))
model1_embed = Embedding(output_dim=100, input_dim=emb_mat.shape[0], weights = [emb_mat], trainable = False)(model1_input)
model1_lstm = Bidirectional(LSTM(256, return_sequences = True))(model1_embed)

model2_input = Input(shape = (J, ))
model2_embed = Embedding(output_dim=100, input_dim=emb_mat.shape[0], weights = [emb_mat], trainable = False)(model2_input)
model2_lstm = Bidirectional(LSTM(256, return_sequences = True))(model2_embed)

multiply_layer = dot([model1_lstm, model2_lstm],axes = 2)

modelling_bilstm_layer = Bidirectional(LSTM(256, return_sequences = True))(multiply_layer)
modelling_bilstm_layer_1 = Bidirectional(LSTM(256))(modelling_bilstm_layer)
modelling_bilstm_layer_return_sequences = Bidirectional(LSTM(256, return_sequences = True))(modelling_bilstm_layer) 

output_start_index = Dense(num_classes,activation='softmax')(modelling_bilstm_layer_1)

output_end_index_lstm = LSTM(num_classes)(modelling_bilstm_layer_return_sequences)
merge = add([output_end_index_lstm,output_start_index])
output_end_index_lstm = Activation('softmax')(merge)


final_model = Model(inputs=[model1_input, model2_input], outputs=[output_start_index, output_end_index_lstm])

final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

final_model.summary()

