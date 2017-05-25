#Attention Layer

import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


sess = tf.Session()
combination = '1,2,1*2,1*3'
combinations = combination.split(",")

def attention_layer(matrix_1,matrix_2):
        matrix_1 = matrix_1.eval(session=sess)
        matrix_2 = matrix_2.eval(session=sess)
        matrix_1 = keras.utils.normalize(matrix_1)
        matrix_2 = keras.utils.normalize(matrix_2)
        num_rows_1 = K.shape(matrix_1)[1]   #no_of_words_in_passage
        num_rows_2 = K.shape(matrix_2)[1]   #no_of_words_in_query
        print(num_rows_2)
        tile_dims_1 = K.concatenate([[1, 1], [num_rows_2], [1]], 0)    #concatenate size of h and u for making it as a same size
        tile_dims_2 = K.concatenate([[1], [num_rows_1], [1, 1]], 0)
        tiled_matrix_1 = K.tile(K.expand_dims(matrix_1, axis=2), tile_dims_1)
        tiled_matrix_2 = K.tile(K.expand_dims(matrix_2, axis=1), tile_dims_2)
        #Finds the similarity of h and u
        cos_sim = K.sum(K.l2_normalize(tiled_matrix_1,axis = -1) * K.l2_normalize(tiled_matrix_2,axis = -1),axis =-1)  
        #Context to query  
        atj = K.softmax(cos_sim) #attention vector
        matrix_2 = tf.constant(matrix_2)
        num_attention_dims = K.ndim(atj)      
        num_matrix_dims = K.ndim(matrix_2) - 1
        for _ in range(num_attention_dims - num_matrix_dims):
            matrix_2 = K.expand_dims(matrix_2, axis=1)   #making the dimension of atj and matrix_2 same
        u_aug = K.sum(K.expand_dims(atj, axis=-1) * matrix_2, -2)    #finding U_Aug
        #Query to Context
        question_passage_similarity = K.max(cos_sim, axis=-1)
        btj = K.softmax(question_passage_similarity)  #Attention Vector
        matrix_1 = tf.constant(matrix_1)
        num_attention_dims2 = K.ndim(btj)
        num_matrix_dims2 = K.ndim(matrix_1) - 1
        for _ in range(num_attention_dims2 - num_matrix_dims2):
            matrix_1 = K.expand_dims(matrix_1, axis=1)
        h_aug =  K.sum(K.expand_dims(btj, axis=-1) * matrix_1, -2)
        h_aug_final = find_h_aug(h_aug = h_aug,matrix= matrix_1)   #finding H_Aug
        final_pass = final_passage([matrix_1,u_aug,h_aug_final])   #finding G
        return final_pass
        
        

def find_h_aug(h_aug,matrix):
    to_repeat = h_aug
    to_copy = matrix
    expanded = K.expand_dims(to_repeat, axis=1)
    ones = [1] * K.ndim(expanded)
    num_repetitions = K.shape(to_copy)[1]
    tile_shape = K.concatenate([ones[:1], [num_repetitions], ones[2:]], 0)
    h_aug_final = K.tile(expanded,tile_shape)
    return h_aug_final



def combination(combination: str, tensors: List['Tensor']):
        if combination.isdigit():
            return tensors[int(combination) - 1]  # indices in the combination string are 1-indexed
        else:
            first_tensor = _get_combination(combination[0], tensors)
            second_tensor = _get_combination(combination[2], tensors)
            if K.int_shape(first_tensor) != K.int_shape(second_tensor):
                shapes_message = "Shapes were: {} and {}".format(K.int_shape(first_tensor),K.int_shape(second_tensor))
            operation = combination[1]
            if operation == '*':
                return first_tensor * second_tensor


def final_passage(x):
        combined_tensor = combination(combinations[0], x)
        for combination in combinations[1:]:
            to_concatenate = _get_combination(combination, x)
            combined_tensor = K.concatenate([combined_tensor, to_concatenate], axis=-1)
        return combined_tensor

