
# coding: utf-8

# In[1]:

# Char to Word(Squad) - Done
# Conversion the whole data into reading data with relevant information - Done
# Tokenizing the context for data extraction - Done
# Preprocessing the data- Word2vec - Done
# Making neccesary list- Done


# In[2]:

import nltk
import re
import json
import os
import numpy as np
from collections import Counter


# In[3]:

out_path = os.path.join('data/squad/','all_data.json')


# In[4]:

with open('data/squad/train-v1.1.json') as json_data:
    train_data = json.load(json_data)
with open('data/squad/dev-v1.1.json') as json_data:
    dev_data = json.load(json_data)


# In[5]:

def save_json(mode, context, query):
    print(mode)
    context_path = os.path.join('data/squad','context_{}.json'.format(mode))
    query_path = os.path.join('data/squad','query_{}.json'.format(mode))
    json.dump(context,open(context_path,'w'))
    json.dump(query,open(query_path,'w'))
    print('Saved the ',mode,' file')
    return 'Success'


# In[6]:

def process_context(context):
    tokens = []
    for token in context:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

sent_tokenize = nltk.sent_tokenize


# In[7]:

def get_2d_spans(context, each_word):
    span = []
    start_index = 0
    for tokens in each_word:
        spans = []
        for token in tokens:
            start_index = context.find(token, start_index) # getting start index of each word after a particular index
            spans.append((start_index, start_index + len(token))) # appending that start and end index into list
            start_index += len(token) # updating the start index to check ahead
        span.append(spans)
    return span

def get_word_span(context, each_word, start, stop):
    each_word_span = get_2d_spans(context, each_word)
    word_span_list = []
    for sent_index, word_span in enumerate(each_word_span):
        for word_index, char_span in enumerate(word_span):
             if (stop >= char_span[0] and start <= char_span[1]):
                word_span_list.append((sent_index, word_index))
    return word_span_list[0], (word_span_list[-1][0], word_span_list[-1][1] + 1)


# In[8]:

def get_word2vec(word_counter):
    word2vec_dict = {}
    with open('data/glove/glove.6B.100d.txt','r',encoding = 'utf-8') as glove_data:
        for line in glove_data:
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter.keys():
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter.keys():
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter.keys():
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter.keys():
                word2vec_dict[word.upper()] = vector
    return word2vec_dict


# In[9]:

# Basic Understanding Code
def read_data():
    data = []
    data_points = {}
    each_context = []
    main_data = []
    for data_index, data_paragraph in enumerate(train_data['data']):
        for each_paragraph in data_paragraph['paragraphs']:
            question_data = each_paragraph['qas']
            context_data = each_paragraph['context']
            for number_of_question in range(len(question_data)):
                    data_points['context'] = context_data
                    data_points['question'] = question_data[number_of_question]['question']
                    answer = question_data[number_of_question]['answers'][0]['text']
                    data_points['answers'] = answer
                    start = question_data[number_of_question]['answers'][0]['answer_start']
                    stop = start + len(answer) 
                    tokenized_context = list(map(word_tokenize, sent_tokenize(context_data)))
                    tokenized_context = [process_context(tokens) for tokens in tokenized_context]
                    y0, y1 = get_word_span(data_points['context'], tokenized_context, start, stop)
                    data_points['answer_start'] = y0
                    data_points['answer_stop'] = y1
                    data.append(data_points)
                    data_points = {}
            each_context.append(data)
            data = []
        main_data.append(each_context)
        each_context = []
    return main_data


# In[10]:

#training_data = read_data()


# In[11]:

def preprocessing_data(mode,data):
    word_counter,char_counter = Counter(), Counter()
    word_context, char_context, actual_context, word_question, char_question, relation, answer_points = [],[],[],[],[],[],[] 
    for data_index, data_paragraph in enumerate(data['data']):
        x, cx = [], []
        word_context.append(x)
        char_context.append(cx)
        for para_index, each_paragraph in enumerate(data_paragraph['paragraphs']):
            context_data = each_paragraph['context']
            context_data = context_data.replace("''", '" ')
            context_data = context_data.replace("``", '" ')
            tokenized_context = list(map(word_tokenize, sent_tokenize(context_data)))
            tokenized_context = [process_context(tokens) for tokens in tokenized_context] 

            # Adding character to tokenized context
            cxi = [[list(token) for token in tokenized] for tokenized in tokenized_context]
            x.append(tokenized_context)
            cx.append(cxi)
            actual_context.append(context_data)

            # Getting the Word and Character Counter for number of questions
            for tokenized in tokenized_context:
                for tokens in tokenized:
                    word_counter[tokens.lower()] += len(each_paragraph['qas'])
                    for each_token in tokens:
                        char_counter[each_token] += len(each_paragraph['qas'])

            # Getting the question-context relation
            r = [data_index,para_index]

            for question_data in each_paragraph['qas']:
                    ques = question_data['question']
                    q = word_tokenize(ques)
                    cq = [list(ques_char) for ques_char in q]
                    y = []  
                    for answer in question_data['answers']:
                        start = answer['answer_start']
                        stop = start + len(answer)
                        # Getting the word span
                        y0, y1 = get_word_span(context_data, tokenized_context, start, stop)
                        y.append([y0, y1])

                    # Generating Word and char counter of only the questions
                    for qi in q:
                        word_counter[qi.lower()] += 1
                        for qq in qi:
                            char_counter[qq] += 1

                    word_question.append(q)
                    char_question.append(cq)
                    answer_points.append(y)
                    relation.append(r)
        print(data_index)
    word2vec = get_word2vec(word_counter)
    context = {'actual_context':actual_context, 'char_context' : char_context, 'word_context': word_context, 'word_counter' : word_counter, 'char_counter' : char_counter, 'word2vec' : word2vec}
    query = {'word_question':word_question, 'char_question':char_question, 'relation' : relation,'answer_points' : answer_points}
    save_json(mode,context,query)


# In[12]:

preprocessing_data(mode = 'test',data = dev_data)


# In[13]:

preprocessing_data(mode = 'train',data = train_data)


# In[14]:

len(dev_data['data'])


# In[15]:

len(train_data['data'])


# In[ ]:



