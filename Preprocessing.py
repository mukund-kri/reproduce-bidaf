
# coding: utf-8

# In[1]:

# Char to Word(Squad) - Done
# Conversion the whole data into reading data with relevant information - Done
# Tokenizing the context for data extraction - Done
# Preprocessing the data- Word2vec - Done
# Making neccesary list- Done
# Making the vocab and conversion each word to id- Done


# In[2]:

import nltk
import re
import json
import os
import numpy as np
from collections import Counter


# In[3]:

with open('data/squad/train-v1.1.json') as json_data:
    train_data = json.load(json_data)
with open('data/squad/dev-v1.1.json') as json_data:
    dev_data = json.load(json_data)


# In[4]:

def save_json(mode, context, query):
    print(mode)
    context_path = os.path.join('data/squad','context_{}.json'.format(mode))
    query_path = os.path.join('data/squad','query_{}.json'.format(mode))
    json.dump(context,open(context_path,'w'))
    json.dump(query,open(query_path,'w'))
    print('Saved the ',mode,' data file')
    return 'Success'


# In[5]:

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


# In[6]:

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


# In[7]:

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


# In[8]:

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


# In[9]:

#training_data = read_data()


# In[10]:

def build_vocab(train_data,dev_data):
    word_counter,char_counter = Counter(),Counter()
    train_data['data'].extend(dev_data['data'])
    for data_index, data_paragraph in enumerate(train_data['data']):
        for para_index, each_paragraph in enumerate(data_paragraph['paragraphs']):
            context_data = each_paragraph['context']
            context_data = context_data.replace("''", '" ')
            context_data = context_data.replace("``", '" ')
            tokenized_context = list(map(word_tokenize, sent_tokenize(context_data)))
            tokenized_context = [process_context(tokens) for tokens in tokenized_context] 

            # Getting the Word and Character Counter for number of questions
            for tokenized in tokenized_context:
                for tokens in tokenized:
                    word_counter[tokens.lower()] += len(each_paragraph['qas'])
                    for each_token in tokens:
                        char_counter[each_token] += len(each_paragraph['qas'])

            for question_data in each_paragraph['qas']:
                    ques = question_data['question']
                    q = word_tokenize(ques) 
                    # Generating Word and char counter of only the questions
                    for qi in q:
                        word_counter[qi.lower()] += 1
                        for qq in qi:
                            char_counter[qq] += 1
    return word_counter,char_counter


# In[11]:

word_counter,char_counter = build_vocab(train_data,dev_data)


# In[12]:

def preprocessing_data(mode,data):
    word_counter,char_counter = Counter(), Counter()
    word_context, char_context, actual_context, word_question, char_question, relation, answer_start,answer_end =[],[],[],[],[],[],[],[] 
    for data_index, data_paragraph in enumerate(data['data']):
        for para_index, each_paragraph in enumerate(data_paragraph['paragraphs']):
            context_data = each_paragraph['context']
            context_data = context_data.replace("''", '" ')
            context_data = context_data.replace("``", '" ')
            tokenized_context = list(map(word_tokenize, sent_tokenize(context_data)))
            tokenized_context = [process_context(tokens) for tokens in tokenized_context] 

            # Adding character to tokenized context
            cxi = [[list(token) for token in tokenized] for tokenized in tokenized_context]
            word_context.append(tokenized_context)
            char_context.append(cxi)
            actual_context.append(context_data)


            # Getting the question-context relation
            r = [data_index,para_index]

            for question_data in each_paragraph['qas']:
                    ques = question_data['question']
                    q = word_tokenize(ques)
                    cq = [list(ques_char) for ques_char in q]
                    y_start,y_end = [],[]  
                    for answer in question_data['answers']:
                        start = answer['answer_start']
                        stop = start + len(answer)
                        # Getting the word span
                        y0, y1 = get_word_span(context_data, tokenized_context, start, stop)
                        y_start.append(y0)
                        y_end.append(y1)
                    word_question.append(q)
                    char_question.append(cq)
                    answer_start.append(y_start)
                    answer_end.append(y_end)
                    relation.append(r)
        print(data_index)
    context = {'actual_context':actual_context, 'char_context' : char_context, 'word_context': word_context}
    query = {'word_question':word_question, 
             'char_question':char_question, 
             'relation' : relation,
             'answer_start' : answer_start,
            'answer_end' : answer_end}
    save_json(context=context,query=query,mode=mode)


# In[13]:

word2vec = get_word2vec(word_counter)


# In[14]:

def save_updated_files():
    dir_path = 'data/squad'
    path = os.path.join(dir_path,'context_train.json')
    json.dump(context_train,open(path,'w'))
    print('Context_train written')
    path = os.path.join(dir_path,'context_test.json')
    json.dump(context_test,open(path,'w'))
    print('Context_test written')
    path = os.path.join(dir_path,'query_train.json')
    json.dump(query_train,open(path,'w'))
    print('Query_train written')
    path = os.path.join(dir_path,'query_test.json')
    json.dump(query_test,open(path,'w'))
    print('Query_test written')
    path = os.path.join(dir_path,'shared.json')
    json.dump(shared,open(path,'w'))
    print('Shared Data file written')


# In[15]:

preprocessing_data(mode = 'test',data = dev_data)


# In[16]:

preprocessing_data(mode = 'train',data = train_data)


# In[17]:

with open('data/squad/context_train.json') as json_data:
    context_train = json.load(json_data)
with open('data/squad/query_train.json') as json_data:
    query_train = json.load(json_data)
with open('data/squad/context_test.json') as json_data:
    context_test = json.load(json_data)
with open('data/squad/query_test.json') as json_data:
    query_test = json.load(json_data)


# In[18]:

all_word2idx = ['NULL'] + ['UNKNOWN'] + [word for word in word_counter.keys()] 


# In[19]:

all_word2idx = {word:idx for idx,word in enumerate(all_word2idx)}


# In[20]:

shared = {}
shared['word2vec'] = word2vec
shared['word_counter'] = word_counter
shared['char_counter'] = char_counter
shared['all_words'] = all_word2idx


# In[21]:

max_sent_len_train = max(len(sent) for con in context_train['word_context'] for sent in con)
max_para_len_train = max(len(con) for con in context_train['word_context'])
max_sent_len_test = max(len(sent) for con in context_test['word_context'] for sent in con)
max_para_len_test = max(len(con) for con in context_test['word_context'])


# In[22]:

maxquery_sent_len_train = max(len(con) for con in query_train['word_question'])
maxquery_sent_len_test = max(len(con) for con in query_test['word_question'])
maxquery_sent_len_train,maxquery_sent_len_test


# In[23]:

max_sent_cont = max(max_sent_len_train,max_sent_len_test)
max_para_cont = max(max_para_len_train,max_para_len_train)
max_sent_query = max(maxquery_sent_len_train,maxquery_sent_len_test)
max_sent_query,max_para_cont,max_sent_cont


# In[24]:

def contexttrain_to_id():
    context_idx = [0] * len(context_train['word_context'])
    for p,each_context in enumerate(context_train['word_context']):
        sent_idx = [0] * max_para_cont
        para_len = len(each_context)
        for s in range(max_para_cont):
            word_idx = [0] * max_sent_cont
            if s >= para_len:
                sent_idx[s] = word_idx
            else:
                each_sentence = each_context[s]
                sent_len = len(each_sentence)
                for w in range(max_sent_cont):
                    if w < sent_len:
                        each_word = each_sentence[w]
                        if each_word.lower() not in word2vec.keys():
                            word_idx[w] = 1
                        else:
                            word_idx[w]=all_word2idx[each_word.lower()]
                sent_idx[s]=word_idx
                
        context_idx[p]=sent_idx
    return context_idx


# In[25]:

def contexttest_to_id():
    context_idx = [0] * len(context_test['word_context'])
    for p,each_context in enumerate(context_test['word_context']):
        sent_idx = [0] * max_para_cont
        para_len = len(each_context)
        for s in range(max_para_cont):
            word_idx = [0] * max_sent_cont
            if s >= para_len:
                sent_idx[s] = word_idx
            else:
                each_sentence = each_context[s]
                sent_len = len(each_sentence)
                for w in range(max_sent_cont):
                    if w < sent_len:
                        each_word = each_sentence[w]
                        if each_word.lower() not in word2vec.keys():
                            word_idx[w] = 1
                        else:
                            word_idx[w]=all_word2idx[each_word.lower()]
                sent_idx[s]=word_idx
                
        context_idx[p]=sent_idx
    return context_idx


# In[26]:

def querytrain_to_id():
    question_idx = [0] * len(query_train['word_question'])
    word_idx = [0] * max_sent_query
    for q,each_question in enumerate(query_train['word_question']):
       
        for w,each_word in zip(range(max_sent_cont),each_question):
                if each_word.lower() not in word2vec.keys():
                    word_idx[w] = 1
                else:
                    word_idx[w]=all_word2idx[each_word.lower()]
        question_idx[q] = word_idx
        word_idx = [0] * max_sent_query
    return question_idx


# In[27]:

def querytest_to_id():
    question_idx = [0] * len(query_test['word_question'])
    word_idx = [0] * max_sent_query
    for q,each_question in enumerate(query_test['word_question']):
       
        for w,each_word in zip(range(max_sent_cont),each_question):
                if each_word.lower() not in word2vec.keys():
                    word_idx[w] = 1
                else:
                    word_idx[w]=all_word2idx[each_word.lower()]
        question_idx[q] = word_idx
        word_idx = [0] * max_sent_query
    return question_idx


# In[28]:

context_train_idx = contexttrain_to_id()
context_test_idx = contexttest_to_id()
query_train_idx = querytrain_to_id()
query_test_idx = querytest_to_id()


# In[29]:

context_train['context_idx'] = context_train_idx
context_test['context_idx'] = context_test_idx
query_train['question_idx'] = query_train_idx
query_test['question_idx'] = query_test_idx
save_updated_files()

