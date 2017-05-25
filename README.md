# reproduce-bidaf
Attempt @ stackroute toe reporduce the bidaf paper and use in a chatbot

### experiments ###
contains the ipython notebooks of the code we wrote throughout the way to approach pur problem.

### preprocessing ###
contains the preprocessing files which splits the data into context train, context test, query train, query test and shared file which conains the data shared by both query and context.

### model ### 
contains the model we built for solving our problem

## Running the code ##

1) Preprocessing:
   We do this step to tranform data from the SQuAD format to a format suitable for our code.

   First, prepare data. Donwload SQuAD data and GloVe and nltk corpus (~850 MB, this will download files to $HOME/data):
   Second, Preprocess Stanford QA dataset (along with GloVe vectors) and save them in $PWD/data/squad (~3 minutes):
  
   ```python -m preprocessing/preprocessing.py```


2) Training:

   ```python -m BidafModel.py```





