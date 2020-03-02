#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:12:23 2020

@author: joshi.purvi
"""

#from chatterbot import ChatBot


import tensorflow

import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from sagemaker.tensorflow import TensorFlow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import unicodedata
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping,ModelCheckpoint
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.manifold import TSNE
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM


def read_review_csv():
    ''' read yelp review data from csv file and display
    return review object
    '''
    review = pd.read_csv("new_dataset_indian_resto_review.csv")
    print(review.head(2))
    return review

def find_useful(df):
    ''' 
    return 0-not useful if stars <=3, otherwise 1-useful
    '''
    if df[6] <= 3:
        return 0
    else:
        return 1
def extract_text_label(review):
    
    ''' Taking only 2 columns from review data frame and return new dataframe '''
    
    review = review[["text","label"]]
    return review 
def dropna(review):
    
    ''' drop null values ffrom review datagrame '''
    
    review = review.dropna()
    print(review["label"].value_counts())
    return review

def create_train_test(review):
    train_set = review.sample(frac=0.90, random_state=0)
    test_set = review.drop(train_set.index)
    return train_set,test_set

def token_sequence(review):
    
    ''' creating token sequence and padding sequence, returns tokenizer and padding sequence '''
    
    max_len=50
    embed_dim=100
    max_words=5000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(review["text"])
    sequences = tokenizer.texts_to_sequences(review["text"])
    data = pad_sequences(sequences,maxlen=max_len,padding='post')
    return tokenizer,data

def create_label_array(review):
    
    ''' returns np array of labels '''
    
    return np.asarray(review["label"])

def create_lstm_model():
    max_len=50
    embed_dim=100
    max_words=5000
    model = Sequential()
    #model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    #model.add(SpatialDropout1D(0.2))
    model.add(Embedding(max_words,embed_dim,input_length=50))#max_len)) #weights = [embeded_matrix]
    #model.add(Dropout(0.3))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1,activation="sigmoid"))
    model.summary()
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    early_stoping = EarlyStopping(monitor="val_loss",patience=5,mode=min)
    save_best = ModelCheckpoint("yelpcomments_model.hdf",save_best_only=True,monitor = "val_loss",mode="min")
    #%%time
    model.fit(train_data,y_train,epochs=20,validation_data=(val_data,y_test),batch_size=128,verbose=1,callbacks=[early_stoping])
    model.fit(train_data,y_train,epochs=20,validation_data=(finaltest_data,test_labels),batch_size=128,verbose=1,callbacks=[early_stoping])


def main():
    review =read_review_csv()
    review["label"] = review.apply(find_useful,axis=1)
    review = extract_text_label(review)
    train_set, test_set = create_train_test(review)
    train_labels = create_label_array(train_set) #validation test
    test_labels = create_label_array(test_set) # unseen test
    X_train, X_test, y_train, y_test = train_test_split(train_set, train_labels, test_size=0.20, random_state=42)
    tokenizer_train,train_data=token_sequence(X_train)
    tokenizer_val_test,val_data=token_sequence(X_test)
    tokenizer_final_test,finaltest_data=token_sequence(test_set)
    create_lstm_model(train_data,y_train,val_data,y_test,finaltest_data,test_labels)
    
if __name__ == "__main__":
    
    ''' starting point of module'''
    
    main()