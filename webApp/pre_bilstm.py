import utils_prepro
import pandas as pd
import hazm
import re
from cleantext import clean
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

train = pd.read_csv('snappfood/train.csv',error_bad_lines=False,sep = '\t', encoding='utf-8')

train = train[['comment', 'label_id']]

"""## Preprocessing text column data"""

# cleaning comments
train['cleaned_comment'] = train['comment'].apply(utils_prepro.apply_cleaning)
train = train[['cleaned_comment', 'label_id']]
x_train, y_train = train['cleaned_comment'].values.tolist(), train['label_id'].values.tolist()

"""# Keras and BiLSTM model"""

max_features = 20000  # Only consider the top 15k words
maxlen = 200  # Only consider the first 200 words of each movie review

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train)

X_train = tokenizer.texts_to_sequences(x_train)

X_train = pad_sequences(X_train, maxlen=maxlen)

"""# Build the model"""

def build_model():
  # Input for variable-length sequences of integers
  inputs = keras.Input(shape=(None,), dtype="int32")
  # Embed each integer in a 128-dimensional vector
  x = layers.Embedding(max_features, 128)(inputs)
  # Add 2 bidirectional LSTMs
  x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
  x = layers.Bidirectional(layers.LSTM(64))(x)
  # Add a classifier
  outputs = layers.Dense(1, activation="sigmoid")(x)
  model = keras.Model(inputs, outputs)
  print(model.summary())
  return model