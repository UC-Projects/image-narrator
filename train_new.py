#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import cv2 
import os
import glob
  


# In[ ]:


import pickle
with open('dict_variables.pkl', 'rb') as file:
    dict_variables = pickle.load(file)
with open('train_variables.pkl', 'rb') as file:
    train_variables = pickle.load(file)
with open('val_variables.pkl', 'rb') as file:
    val_variables = pickle.load(file)


# In[ ]:


X_train = np.array(train_variables['X_train'])
y_in_train = np.array(train_variables['y_in_train'], dtype='float64')
y_out_train = np.array(train_variables['y_out_train'], dtype='float64')


# In[ ]:


X_val = np.array(val_variables['X_val'])
y_in_val = np.array(val_variables['y_in_val'], dtype='float64')
y_out_val = np.array(val_variables['y_out_val'], dtype='float64')


# In[ ]:


captions_dict = dict_variables['captions_dict']
new_dict = dict_variables['new_dict']
images_features = dict_variables['images_features']


# In[ ]:


from tensorflow.keras.utils import to_categorical
MAX_LEN = 0
for k, vv in captions_dict.items():
    for v in vv:
        if len(v) > MAX_LEN:
            MAX_LEN = len(v)


# In[ ]:


from keras.utils import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model


# In[ ]:


from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, RepeatVector, TimeDistributed, Activation,Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

# Define constants
embedding_size = 128
max_len = MAX_LEN
vocab_size = len(new_dict)

# Define image model
image_model = Sequential([
    Dense(embedding_size, input_shape=(2048,), activation='relu'),
    RepeatVector(max_len)
])

# Define language model
language_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len),
    LSTM(256, return_sequences=True),
    TimeDistributed(Dense(embedding_size))
])

# Concatenate image and language models
conca = Concatenate()([image_model.output, language_model.output])

x = LSTM(256, return_sequences=True)(conca)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = LSTM(128, return_sequences=True)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Global LSTM layer
x = LSTM(512, return_sequences=False)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Dense layers
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Output layer
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)


# Create the model
model = Model(inputs=[image_model.input, language_model.input], outputs=out)

# Compile the model
optimizer = RMSprop(learning_rate=0.001)  # You can adjust the learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Print the model summary
model.summary()


# In[ ]:


history = model.fit(
    [X_train, y_in_train], 
    y_out_train, 
    epochs=150,
    batch_size=128,
    validation_data=([X_val, y_in_val], y_out_val)
)


# In[ ]:


inv_dict = {v:k for k, v in new_dict.items()}


# In[ ]:


model.save('image_narrator_model.h5')
model.save_weights('in_model_weights.h5')


# In[ ]:


def get_image(x):
    test_img_path = images[x]
    img = cv2.imread(test_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    test_img = img.reshape(1,224,224,3)
    
    return test_img


# In[ ]:


import cv2
# Later, load the model
from tensorflow.keras.models import load_model

loaded_model = load_model('image_narrator_model.h5')
def generate_caption(image_index):
    # Get image
    test_feature = loaded_model.predict(get_image(image_index)).reshape(1,2048)
    if test_feature is None:
        return
    
    test_img_path = images[image_index]
    test_img = cv2.imread(test_img_path)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    # Initialize caption input
    text_inp = ['startofseq']

    # Generate caption
    count = 0
    caption = ''
    while count < 25:
        count += 1
        encoded = [new_dict.get(word, new_dict['<OUT>']) for word in text_inp]
        encoded = [encoded]
        encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=MAX_LEN)
        prediction = np.argmax(loaded_model.predict([test_feature, encoded]))
        sampled_word = inv_dict[prediction]
        caption = caption + ' ' + sampled_word
        if sampled_word == 'endofseq':
            break
        text_inp.append(sampled_word)

    # Display image and caption using OpenCV
    cv2.imshow('image', test_img)
    print('Caption:', caption)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

