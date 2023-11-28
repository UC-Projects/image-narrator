#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import cv2 
import os
import glob


# In[2]:


#importing Images
images_path="flickr30k/Images/flickr30k_images/"
images=glob.glob(images_path+"*.jpg")
len(images)
images[:5]
import matplotlib.pyplot as plt
for i in range(5):
    plt.figure()
    img=cv2.imread(images[i])
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)


# In[3]:


from tensorflow.keras.applications import ResNet50
incept_model=ResNet50(include_top=True)
#incept_model.summary()  

from keras.models import Model
last=incept_model.layers[-2].output
modele=Model(inputs = incept_model.input,outputs = last)
modele.summary()    


# In[4]:


caption_path = 'flickr30k/captions.txt'
with open(caption_path, 'r') as f:
    captions = f.readlines()
    
# Creating lists to store image paths and corresponding captions
image_paths = []
captions_dict = {}
images_features = {}
count = 0

for caption in captions[1:int(len(captions)*0.5)]:
    image_id,caption_text= caption.strip().split('.jpg,')
    image_path = os.path.join(images_path, image_id+'.jpg')
    if(image_path not in image_paths):
        image_paths.append(image_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        img = img.reshape(1,224,224,3)
        pred = modele.predict(img).reshape(2048,)  
        img_name = image_path.split('/')[-1]
        caption=caption_text.strip('"')
        if img_name not in captions_dict:
            captions_dict[img_name] = [caption]
                
        else:
            captions_dict[img_name].append(caption)
        images_features[img_name] = pred
    count += 1
    
    if count % 50 == 0:
        print(count)


# In[5]:


import matplotlib.pyplot as plt

for i in range(5):
    plt.figure()
    img_name = image_paths[i]
    
    img = cv2.imread(img_name)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.xlabel(":".join(captions_dict[img_name.split('/')[-1]]))
    plt.imshow(img)


# In[6]:


def preprocessed(txt):
    modified = txt.lower()
    modified = 'startofseq ' + modified + ' endofseq'
    return modified


# In[7]:


for k,v in captions_dict.items():
    for vv in v:
        captions_dict[k][v.index(vv)] = preprocessed(vv)


# In[8]:


count_words = {}
for k,vv in captions_dict.items():
    for v in vv:
        for word in v.split():
            if word not in count_words:

                count_words[word] = 0

            else:
                count_words[word] += 1


# In[9]:


THRESH = -1
count = 1
new_dict = {}
for k,v in count_words.items():
    if count_words[k] > THRESH:
        new_dict[k] = count
        count += 1
        


# In[10]:


new_dict['<OUT>'] = len(new_dict) 
captions_backup = captions_dict.copy()
captions_dict = captions_backup.copy()
for k, vv in captions_dict.items():
    for v in vv:
        encoded = []
        for word in v.split():  
            if word not in new_dict:
                encoded.append(new_dict['<OUT>'])
            else:
                encoded.append(new_dict[word])


        captions_dict[k][vv.index(v)] = encoded


# In[11]:


import pickle
    
dict_variables = {
    'captions_dict': captions_dict,
    'new_dict': new_dict,
    'images_features': images_features,
}

with open('dict_variables.pkl', 'wb') as file:
    pickle.dump(dict_variables, file)


# In[12]:


from tensorflow.keras.utils import to_categorical
MAX_LEN = 0
for k, vv in captions_dict.items():
    for v in vv:
        if len(v) > MAX_LEN:
            MAX_LEN = len(v)


# In[17]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
Batch_size = 5000
VOCAB_SIZE = len(new_dict)

def generator(photo, caption):
    n_samples = 0
    
    X = []
    y_in = []
    y_out = []
    
    for k, vv in caption.items():
        for v in vv:
            for i in range(1, len(v)):
                X.append(photo[k])

                in_seq= [v[:i]]
                out_seq = v[i]

                in_seq = pad_sequences(in_seq, maxlen=MAX_LEN, padding='post', truncating='post')[0]
                out_seq = to_categorical([out_seq], num_classes=VOCAB_SIZE)[0]

                y_in.append(in_seq)
                y_out.append(out_seq)
            
    return X, y_in, y_out


# In[18]:


X, y_in, y_out = generator(images_features, captions_dict)
len(X), len(y_in), len(y_out)


# In[19]:


import pickle
train_variables = {
    'X': X,
    'y_in': y_in,
    'y_out': y_out,
}

with open('train_variables.pkl', 'wb') as file:
    pickle.dump(train_variables, file)


# In[4]:


import pickle
with open('dict_variables.pkl', 'rb') as file:
    dict_variables = pickle.load(file)
with open('train_variables.pkl', 'rb') as file:
    train_variables = pickle.load(file)
with open('val_variables.pkl', 'rb') as file:
    val_variables = pickle.load(file)


# In[5]:


X = np.array(train_variables['X'])
y_in = np.array(train_variables['y_in'], dtype='float64')
y_out = np.array(train_variables['y_out'], dtype='float64')


# In[6]:


print("train shape---->  ",X.shape, y_in.shape, y_out.shape)


# In[8]:


X_val = np.array(val_variables['X_val'])
y_in_val = np.array(val_variables['y_in_val'], dtype='float64')
y_out_val = np.array(val_variables['y_out_val'], dtype='float64')


# In[9]:


print("val shape ------>",X_val.shape, y_in_val.shape, y_out_val.shape)


# In[10]:


captions_dict = dict_variables['captions_dict']
new_dict = dict_variables['new_dict']
images_features = dict_variables['images_features']


# In[11]:


len(captions_dict), len(new_dict), len(images_features)


# In[13]:


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


# In[14]:


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


# In[15]:


history = model.fit(
    [X, y_in], 
    y_out, 
    epochs=50,
    batch_size=32,
    validation_data=([X_val, y_in_val], y_out_val)
)
#model.fit([X, y_in], y_out, batch_size=32, epochs=50)


# In[ ]:


inv_dict = {v:k for k, v in new_dict.items()}


# In[21]:


model.save('image_narrator_model.h5')
model.save_weights('in_model_weights.h5')


# In[19]:


def get_image(x):
    test_img_path = images[x]
    img = cv2.imread(test_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    test_img = img.reshape(1,224,224,3)
    
    return test_img


# In[20]:


import cv2
# Later, load the model
from tensorflow.keras.models import load_model

loaded_model = load_model('model.h5')
def generate_caption(image_index):
    # Get image
    test_feature = modele.predict(get_image(image_index)).reshape(1,2048)
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


# In[ ]:




