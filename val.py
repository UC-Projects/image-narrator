#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import cv2 
import os
import glob


# In[19]:


from tensorflow.keras.applications import ResNet50
incept_model=ResNet50(include_top=True)
#incept_model.summary()  

from keras.models import Model
last=incept_model.layers[-2].output
modele=Model(inputs = incept_model.input,outputs = last)
modele.summary()    


# In[20]:


images_path="flickr30k/Images/flickr30k_images/"
images=glob.glob(images_path+"*.jpg")
caption_path = 'flickr30k/captions.txt'
with open(caption_path, 'r') as f:
    val_captions = f.readlines()
    
# Creating lists to store image paths and corresponding val_captions
val_image_paths = []
val_captions_dict = {}
val_images_features = {}
val_count = 0

for caption in val_captions[int(len(val_captions)*0.7):]:
    image_id,caption_text= caption.strip().split('.jpg,')
    image_path = os.path.join(images_path, image_id+'.jpg')
    if(image_path not in val_image_paths):
        val_image_paths.append(image_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        img = img.reshape(1,224,224,3)
        pred = modele.predict(img).reshape(2048,)  
        img_name = image_path.split('/')[-1]
        caption=caption_text.strip('"')
        if img_name not in val_captions_dict:
            val_captions_dict[img_name] = [caption]
        else:
            val_captions_dict[img_name].append(caption)
        val_images_features[img_name] = pred
    val_count += 1
    
    if val_count % 50 == 0:
        print(val_count)


# In[21]:


def preprocessed(txt):
    modified = txt.lower()
    modified = 'startofseq ' + modified + ' endofseq'
    return modified


# In[22]:


for k,v in val_captions_dict.items():
    for vv in v:
        val_captions_dict[k][v.index(vv)] = preprocessed(vv)


# In[23]:


val_count_words = {}
for k,vv in val_captions_dict.items():
    for v in vv:
        for word in v.split():
            if word not in val_count_words:

                val_count_words[word] = 0

            else:
                val_count_words[word] += 1


# In[24]:


THRESH = -1
val_count = 1
new_dict = {}
for k,v in val_count_words.items():
    if val_count_words[k] > THRESH:
        new_dict[k] = val_count
        val_count += 1
        


# In[25]:


new_dict['<OUT>'] = len(new_dict) 
val_captions_backup = val_captions_dict.copy()
val_captions_dict = val_captions_backup.copy()
for k, vv in val_captions_dict.items():
    for v in vv:
        val_encoded = []
        for word in v.split():  
            if word not in new_dict:
                val_encoded.append(new_dict['<OUT>'])
            else:
                val_encoded.append(new_dict[word])


        val_captions_dict[k][vv.index(v)] = val_encoded


# In[26]:


from tensorflow.keras.utils import to_categorical
from keras.utils import pad_sequences
MAX_LEN = 0
for k, vv in val_captions_dict.items():
    for v in vv:
        if len(v) > MAX_LEN:
            MAX_LEN = len(v)


# In[27]:


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


# In[29]:


X_val, y_in_val, y_out_val = generator(val_images_features, val_captions_dict)


# In[30]:


import pickle
val_variables = {
    'X_val': X_val,
    'y_in_val': y_in_val,
    'y_out_val': y_out_val,
}

with open('val_variables.pkl', 'wb') as file:
    pickle.dump(val_variables, file)


# In[31]:


with open('val_variables.pkl', 'rb') as file:
    loaded_val_variables = pickle.load(file)


# In[32]:


X_val = np.array(val_variables['X_val'])
y_in_val = np.array(val_variables['y_in_val'], dtype='float64')
y_out_val = np.array(val_variables['y_out_val'], dtype='float64')


# In[34]:


X_val.shape, y_in_val.shape, y_out_val.shape


# In[ ]:




