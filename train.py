#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[1]:


caption_path = 'flickr30k/captions.txt'
with open(caption_path, 'r') as f:
    captions = f.readlines()
    
# Creating lists to store image paths and corresponding captions
image_paths = []
captions_dict = {}
images_features = {}
count = 0

for caption in captions[1:len(captions)]:
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
print(len(X), len(y_in), len(y_out))


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_in_train, y_in_val, y_out_train, y_out_val = train_test_split(
    X, y_in, y_out, test_size=0.2, random_state=42
)


# In[19]:


import pickle
train_variables = {
    'X_train': X_train,
    'y_in_train': y_in_train,
    'y_out_train': y_out_train,
}

with open('train_variables.pkl', 'wb') as file:
    pickle.dump(train_variables, file)

val_variables = {
    'X_val': X_val,
    'y_in_val': y_in_val,
    'y_out_val': y_out_val,
}

with open('val_variables.pkl', 'wb') as file:
    pickle.dump(val_variables, file)


# In[ ]:




