#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import pathlib
import tensorflow as tf
from tensorflow import keras
from keras import layers


# In[5]:


import pickle


# In[6]:


normal = pathlib.Path("C:/Users/testi/OneDrive/Documents/Python Scripts/Demo Dataset - Iris Defect Detection Project Demo - Menmozhi Technologies/dataset/normal") # Path to normal images directory
glaucoma = pathlib.Path("C:/Users/testi/OneDrive/Documents/Python Scripts/Demo Dataset - Iris Defect Detection Project Demo - Menmozhi Technologies/dataset/glaucoma") # Path to glaucoma images directory
retinopathy = pathlib.Path("C:/Users/testi/OneDrive/Documents/Python Scripts/Demo Dataset - Iris Defect Detection Project Demo - Menmozhi Technologies/dataset/diabetic_retinopathy") # Path to diabetic retinopathy images directory
cataract = pathlib.Path("C:/Users/testi/OneDrive/Documents/Python Scripts/Demo Dataset - Iris Defect Detection Project Demo - Menmozhi Technologies/dataset/cataract") # Path to cataract images directory


# In[8]:


# Create a dictionary of image paths
images_dict = {
    "normal": list(normal.glob("*.jpg")),
    "glaucoma": list(glaucoma.glob("*.jpg")),
    "diabetic_retinopathy": list(retinopathy.glob("*.jpg")),
    "cataract": list(cataract.glob("*.jpg"))
    }


# In[9]:


# Define labels dictionary
labels_dict = {
    "normal": 0, "glaucoma": 1, "diabetic_retinopathy": 2, "cataract": 3
}


# In[10]:


# Load and preprocess images
X, y = [], []
for label, images in images_dict.items():
    for image in images:
        image = cv2.imread(str(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (180, 180))
        if image is not None:
            X.append(image)
            y.append(labels_dict[label])


# In[11]:


X = np.array(X) / 255.0
y = np.array(y)


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[14]:


data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomRotation(0.2),
    keras.layers.experimental.preprocessing.RandomContrast(0.3),
    keras.layers.experimental.preprocessing.RandomZoom(0.3),
    keras.layers.experimental.preprocessing.RandomZoom(0.7)
    ])


# In[15]:


model = keras.Sequential([
    data_augmentation,
    layers.Conv2D(64, (5, 5), padding="same", input_shape=(180, 180, 3), activation="softmax"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(16, (5, 5), padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(8, (5, 5), padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(50, activation="sigmoid"),
])


# In[16]:


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# In[17]:


# Train the model
model.fit(X_train, y_train, epochs=75)


# In[18]:


model.evaluate(X_test, y_test)


# In[19]:


with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[20]:


model.save('model.h5')


# In[ ]:




