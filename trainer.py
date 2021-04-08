#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import LinearSVC
from scipy import ndimage, misc
from PIL import Image       # PIL is the "Python Imaging Library", it knows how to read jpegs.
import pickle


# In[10]:


#the training dataset is loaded in
dataset = np.load("hog-svm-train-dataset.64x64x3.npz")  # Load the npz file.

#the positive and negative datasets are loaded and concantenated
posex64 = dataset['posex']
negex64 = dataset['negex']
allex64 = np.concatenate((posex64, negex64))
#the process is repeated with scaled versions of the input images
#48x48
posex48 = ndimage.zoom(posex64, [1, 0.75, 0.75, 1])
negex48 = ndimage.zoom(negex64, [1, 0.75, 0.75, 1])
allex48 = np.concatenate((posex48, negex48))
#40x40
posex40 = ndimage.zoom(posex64, [1, 0.625, 0.625, 1])
negex40 = ndimage.zoom(negex64, [1, 0.625, 0.625, 1])
allex40 = np.concatenate((posex40, negex40))
#32x32
posex32 = ndimage.zoom(posex64, [1, 0.5, 0.5, 1])
negex32 = ndimage.zoom(negex64, [1, 0.5, 0.5, 1])
allex32 = np.concatenate((posex32, negex32))


# In[14]:



    
all_hog64 = []
all_hog48 = []
all_hog40 = []
all_hog32 = []
all_labels = np.concatenate((np.ones(len(posex64)),-np.ones(len(negex64))))


for i in range(len(allex64)):
    hog_i = hog(allex64[i], orientations=9, # 9 orientation bins
                      pixels_per_cell=(8,8),     # 8*8 pixel subregions
                       cells_per_block=(2,2),    # 2*2 subregion merge
                       visualize=False,           # show visualization
                       multichannel=True)        # input is RGB
    
    all_hog64.append(hog_i)
    
    
for i in range(len(allex48)):
    hog_i = hog(allex48[i], orientations=9, # 9 orientation bins
                      pixels_per_cell=(6,6),     # 8*8 pixel subregions
                       cells_per_block=(2,2),    # 2*2 subregion merge
                       visualize=False,           # show visualization
                       multichannel=True)        # input is RGB
    
    all_hog48.append(hog_i)
    
    
for i in range(len(allex40)):
    hog_i= hog(allex40[i], orientations=9, # 9 orientation bins
                      pixels_per_cell=(5,5),     # 8*8 pixel subregions
                       cells_per_block=(2,2),    # 2*2 subregion merge
                       visualize=False,           # show visualization
                       multichannel=True)        # input is RGB
    
    all_hog40.append(hog_i)
    
    
for i in range(len(allex32)):
    hog_i= hog(allex32[i], orientations=9, # 9 orientation bins
                      pixels_per_cell=(4,4),     # 8*8 pixel subregions
                       cells_per_block=(2,2),    # 2*2 subregion merge
                       visualize=False,           # show visualization
                       multichannel=True)        # input is RGB
    
    all_hog32.append(hog_i)
    


# In[16]:



classifier64= LinearSVC()
classifier48= LinearSVC()
classifier40= LinearSVC()
classifier32= LinearSVC()

classifier64.fit(all_hog64, all_labels)
classifier48.fit(all_hog48, all_labels)
classifier40.fit(all_hog40, all_labels)
classifier32.fit(all_hog32, all_labels)

pickle.dump(classifier64, open("lin64.pkl","wb"))
pickle.dump(classifier48, open("lin48.pkl","wb"))
pickle.dump(classifier40, open("lin40.pkl","wb"))
pickle.dump(classifier32, open("lin32.pkl","wb"))


# In[ ]:




