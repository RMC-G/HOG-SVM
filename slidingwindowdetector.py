#!/usr/bin/env python
# coding: utf-8

# In[180]:


number_A = 1
number_B = 1


# In[181]:


import sys
import numpy as np
from numpy import dstack
import matplotlib.pyplot as plt
import matplotlib
from skimage import color
from skimage.feature import hog
from sklearn.svm import LinearSVC
import pickle
from PIL import Image       


# In[182]:


print("Loading image..")
filename = str("hog-svm-testimages/+ sys.argv[1])
im1 = np.array(Image.open(filename)) # Open the image and convert to a Numpy array.
#plt.figure("Testing image s05.jpg")   # And display.
#plt.imshow(im1)
#plt.show()
print("Opening classifiers..")
classifier64 = pickle.load(open("lin64.pkl","rb"))
classifier48 = pickle.load(open("lin48.pkl","rb"))
classifier40 = pickle.load(open("lin40.pkl","rb"))
classifier32 = pickle.load(open("lin32.pkl","rb"))


# In[183]:


def rectangle(size,x,y):
        color_list={64:'r',48:'g',40:'m',32:'y'}
        color= color_list.get(size)
        rect = matplotlib.patches.Rectangle((x,y),size,size, edgecolor=color, fill=False)
        return rect


# In[184]:


#takes in image and size;'slides' window across image returning
#array of HOG values and array of coordinates for each iteration
def slidingWindow(image,size):
    pixels=size/8
    height,width,c= image.shape
    windows, xy = [],[]
    #a larger step size greatly increases compilation time
    for y in range(0,(height-size),8):
        for x in range(0,(width-size),8):
            
            window=np.array(image[y:y+size,x:x+size],dtype='float32')
            hog_i = hog(window, orientations=9, # 9 orientation bins
                      pixels_per_cell=(pixels,pixels),     # variable pixel subregions
                       cells_per_block=(2,2),    # 2*2 subregion merge
                       visualize=False,           # show visualization
                       multichannel=True)        # input is RGB
            xy.append([x,y])
            windows.append(hog_i)
            #Call a function to check the value kernel versus the descriptors
    return windows, xy

            


# In[185]:


# Create lists of activations and coords
values = []
print("Calculating activation values..")
#creating arrays of activations and coordinates
hog_64, xy_64 = slidingWindow(im1,64)
activations_64 = classifier64.decision_function(hog_64)
cull_threshold = 0.3*np.max(activations_64)
for act in activations_64:
    if act >= cull_threshold:
        index = np.nonzero(activations_64==act)[0][0]
        values.append((xy_64[index][0],xy_64[index][1],act,64))


hog_48, xy_48 = slidingWindow(im1,48)
activations_48 = classifier48.decision_function(hog_48)
cull_threshold = 0.3*np.max(activations_48)
for act in activations_48:
    if act >= cull_threshold:
        index = np.nonzero(activations_48==act)[0][0]
        values.append((xy_48[index][0],xy_48[index][1],act,48))

        
        
hog_40, xy_40 = slidingWindow(im1,40)
activations_40 = classifier40.decision_function(hog_40)
cull_threshold = 0.3*np.max(activations_40)
for act in activations_40:
    if act >= cull_threshold:
        index = np.nonzero(activations_40==act)[0][0]
        values.append((xy_40[index][0],xy_40[index][1],act,40))

hog_32, xy_32 = slidingWindow(im1,32)
activations_32 = classifier32.decision_function(hog_32)
cull_threshold = 0.3*np.max(activations_32)
for act in activations_32:
    if act >= cull_threshold:
        index = np.nonzero(activations_32==act)[0][0]
        values.append((xy_32[index][0],xy_32[index][1],act,32))


# In[186]:


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[3], boxB[0]+boxB[3])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    else:
        return interArea
  


# In[187]:


data64 = []
data48 = []
data40 = []
data32 = []

def data_sort(values):
    for data in values:
        if data[3] == 64:
            data64.append(data)
        elif data[3] == 48:
            data48.append(data)
        elif data[3] == 40:
            data40.append(data)
        elif data[3] == 32:
            data32.append(data)
    return
data_sort(values)


# In[188]:


def recursive_function(array):
    temp_arr = []
    temp_leftover = []
    max_act = []

    # Iterate over all elements. Compare to all others
    if len(array) > 0:
        print("  x   y   Activation level   Size")
        temp_arr.append(array[0])
        for i in range(1,len(array)):  
            if bb_intersection_over_union(array[0],array[i]) > 0:
                temp_arr.append(array[i])
            else:
                temp_leftover.append(array[i])

        max_act = np.amax(temp_arr, axis = 0)      
        
        for var in temp_arr:
            if var[2] == max_act[2]:
                activations.append(var)
                
                print(var)
        if len(temp_leftover) > 0:
            recursive_function(temp_leftover)
        else:
            return
    else:
        return 


# In[189]:



def plotty(activations,string):
    global number_A
    global number_B
    if string == 'Best':
        number = number_B
    else:
        number = number_A
    
    figure2, ax2 = plt.subplots(1)
    # boundingBoxs = []
    for value in activations:
        box = rectangle(value[3],value[0],value[1])
        ax2.add_patch(box)
    ax2.imshow(im1)
    stringb=('%s_detections/Test_Image_%d.png'%(string,number))
    #plt.savefig(stringb)
    plt.show()
    if string == 'Best':
        number_B = number_B + 1
    else:
        number_A = number_A + 1


# In[190]:







print('Best Detections:')
activations = []
string = ('Best')
recursive_function(values)
plotty(activations,string)

print('All Detections:')
activations = []
string = ('All')
recursive_function(data32)
recursive_function(data40)
recursive_function(data48)
recursive_function(data64)

plotty(activations,string)
#return

