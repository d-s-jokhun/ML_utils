#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import tensorflow as tf
import pathlib

# In[66]:


images=[]
with zipfile.ZipFile('BD-1047.zip') as archive:
    img_filenames = [filename for filename in archive.namelist() if '.tif' in filename]
    for img_filename in img_filenames:
        images.append(np.array(Image.open(io.BytesIO(archive.read(img_filename)))))


# In[71]:

images=np.array(images)
plt.imshow(images[0],cmap='gray')



# In[76]:

    
# In[4]:


ImageDataGen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=180,
    fill_mode="constant",
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=True,
)


ImageDataGen.flow_from_directory(
    './trial',
    target_size=(64, 64),
    color_mode="grayscale",
    classes=None,
    class_mode="categorical",
    batch_size=4,
    shuffle=True,
    seed=None,
    interpolation="nearest",
)


# In[6]:


data_dir=pathlib.Path('trial')



# In[ ]:
image_count = len(list(data_dir.glob('*/*.tif')))
print(image_count)

#%%
    
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)




