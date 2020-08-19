#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import operator
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing as mp


# ### Define the root directory

# In[2]:


root_dir = os.path.abspath(r'\\kuehlapis.mbi.nus.edu.sg\home\jokhun\Pro 1\U2OS small mol screening\Segmented_SmallMol')
print(root_dir)


# ### Getting list of classes available

# In[3]:


root_dir = pathlib.Path(root_dir)
classes_avail = sorted([zipped.name[:-4] for zipped in root_dir.glob('*.zip')])

print('Classes available :\n',classes_avail)
print('\nNo. of Classes available =',len(classes_avail))


# ### Shortlisting classes

# In[4]:


Shortlist_Classes = True # Set to False in order to select all available classes
selection = [slice(30300,len(classes_avail))] # List of classes to be selected. Only used if Select_Classes is True.

if Shortlist_Classes:
    selected_classes = sorted(list(operator.itemgetter(*selection)(classes_avail)))
else:
    selected_classes = classes_avail

print('Classes selected :\n',selected_classes)
print('\nNo. of Classes selected =',len(selected_classes))


# ### Determining number of files in each class

# In[5]:


def Class_size (Class):
#     with zipfile.ZipFile(root_dir.joinpath(Class+'.zip')) as archive:
#         Class_size = len(archive.namelist())-1
#     print (Class)
    return Class


# In[ ]:


# Class_sizes = []
# for Class in selected_classes:
#     with zipfile.ZipFile(root_dir.joinpath(Class+'.zip')) as archive:
#         Class_sizes.append(len(archive.namelist())-1)

if __name__=="__main__":
    mp.freeze_support()
    with mp.Pool() as pool:
        Class_sizes = pool.map(Class_size, selected_classes)

# plt.hist(Class_sizes, bins=None)
# plt.show()
# print('Class sizes :-\n',[Class+':'+str(Class_sizes[i]) for i,Class in enumerate(selected_classes)])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




