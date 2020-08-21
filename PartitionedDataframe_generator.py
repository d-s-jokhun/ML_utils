#!/usr/bin/env python
# coding: utf-8

# In[112]:


import os
import operator
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from DataPartition import DataPartition
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys


# ### Searching for available classes

# In[66]:


get_labels_from = 'folders' # 'Filenames' or 'Folders'

# data_path = os.path.abspath("/gpfs0/home/jokhun/Pro 1/U2OS small mol screening/Segmented_SmallMol")
# data_path = os.path.abspath('//deptnas.nus.edu.sg/BIE/MBELab/jokhun/Pro 1/U2OS small mol screening/Segmented_SmallMol/')
data_path = os.path.abspath('/MBELab/jokhun/Pro 1/U2OS small mol screening/Segmented/')

if get_labels_from.lower() == 'filenames':
    filenames = [filename for filename in os.listdir(data_path) if os.path.isfile(os.path.join(data_path,filename))]
    classes_avail = sorted(list(set([filename[filename.rindex('_') + 1 : filename.index('.tif')] 
                                     for filename in filenames])))
    print('No. of Classes available =',len(classes_avail))
    print('\nClasses available :\n',classes_avail)
elif get_labels_from.lower() == 'folders':
    classes_avail = sorted([Class for Class in os.listdir(data_path)
                            if os.path.isdir(os.path.join(data_path,Class))])
    print('No. of Classes available =',len(classes_avail))
    print('\nClasses available :\n',classes_avail)
else: sys.exit("Invalid 'get_labels_from'")


# #### Selecting classes

# In[67]:


Select_Classes = True # Set to False in order to select all available classes
selection = list(range(1,len(classes_avail))) # List of classes to be selected. Only used if Select_Classes is True.

if Select_Classes:
    selected_classes = list(operator.itemgetter(*selection)(classes_avail))
else:
    selected_classes = classes_avail

print('No. of Classes available =',len(selected_classes))
print('\nClasses available :\n',selected_classes)


# #### Getting paths and determining number of files in each calss

# In[68]:


ClassPaths={}
selected_classes = sorted(selected_classes)
for Class in selected_classes:
    if get_labels_from.lower() == 'filenames':
        ClassPaths[Class]=sorted(glob(os.path.join(
            data_path,f"*_{Class}.tif"
        )))
    elif get_labels_from.lower() == 'folders':
        ClassPaths[Class]=sorted(glob(os.path.join(
            data_path,Class,"*.tif"
        )))
    else: sys.exit("Invalid 'get_labels_from'")
Class_sizes = [len(ClassPaths[Class]) for Class in ClassPaths.keys()]
plt.hist(Class_sizes, bins=None)
plt.show()
print('Class sizes :-\n',[Class+':'+str(len(ClassPaths[Class])) for Class in ClassPaths.keys()])


# ## Filtering classes according to class size

# In[69]:


Filter_by_Size = True
lower_SizeLim = 1000
upper_SizeLim = 7000

if Filter_by_Size:
    selected_ClassPaths = {}
    for Class in ClassPaths.keys():
        if len(ClassPaths[Class])>=lower_SizeLim and len(ClassPaths[Class])<=upper_SizeLim:
            selected_ClassPaths[Class] = ClassPaths[Class]
else:
    selected_ClassPaths = ClassPaths
    
Class_sizes = [len(selected_ClassPaths[Class]) for Class in selected_ClassPaths.keys()]
plt.hist(Class_sizes, bins=None)
plt.show()
print('No. of classes selected :',len(selected_ClassPaths.keys()))
Class_sizes = [len(selected_ClassPaths[Class]) for Class in selected_ClassPaths.keys()]
Min_Classes = [Class for Class in selected_ClassPaths.keys() if len(selected_ClassPaths[Class])==np.amin(Class_sizes)]
print('Min Class size:',np.amin(Class_sizes), Min_Classes)
Max_Classes = [Class for Class in selected_ClassPaths.keys() if len(selected_ClassPaths[Class])==np.amax(Class_sizes)]
print('Max Class size:',np.amax(Class_sizes), Max_Classes)
print('Class sizes :-\n',[Class+':'+str(len(selected_ClassPaths[Class])) for Class in selected_ClassPaths.keys()])


# ## Partitioning paths from each class into Train, Val and Test sets

# In[70]:


RanSeed = None
Partition = [0.79,0.2,0.01]

PartitionedPaths = DataPartition(selected_ClassPaths, Partition=Partition, RanSeed=RanSeed)
random.seed(RanSeed)
Tr_Paths=[]; Val_Paths=[]; Ts_Paths=[];
for key in PartitionedPaths.keys():
    Tr_Set,Val_Set,Ts_Set=PartitionedPaths[key]['Tr_Set'],PartitionedPaths[key]['Val_Set'],PartitionedPaths[key]['Ts_Set']
    Tr_Paths.extend(Tr_Set), Val_Paths.extend(Val_Set), Ts_Paths.extend(Ts_Set)
random.shuffle(Tr_Paths); random.shuffle(Val_Paths); random.shuffle(Ts_Paths);
Tr_Paths=np.array(Tr_Paths); Val_Paths=np.array(Val_Paths); Ts_Paths=np.array(Ts_Paths);

if get_labels_from.lower() == 'filenames':
    Tr_Y = [path[path.rindex('_') + 1 : path.index('.tif')] for path in Tr_Paths]
    Val_Y = [path[path.rindex('_') + 1 : path.index('.tif')] for path in Val_Paths]
    Ts_Y = [path[path.rindex('_') + 1 : path.index('.tif')] for path in Ts_Paths]
elif get_labels_from.lower() == 'folders':
    Tr_Y = [os.path.basename(os.path.dirname(path)) for path in Tr_Paths]
    Val_Y = [os.path.basename(os.path.dirname(path)) for path in Val_Paths]
    Ts_Y = [os.path.basename(os.path.dirname(path)) for path in Ts_Paths]
else: sys.exit("Invalid 'get_labels_from'")
    
print ('Total number of paths = ' + str(len(Tr_Paths)+len(Val_Paths)+len(Ts_Paths)))
print ('\nLength of Training Set = ' + str(len(Tr_Paths)))
values, counts = np.unique(Tr_Y, return_counts=True)
Dis_counts = [len(selected_ClassPaths[Class])-(round(np.amin(Class_sizes)*Partition[2])+round(np.amin(Class_sizes)*Partition[1])) for Class in selected_ClassPaths.keys()]
print ('Classes in Training Set : ' + str(values) + '\n -Frequencies : ' + str(counts[0]) + '\n -Min Distinct Frequencies : ' + str(np.amin(Dis_counts)) + '\n -Distinct Frequencies : ' + str(Dis_counts))
print ('\nLength of Validation Set = ' + str(len(Val_Paths)))
values, counts = np.unique(Val_Y, return_counts=True)
Dis_counts = round(np.amin(Class_sizes)*Partition[1])
print ('Classes in Validation Set : ' + str(values) + '\n -Frequencies : ' + str(counts[0]) + '\n -Distinct Frequencies : ' + str(Dis_counts))
print ('\nLength of Test Set = ' + str(len(Ts_Paths)))
values, counts = np.unique(Ts_Y, return_counts=True)
Dis_counts = round(np.amin(Class_sizes)*Partition[2])
print ('Classes in Test Set : ' + str(values) + '\n -Frequencies : ' + str(counts[0]) + '\n -Distinct Frequencies : ' + str(Dis_counts))

print (f'\n1st element of Training Set : {Tr_Y[0]}\n' + str(Tr_Paths[0]))
print (f'1st element of Validation Set : {Val_Y[0]}\n' + str(Val_Paths[0]))
print (f'1st element of Test Set : {Ts_Y[0]}\n' + str(Ts_Paths[0]))


# # Encoding the classes into labels

# In[71]:


ResponseEncoder = LabelEncoder()
ResponseEncoder.fit(list(Tr_Y) + list(Val_Y) + list(Ts_Y))
class_names = ResponseEncoder.classes_
NumOfClasses = len(class_names)
print('Number of calsses in the data: '+str(NumOfClasses))
print('Classes in the Data: ' + str(class_names))
Y_Train = ResponseEncoder.transform(Tr_Y)
Y_Val = ResponseEncoder.transform(Val_Y)
Y_Test = ResponseEncoder.transform(Ts_Y)
print ('1st element of Tr_Y, Val_Y and Ts_Y : ' + str(Tr_Y[0]) + ', ' + str(Val_Y[0]) + ', ' + str(Ts_Y[0]))
print ('1st element of Y_Train, Y_Val and Y_Test : ' + str(Y_Train[0]) + ', ' + str(Y_Val[0]) + ', ' + str(Y_Test[0]))


# ## Converting paths to relative

# In[121]:


if get_labels_from.lower() == 'filenames':
    rel_Tr_Paths = [os.path.basename(path) for path in Tr_Paths]
    rel_Val_Paths = [os.path.basename(path) for path in Val_Paths]
    rel_Ts_Paths = [os.path.basename(path) for path in Ts_Paths]
elif get_labels_from.lower() == 'folders':
    rel_Tr_Paths = [os.path.join(os.path.basename(os.path.dirname(path)),os.path.basename(path)) for path in Tr_Paths]
    rel_Val_Paths = [os.path.join(os.path.basename(os.path.dirname(path)),os.path.basename(path)) for path in Val_Paths]
    rel_Ts_Paths = [os.path.join(os.path.basename(os.path.dirname(path)),os.path.basename(path)) for path in Ts_Paths]
else: sys.exit("Invalid 'get_labels_from'")


# # Creating dataframes

# In[122]:


df_Train = pd.DataFrame(zip(rel_Tr_Paths,Y_Train,Tr_Y), columns = ['rel_path', 'label', 'class']) 
df_Val = pd.DataFrame(zip(rel_Val_Paths,Y_Val,Val_Y), columns = ['rel_path', 'label', 'class']) 
df_Test = pd.DataFrame(zip(rel_Ts_Paths,Y_Test,Ts_Y), columns = ['rel_path', 'label', 'class']) 


# In[ ]:





# In[ ]:





# In[ ]:




