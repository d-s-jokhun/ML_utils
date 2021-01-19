
#%%

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from Dataset_from_Dataframe import Dataset_from_Dataframe
import numpy as np


#%% Defining paths
DF_Tr_Path = "/gpfs0/home/jokhun/Pro 1/U2OS small mol screening/ds_17cls_3985-4000_Tr.csv.xz"
DF_Val_Path = "/gpfs0/home/jokhun/Pro 1/U2OS small mol screening/ds_17cls_3985-4000_Val.csv.xz"
Img_dir = '/gpfs0/home/jokhun/Pro 1/U2OS small mol screening/Segmented_3Ch'


#%% Loading Dataframes

Tr_cache_path = os.path.join(os.path.dirname(DF_Tr_Path),os.path.basename(DF_Tr_Path).split('.')[0]+'_Cached')
Val_cache_path = os.path.join(os.path.dirname(DF_Val_Path),os.path.basename(DF_Val_Path).split('.')[0]+'_Cached')

print (f'Cache path:\n{Tr_cache_path}\n{Val_cache_path}')

DataFrame_Tr = pd.read_csv(DF_Tr_Path)
DataFrame_Val = pd.read_csv(DF_Val_Path)

Classes_Tr = pd.DataFrame([os.path.dirname(Class)[2:-4] for Class in DataFrame_Tr['rel_path']],columns=['Classes'])
Classes_Val = pd.DataFrame([os.path.dirname(Class)[2:-4] for Class in DataFrame_Val['rel_path']],columns=['Classes'])

print ('Training & validation dataframes loaded!')
print (f'Train & Val dataset sizes:\n {DataFrame_Tr.shape[0]}, {DataFrame_Val.shape[0]}')


#%% Encoding response

ResponseEncoder = LabelEncoder()
ResponseEncoder.fit(Classes_Tr['Classes'].append(Classes_Val['Classes']))

class_names = ResponseEncoder.classes_
NumOfClasses = len(class_names)
print('Number of classes in the data: '+str(NumOfClasses))

DataFrame_Tr = pd.concat([DataFrame_Tr['rel_path'],\
    pd.DataFrame(ResponseEncoder.transform(Classes_Tr['Classes']),columns=['label'])\
        ],axis=1) 
DataFrame_Val = pd.concat([DataFrame_Val['rel_path'],\
    pd.DataFrame(ResponseEncoder.transform(Classes_Val['Classes']),columns=['label'])\
        ],axis=1)

print('\n1st rel_path of DataFrame_Tr:',DataFrame_Tr['rel_path'][0])
print('1st Class of Training dataframe:',Classes_Tr['Classes'][0])
print('1st label of DataFrame_Tr:',DataFrame_Tr['label'][0])
print('Decoded class from 1st label of DataFrame_Tr:',\
    ResponseEncoder.inverse_transform([DataFrame_Tr['label'][0]])[0])


#%% Instantiating datasets
batch_size = 8192

Dataset_Tr = Dataset_from_Dataframe(dataframe=DataFrame_Tr,Img_dir=Img_dir,\
    batch_size=batch_size,shuffle=False,shuffle_buffer_size=None,\
        cache_path=Tr_cache_path)        
Dataset_Val = Dataset_from_Dataframe(dataframe=DataFrame_Val,Img_dir=Img_dir,\
    batch_size=batch_size,shuffle=False,shuffle_buffer_size=None,\
        cache_path=Val_cache_path)
print('Datasets created!')


#%% Caching training dataset

Num_of_TrSteps = int(np.ceil(DataFrame_Tr.shape[0]/batch_size))
step = 0
for batch in Dataset_Tr:
    step+=1
    print (f'Training dataset: Step {step} of {Num_of_TrSteps}')

#%% Caching validation dataset

Num_of_ValSteps = int(np.ceil(DataFrame_Val.shape[0]/batch_size))
step = 0
for batch in Dataset_Val:
    step+=1
    print (f'Validation dataset: Step {step} of {Num_of_ValSteps}')



#%%

print ('Dataset Cached!')


