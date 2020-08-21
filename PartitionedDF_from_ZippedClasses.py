#!/usr/bin/env python
# coding: utf-8

#%% 
import pathlib
import operator
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing as mp
import pandas as pd
from DataPartition import DataPartition
import random
from sklearn.preprocessing import LabelEncoder


#%% Custome functions

def namelist_in_archive (archive):
    with zipfile.ZipFile(archive) as archive:
        namelist = archive.namelist()
    return namelist


def AddFolder2Files (folder_str, filenames_ls):
    paths = [os.path.join(folder_str,filename) for filename in filenames_ls]
    return paths


#%% Define the root directory

root_dir = os.path.abspath(r"/gpfs0/home/jokhun/Pro 1/U2OS small mol screening/Segmented_SmallMol")
# root_dir = os.path.abspath(r'\\kuehlapis.mbi.nus.edu.sg\home\jokhun\Pro 1\U2OS small mol screening\Segmented_SmallMol')
print(root_dir)


#%% Getting list of available classes

root_dir = pathlib.Path(root_dir)
classes_avail = sorted([zipped.name[:-4] for zipped in root_dir.glob('*.zip')])

# print('Classes available :\n',classes_avail)
print('\nNo. of Classes available =',len(classes_avail))


#%% Shortlisting classes to consider

Shortlist_Classes = True # Set to False in order to select all available classes
# shortlist_idx = [slice(30300,len(classes_avail))] # List of classes to be selected. Only used if Select_Classes is True.
exclusion_idx = [classes_avail.index('DMSO_complete'),\
    classes_avail.index('DMSO_2000'),\
        classes_avail.index('DMSO_3000')]
shortlist_idx = [i for i in range(len(classes_avail)) if i not in exclusion_idx]

if Shortlist_Classes:
    shortlisted_classes = sorted(list(operator.itemgetter(*shortlist_idx)(classes_avail)))
else:
    shortlisted_classes = classes_avail

# print('Classes shortlisted :\n',shortlisted_classes)
print('\nNo. of Classes shortlisted =',len(shortlisted_classes))


#%% Determining the size of shortlisted classes

if __name__=='__main__':
    class_paths = [root_dir.joinpath(Class+'.zip') for Class in shortlisted_classes]
    with mp.Pool() as pool:
        Class_sizes = pool.map(len, pool.map(namelist_in_archive, class_paths))
    print('Class sizes determined!')


#%% Define the limits on number of files for each class and view the resulting class size distribution

lower_SizeLim = 3000 # use None to remove limits
upper_SizeLim = 6000 #int(round(np.percentile(Class_sizes,99.99))) # use None to remove limits

print(f"Limits chosen: ({lower_SizeLim},{upper_SizeLim})")
if type(lower_SizeLim)==type(None): lower_SizeLim=np.amin(Class_sizes)
if type(upper_SizeLim)==type(None): upper_SizeLim=np.amax(Class_sizes)
selected_ClassSizes=[]; selected_classes=[]; lower_exclu_ls=[]; upper_exclu_ls=[]
for i,size in enumerate(Class_sizes):
    if size<lower_SizeLim: 
        lower_exclu_ls.append((shortlisted_classes[i],size))
    elif size>upper_SizeLim: 
        upper_exclu_ls.append((shortlisted_classes[i],size))
    else: 
        selected_classes.append (shortlisted_classes[i])
        selected_ClassSizes.append (size)
if len(lower_exclu_ls)>1: lower_exclu_ls.sort(key=lambda x: x[1])
if len(upper_exclu_ls)>1: upper_exclu_ls.sort(key=lambda x: x[1])

print ('Length of lower exclusion list =',len(lower_exclu_ls))
if len(lower_exclu_ls)>0: print('Largest lower excluded class:',lower_exclu_ls[-1])
print ('Length of upper exclusion list =',len(upper_exclu_ls))
if len(upper_exclu_ls)>0: print('Smallest upper excluded class:',upper_exclu_ls[0])

plt.hist(selected_ClassSizes, bins=100)
plt.show()
min_size = np.amin(selected_ClassSizes); max_size = np.amax(selected_ClassSizes); min_list= []; max_list = []
for i,size in enumerate(selected_ClassSizes):
    if size==min_size: min_list.append(selected_classes[i]+':'+str(size))
    elif size==max_size: max_list.append(selected_classes[i]+':'+str(size))
print('No. of classes selected:',len(selected_classes))
print('Smallest classes:-\n',min_list,'\n','Largest classes:-\n',max_list)
# print('Class sizes :-\n',[Class+':'+str(selected_ClassSizes[i]) for i,Class in enumerate(selected_classes)])


#%% Getting all the paths

if __name__=='__main__':
    selected_ClassPaths = [root_dir.joinpath(Class+'.zip') for Class in selected_classes]
    with mp.Pool() as pool:
        selected_ClassPaths = pool.map(namelist_in_archive,selected_ClassPaths)
        selected_ClassPaths = pool.starmap(AddFolder2Files, \
            zip([Class+'.zip' for Class in selected_classes], \
                selected_ClassPaths))
    selected_ClassPaths = dict(zip(selected_classes,selected_ClassPaths))
    print(f'{sum([len(selected_ClassPaths[Class]) for Class in selected_ClassPaths.keys()])} filepaths acquired!')


# %% Partitioning paths from each class into Train, Val and Test sets

RanSeed = None
Partition = [0.8, 0.15, 0.05]

print ('Data Partition [Tr, Val, Ts]:', Partition)
PartitionedPaths = DataPartition(selected_ClassPaths, Partition=Partition, RanSeed=RanSeed)
random.seed(RanSeed)
Tr_Paths=[]; Val_Paths=[]; Ts_Paths=[]
for key in PartitionedPaths.keys():
    Tr_Set,Val_Set,Ts_Set=PartitionedPaths[key]['Tr_Set'],PartitionedPaths[key]['Val_Set'],PartitionedPaths[key]['Ts_Set']
    Tr_Paths.extend(Tr_Set), Val_Paths.extend(Val_Set), Ts_Paths.extend(Ts_Set)
random.shuffle(Tr_Paths); random.shuffle(Val_Paths); random.shuffle(Ts_Paths)
Tr_Paths=np.array(Tr_Paths); Val_Paths=np.array(Val_Paths); Ts_Paths=np.array(Ts_Paths)

# Obtaining Classes
Tr_Y = [os.path.dirname(path)[:-4] for path in Tr_Paths]
Val_Y = [os.path.dirname(path)[:-4] for path in Val_Paths]
Ts_Y = [os.path.dirname(path)[:-4] for path in Ts_Paths]

print ('Total number of paths = ' + str(len(Tr_Paths)+len(Val_Paths)+len(Ts_Paths)))

print ('\nLength of Training Set = ' + str(len(Tr_Paths)))
values, counts = np.unique(Tr_Y, return_counts=True)
Dis_counts = [len(selected_ClassPaths[Class])-(round(np.amin(selected_ClassSizes)*Partition[2])+round(np.amin(selected_ClassSizes)*Partition[1])) for Class in selected_ClassPaths.keys()]
print ('No. of classes in Training Set :',len(values))
# print ('Classes in Training Set : ' + str(values))
print (' -Frequencies : ' + str(counts[0]))
print (' -Min Distinct Frequencies : ' + str(np.amin(Dis_counts)))
# print (' -Distinct Frequencies : ' + str(Dis_counts))
print ('\nLength of Validation Set = ' + str(len(Val_Paths)))
values, counts = np.unique(Val_Y, return_counts=True)
Dis_counts = round(np.amin(selected_ClassSizes)*Partition[1])
print ('No. of classes in Validation Set : ',len(values))
# print ('Classes in Validation Set : ' + str(values))
print (' -Frequencies : ' + str(counts[0]))
print (' -Distinct Frequencies : ' + str(Dis_counts))
print ('\nLength of Test Set = ' + str(len(Ts_Paths)))
values, counts = np.unique(Ts_Y, return_counts=True)
Dis_counts = round(np.amin(selected_ClassSizes)*Partition[2])
print ('No. of classes in Test Set : ',len(values))
# print ('Classes in Test Set : ' + str(values))
print (' -Frequencies : ' + str(counts[0]))
print (' -Distinct Frequencies : ' + str(Dis_counts))

print (f'\n1st element of Training Set : {Tr_Y[0]}\n' + str(Tr_Paths[0]))
print (f'1st element of Validation Set : {Val_Y[0]}\n' + str(Val_Paths[0]))
print (f'1st element of Test Set : {Ts_Y[0]}\n' + str(Ts_Paths[0]))


#%% Encoding classes into labels

ResponseEncoder = LabelEncoder()
ResponseEncoder.fit(list(Tr_Y) + list(Val_Y) + list(Ts_Y))
class_names = ResponseEncoder.classes_
NumOfClasses = len(class_names)
print('Number of calsses in the data: '+str(NumOfClasses))
# print('Classes in the Data: ' + str(class_names))
Y_Train = ResponseEncoder.transform(Tr_Y)
Y_Val = ResponseEncoder.transform(Val_Y)
Y_Test = ResponseEncoder.transform(Ts_Y)
print ('1st element of Tr_Y, Val_Y and Ts_Y : ' + str(Tr_Y[0]) + ', ' + str(Val_Y[0]) + ', ' + str(Ts_Y[0]))
print ('1st element of Y_Train, Y_Val and Y_Test : ' + str(Y_Train[0]) + ', ' + str(Y_Val[0]) + ', ' + str(Y_Test[0]))


#%% Creating dataframes

df_Train = pd.DataFrame(zip(Tr_Paths,Y_Train,Tr_Y), columns = ['rel_path', 'label', 'class']) 
df_Val = pd.DataFrame(zip(Val_Paths,Y_Val,Val_Y), columns = ['rel_path', 'label', 'class']) 
df_Test = pd.DataFrame(zip(Ts_Paths,Y_Test,Ts_Y), columns = ['rel_path', 'label', 'class']) 
print ('Dataframes created!')


#%%



