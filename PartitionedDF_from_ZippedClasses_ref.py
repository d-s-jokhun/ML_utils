# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:14:11 2020

@author: d.s.jokhun
"""
#%%
import pathlib
import operator
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing as mp


# #%% Define the root directory
# root_dir = os.path.abspath(r'\\kuehlapis.mbi.nus.edu.sg\home\jokhun\Pro 1\U2OS small mol screening\Segmented_SmallMol')
# print(root_dir)


# #%% Getting list of classes available
# root_dir = pathlib.Path(root_dir)
# classes_avail = sorted([zipped.name[:-4] for zipped in root_dir.glob('*.zip')])

# print('Classes available :\n',classes_avail)
# print('\nNo. of Classes available =',len(classes_avail))
    

#%% Shortlisting classes 


# Shortlist_Classes = True # Set to False in order to select all available classes
# selection = [slice(30300,len(classes_avail))] # List of classes to be selected. Only used if Select_Classes is True.

# if Shortlist_Classes:
#     selected_classes = sorted(list(operator.itemgetter(*selection)(classes_avail)))
# else:
#     selected_classes = classes_avail

# print('Classes selected :\n',selected_classes)
# print('\nNo. of Classes selected =',len(selected_classes))


#%% Determining number of files in each class
# def Class_size (a,b):
#     # with zipfile.ZipFile(root_dir.joinpath(Class+'.zip')) as archive:
#     #     Class_size = len(archive.namelist())-1
#     Class_size = a+b
#     return Class_size



#%%
import time


def kamal_add(a,b):
    return (a+b)


# def pool_add():
#     with mp.Pool() as pool:
#         a = pool.starmap(kamal_add, [(2,3),(4,5),(4,6)])
#     print(a)

# start = time.perf_counter()
# if __name__=="__main__":
#     with mp.Pool() as pool:
#         a = pool.starmap(kamal_add, [(2,3),(4,5),(4,6)])
#     print(a)
# print ('duration = ',time.perf_counter()-start,'s')

start = time.perf_counter()
if __name__=="__main__":
    with mp.Pool() as pool:
        a = pool.starmap(kamal_add, [(2,3),(4,5),(4,6)])
    print(a)
print ('duration = ',time.perf_counter()-start,'s')




#%%

# if __name__=="__main__":
#     with mp.Pool() as pool:
#         Class_sizes = pool.map(Class_size, selected_classes)
        
# #%%
# from multiprocessing import Pool

# def f(x):
#     return x*x

# if __name__ == '__main__':
#     # start 4 worker processes
#     with Pool(processes=4) as pool:
#         # print "[0, 1, 4,..., 81]"
#         a=pool.starmap(sum, [(2,3),(4,5),(3,4)])
        
# #%%
# import multiprocessing as mp
# from kamal_sum import kamal_sum

# #%%


# if __name__=="__main__":
#     with mp.Pool() as pool:
#         a = pool.starmap(kamal_sum, [(2,3),(4,5),(4,6)])
# print(a)

# # print(kamal_sum(5,3))

# #%%
# import multiprocessing

# def worker(num):
#     """Returns the string of interest"""
#     return "worker %d" % num

# def main():
#     pool = multiprocessing.Pool(4)
#     results = pool.map(worker, range(10))

#     pool.close()
#     pool.join()

#     for result in results:
#         # prints the result string in the main process
#         print(result)

# if __name__ == '__main__':
#     # Better protect your main function when you use multiprocessing
#     main()

# #%%


# Class_sizes = []
# for Class in selected_classes:
#     with zipfile.ZipFile(root_dir.joinpath(Class+'.zip')) as archive:
#         Class_sizes.append(len(archive.namelist())-1)
        
# plt.hist(Class_sizes, bins=None)
# plt.show()
# print('Class sizes :-\n',[Class+':'+str(Class_sizes[i]) for i,Class in enumerate(selected_classes)])

        
# #%% Filtering classes according to class size

# Filter_by_Size = True
# lower_SizeLim = 7
# upper_SizeLim = 10

# if Filter_by_Size:
#     filtered_classes = [Class for i,Class in enumerate(selected_classes) 
#                         if Class_sizes[i]>=lower_SizeLim and Class_sizes[i]<=upper_SizeLim]
#     filtered_Class_sizes = [size for size in Class_sizes if size>=lower_SizeLim and size<=upper_SizeLim]
# else:
#     filtered_classes = selected_classes
#     filtered_Class_sizes = Class_sizes

# plt.hist(filtered_Class_sizes, bins=None)
# plt.show()
# print('No. of classes selected :',len(filtered_classes))
# Min_Classes = [Class for i,Class in enumerate(filtered_classes) 
#                if filtered_Class_sizes[i]==np.amin(filtered_Class_sizes)]
# print('Min Class size:',np.amin(filtered_Class_sizes), Min_Classes)
# Max_Classes = [Class for i,Class in enumerate(filtered_classes) 
#                if filtered_Class_sizes[i]==np.amax(filtered_Class_sizes)]
# print('Max Class size:',np.amax(filtered_Class_sizes), Max_Classes)
# print('Class sizes :-\n',[Class+':'+str(filtered_Class_sizes[i]) for i,Class in enumerate(filtered_classes)])


# #%% Partitioning paths from each class into Train, Val and Test sets

# RanSeed = None
# Partition = [0.79,0.2,0.01]

# PartitionedPaths = DataPartition(selected_ClassPaths, Partition=Partition, RanSeed=RanSeed)
# random.seed(RanSeed)
# Tr_Paths=[]; Val_Paths=[]; Ts_Paths=[];
# for key in PartitionedPaths.keys():
#     Tr_Set,Val_Set,Ts_Set=PartitionedPaths[key]['Tr_Set'],PartitionedPaths[key]['Val_Set'],PartitionedPaths[key]['Ts_Set']
#     Tr_Paths.extend(Tr_Set), Val_Paths.extend(Val_Set), Ts_Paths.extend(Ts_Set)
# random.shuffle(Tr_Paths); random.shuffle(Val_Paths); random.shuffle(Ts_Paths);
# Tr_Paths=np.array(Tr_Paths); Val_Paths=np.array(Val_Paths); Ts_Paths=np.array(Ts_Paths);


        
        
        
# #%%
# images=[]
# with zipfile.ZipFile('BD-1047.zip') as archive:
#     img_filenames = [filename for filename in archive.namelist() if '.tif' in filename]
#     for img_filename in img_filenames:
#         images.append(np.array(Image.open(io.BytesIO(archive.read(img_filename)))))
# #%%
# archive=zipfile.ZipFile(root_dir.joinpath('class1.zip'))
# a=archive.namelist()






