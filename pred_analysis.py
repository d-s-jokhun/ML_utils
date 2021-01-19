

#%%
import pandas as pd
import itertools
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
import numpy as np
from plot_ConfMat import plot_ConfMat
import matplotlib.pyplot as plt
import pathlib


#%%
# if True:
#     pred_path = pred_paths[0]

def pred_analysis (pred_path):

    #%% Importing class_names, ground_truth and predictions

    data = itertools.zip_longest(*csv.reader(open(pred_path, "r")), fillvalue=None)
    header = data.__next__()
    data = pd.DataFrame.from_records(data,columns = header)
    class_names = data.iloc[:,0]
    class_names.replace('', np.nan, inplace=True)
    class_names.dropna(inplace=True)
    data =  data.iloc[:,1:].astype(np.int64)


    #%% Preliminary analysis

    classes_2_Consider = class_names #class_names[25:50] #['17-PA','4-IBP']

    tmp_index = pd.Index(class_names)
    lbl_being_considered = [tmp_index.get_loc(class_2_Consider) for class_2_Consider in classes_2_Consider]
    tmp_index = pd.Index(data['ground_truth'])
    idx_being_considered=np.zeros(data['ground_truth'].shape)
    for lbl in lbl_being_considered:
        idx_being_considered+=tmp_index.get_loc(lbl)
    idx_being_considered=idx_being_considered.astype(bool)

    Y_True = data['ground_truth'][idx_being_considered]
    # Y_True = class_names[data['ground_truth'][idx_being_considered]]

    #%%

    precision = []
    recall = []
    f1Score = []
    r2 = []

    for time_pt in range (1,data.shape[1]):
        Y_Pred = data.iloc[:,time_pt][idx_being_considered]
        # Y_Pred = class_names[data.iloc[:,-1][idx_being_considered]]

        model_report = classification_report(Y_True, Y_Pred,\
            labels=lbl_being_considered, target_names=classes_2_Consider, output_dict=True)
        model_report_df = pd.DataFrame.from_dict(model_report, orient='columns')\
            .transpose()
        precision.append(model_report_df['precision'])
        recall.append(model_report_df['recall'])
        f1Score.append(model_report_df['f1-score'])

        r2.append(r2_score(Y_True, Y_Pred))

    precision_df = pd.DataFrame(precision,columns=model_report_df.index, index=data.columns[1:time_pt+1])
    recall_df = pd.DataFrame(recall,columns=model_report_df.index, index=data.columns[1:time_pt+1])
    f1Score_df = pd.DataFrame(f1Score,columns=model_report_df.index, index=data.columns[1:time_pt+1])
    mdlSummary_df = pd.concat([precision_df.iloc[:,-3:],recall_df.iloc[:,-2:],f1Score_df.iloc[:,-2:],pd.DataFrame(r2,columns=['R^2 score'],index=data.columns[1:time_pt+1])], axis=1)
    mdlSummary_df.columns=['Accuracy','macro avg Precision','weighted avg Precision','macro avg Recall','weighted avg Recall','macro avg f1-score','weighted avg f1-score','R^2 score']

    plt.figure()
    ax = plt.subplot()
    ax.plot(mdlSummary_df.index, mdlSummary_df['Accuracy'], c='black', label='Accuracy')
    ax.plot(mdlSummary_df.index, mdlSummary_df['macro avg Precision'], c='blue', label='Specificity (Precision)')
    ax.plot(mdlSummary_df.index, mdlSummary_df['macro avg Recall'], c='green', label='Sensitivity (Recall)')
    ax.plot(mdlSummary_df.index, mdlSummary_df['macro avg f1-score'], c='red', label='f1-score')
    # ax.plot(mdlSummary_df.index, mdlSummary_df['R^2 score'], c='purple', label='R^2-score')
    ax.legend()
    plt.xticks(ticks=range(0,len(plt.xticks()[0]),20),rotation=30)


    max_f1_idx = mdlSummary_df['macro avg f1-score'].iloc[::-1]
    max_f1_idx = max_f1_idx.idxmax(axis = 0)
    max_Acc_idx = mdlSummary_df['Accuracy'].iloc[::-1]
    max_Acc_idx = max_Acc_idx.idxmax(axis = 0)

    max_f1Score_mdlwise = pd.DataFrame(mdlSummary_df.loc[max_f1_idx,:]).transpose()
    max_Acc_mdlwise = pd.DataFrame(mdlSummary_df.loc[max_Acc_idx,:]).transpose()

    Y_Pred = data[max_f1_idx][idx_being_considered]
    ConfMat = confusion_matrix(Y_True, Y_Pred,\
        sample_weight=None, normalize='true')
    plot = plot_ConfMat(ConfMat,class_names=None)#classes_2_Consider)

    Y_Pred = data[max_Acc_idx][idx_being_considered]
    ConfMat = confusion_matrix(Y_True, Y_Pred,\
        sample_weight=None, normalize='true')
    plot = plot_ConfMat(ConfMat,class_names=None)#classes_2_Consider)



    # %% Classwise analysis

    max_f1Score_Classwise = pd.DataFrame(columns=classes_2_Consider, index=[0])

    max_f1_ConfMat = {}
    for i,class_ in enumerate(classes_2_Consider):

        # fig = plt.figure()
        # plt.title(class_)
        # plt.plot(precision_df.index, precision_df[class_], c='blue', label='Specificity (Precision)')
        # plt.plot(recall_df.index, recall_df[class_], c='green', label='Sensitivity (Recall)')
        # plt.plot(f1Score_df.index, f1Score_df[class_], c='red', label='f1-score')
        # # ax1.plot(mdlSummary_df.index, mdlSummary_df['R^2 score'], c='purple', label='R^2-score')
        # plt.legend()
        # plt.xticks(ticks=range(0,len(plt.xticks()[0]),20),rotation=30)

        max_f1Score_idx = f1Score_df[class_].iloc[::-1]
        max_f1Score_idx = max_f1Score_idx.idxmax(axis = 0)
        Y_Pred = data[max_f1Score_idx][idx_being_considered]
        MultiClass_ConfMat = multilabel_confusion_matrix(Y_True, Y_Pred, labels=lbl_being_considered)
        max_f1_ConfMat [class_] = MultiClass_ConfMat[i,:,:]
        SampleSize_0 = sum(max_f1_ConfMat [class_].ravel()[:2])
        SampleSize_1 = sum(max_f1_ConfMat [class_].ravel()[2:])
        # plot = plot_ConfMat(max_f1_ConfMat[class_]/[[SampleSize_0],[SampleSize_1]],class_names=None)#classes_2_Consider)
        # plot.set_size_inches(3*plot.get_size_inches())
        # plt.title(class_)

        max_f1Score_Classwise[class_][0]=f1Score_df[class_][max_f1Score_idx]

    return max_f1Score_mdlwise, max_Acc_mdlwise, max_f1Score_Classwise
    
    


# %%

if __name__=='__main__':
    
    pred_paths = list(pathlib.Path('/gpfs0/home/jokhun/Pro 1/old models').rglob('*.csv'))
    
    max_f1Score_mdlwise=[]; max_Acc_mdlwise=[]; max_f1Score_Classwise=[]
    for pred_path in pred_paths:
        pred_anal_result=pred_analysis(pred_path)
        max_f1Score_mdlwise.append(pred_anal_result[0])
        max_Acc_mdlwise.append(pred_anal_result[1])
        max_f1Score_Classwise.append(pred_anal_result[2])
    max_f1Score_mdlwise = pd.concat(max_f1Score_mdlwise,ignore_index=True)
    max_Acc_mdlwise = pd.concat(max_Acc_mdlwise,ignore_index=True)
    max_f1Score_Classwise = pd.concat(max_f1Score_Classwise,ignore_index=True)

    mdl_max_f1Score = max_f1Score_mdlwise['macro avg f1-score'].idxmax(axis=0)
    mdl_max_Acc = max_Acc_mdlwise['Accuracy'].idxmax(axis=0)
    Class_max_f1Score={}
    for class_ in max_f1Score_Classwise.columns:
        Class_max_f1Score[class_] = np.argmax(max_f1Score_Classwise[class_])



#%%




#%%



