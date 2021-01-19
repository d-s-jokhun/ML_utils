

#%%

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools

#%%

def plot_ConfMat(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    if cm.shape[0]>50:
        print ('Class names will not be printed if more than 50 classes are present!')
        figure = plt.figure(figsize=(10,10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues,\
            norm=matplotlib.colors.Normalize(vmin=None, vmax=None, clip=False))
        plt.title("Confusion matrix")
        plt.colorbar()
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure
    else: 
        figure = plt.figure(figsize=(0.75*cm.shape[1], 0.75*cm.shape[0]))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues,\
            norm=matplotlib.colors.Normalize(vmin=None, vmax=None, clip=False))
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, round(cm[i, j],2), horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

