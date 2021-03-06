#!/usr/bin/env python
# coding: utf-8

# In[1]:


# readme

# import datetime
# import os
# import tensorflow as tf

# sess_DateTime = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
# ConfMat_Path = os.path.join(Model_Path,"logs",(model.name+'_'+sess_DateTime))
# log_confusion_matrix=callback_ConfMat(model=model, X=X_Val, Y=Y_Val, class_names=class_names, logdir=ConfMat_Path)
# Define the per-epoch callback.
# ConfMat_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

# add ConfMat_cb in the list of callbacks while training


# In[5]:


import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics
import io
import numpy as np
import itertools


# In[ ]:


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


# In[ ]:


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(0.75*len(class_names), 0.75*len(class_names)))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


# In[ ]:


def callback_ConfMat(model, dataset, class_names, logdir, freq=1):
    file_writer_cm = tf.summary.create_file_writer(logdir+'/ConfMat')
    if type(freq)==type(None) or freq==[]:
        freq=1

    def datagen(dataset):
        global Y_callback_ConfMat
        Y_callback_ConfMat=[]
        for batch in dataset:
            X,lbl = batch
            Y_callback_ConfMat.extend(lbl)
            yield X

    def log_confusion_matrix(epoch, logs):
        if epoch==0 or (epoch+1)%freq == 0:
            # Use the model to predict the values from the validation dataset.
            pred_raw = model.predict(datagen(dataset),steps=tf.data.experimental.cardinality(dataset))
            pred = np.argmax(pred_raw, axis=1)

            # Calculate the confusion matrix.
            cm = sklearn.metrics.confusion_matrix(np.array(Y_callback_ConfMat), pred)
            # Log the confusion matrix as an image summary.
            figure = plot_confusion_matrix(cm, class_names=class_names)
            cm_image = plot_to_image(figure)

            # Log the confusion matrix as an image summary.
            with file_writer_cm.as_default():
                tf.summary.image("Confusion Matrix", cm_image, step=epoch)
    
    return log_confusion_matrix


# In[ ]:





# In[ ]:




