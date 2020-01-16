import tensorflow as tf
import tensorflow.keras.backend as backend
import GPUtil
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import random
import os
from datetime import datetime
import sys
from sklearn.model_selection import train_test_split

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
backend.set_session(tf.Session(config=config))
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUtil.getFirstAvailable()[0])

def item2vecModel(vocab_size, embedding_dim):
    target = keras.layers.Input(shape=(1,), name='target')
    context = keras.layers.Input(shape=(1,), name='context')
    shared_embedding = keras.layers.Embedding(vocab_size, embedding_dim, input_length=1, name='shared_embedding')
    embedding_target = shared_embedding(target)
    embedding_context = shared_embedding(context)
    merged_vector = keras.layers.dot([embedding_target, embedding_context], axes=-1)
    reshaped_vector = keras.layers.Reshape((1,), input_shape=(1,1))(merged_vector)
    prediction = keras.layers.Dense(1, input_shape=(1,), activation='sigmoid')(reshaped_vector)
    model = keras.models.Model(inputs=[target, context], outputs=prediction)
    return model


if __name__ == '__main__':
    print('Starting...')

    data = pd.read_csv('train.csv')
    # target = data['target']
    # context = data['context']
    # label = data['label']
    data_train, data_valid = train_test_split(data, test_size=0.1)
    target_train = data_train['target']
    context_train = data_train['context']
    label_train = data_train['label']
    target_valid = data_valid['target']
    context_valid = data_valid['context']
    label_valid = data_valid['label']

    anchor_list = pd.read_csv('anchor_list.csv')
    frequent_anchor = anchor_list[anchor_list['count'] >= 10]

    # target = target.values.T.reshape(len(target),1)
    # context = context.values.T.reshape(len(target),1)
    # label = label.values.T.reshape(len(target),1)

    V = len(frequent_anchor) + 1
    D = 30
    model = item2vecModel(V, D)
    # optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    # model.compile(optimizer=keras.optimizers.SGD(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint_cb = keras.callbacks.ModelCheckpoint("model_ckp.model")
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10)

    try:
        model.fit((target_train, context_train), label_train, 
            validation_data=((target_valid, context_valid), label_valid),epochs=15, callbacks=[checkpoint_cb, early_stopping_cb])
    except KeyboardInterrupt:
        model.save('model_40_d30_interrupted.model')
