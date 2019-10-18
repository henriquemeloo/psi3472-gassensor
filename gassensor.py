#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.optimizers import Adam


## Gerando base de dados
# Lendo tabelas dos sensores
sensor_ch4 = pd.read_csv(
    './sensor_ch4.csv', sep=',', decimal=','
    ).rename(columns={'medida':'medida_ch4'})
sensor_co = pd.read_csv(
    './sensor_co.csv', sep=',', decimal=','
    ).rename(columns={'medida':'medida_co'})
sensor_h2 = pd.read_csv(
    './sensor_h2.csv', sep=',', decimal=','
    ).rename(columns={'medida':'medida_h2'})
# Agrupando tabelas
df = sensor_ch4.merge(
    sensor_h2, on=['ch4', 'co', 'h2']
    ).merge(
        sensor_co, on=['ch4', 'co', 'h2']
        ).drop(
            columns=['h2', 'co']
            )
# Gerando conjuntos de treino e teste
msk = np.random.rand(len(df)) < 0.85
train_data = df[msk]
test_data = df[~msk]
print('{len_train} elements in train and {len_test} elements in test'.format(
    len_train=len(train_data), len_test=len(test_data))
)
# Variaveis de entrada e de saida
inputs = ['medida_ch4', 'medida_h2', 'medida_co']
output = 'ch4'

## Treinando modelo
# define the keras model
model = Sequential()
model.add(Dense(3, input_dim=3, activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(15, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='linear'))
# compile the keras model
adam = Adam(lr=1e-1)
model.compile(loss='mean_squared_error', optimizer='adam')
# fit the keras model on the dataset
history = model.fit(
    x=train_data[inputs],
    y=train_data[output],
    epochs=15,
    batch_size=91,
    validation_split=.18,
    verbose=False
)

## Avaliando resultado
# evaluate model
train_rms = np.sqrt(history.history['loss'][-1])
val_rms = np.sqrt(history.history['val_loss'][-1])
test_rms = np.sqrt(
    model.evaluate(x=test_data[inputs], y=test_data[output], verbose=False)
)
print("""
RMS de treino: {train_rms}
RMS de validacao: {val_rms}
RMS de teste: {test_rms}
    """.format(
    train_rms=train_rms,
    val_rms=val_rms,
    test_rms=test_rms
))
# Plotando resultado
plt.rcParams.update({'font.size':14})
plt.figure(figsize=(9,9))
plt.plot(np.sqrt(history.history['loss']))
plt.plot(np.sqrt(history.history['val_loss']))
plt.legend([u'RMS de treino', u'RMS de validação'])
plt.ylabel('RMS')
plt.xlabel(u'Época de treinamento')
plt.title(u'Progressão do treinamento da rede neural do multissensor de gases')
plt.savefig('./images/parteA_rms_treino_validacao.png')
