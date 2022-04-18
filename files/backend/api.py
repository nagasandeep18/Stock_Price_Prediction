import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def minmaxscaler(X, min, max):
    omax, omin = X.max(axis=0), X.min(axis=0)
    
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min
    
    return X_scaled, omax, omin

def inverse_scalar(X, omax, omin, min, max):
    X = X - min
    X = X / (max - min)
    
    p1 = X + omin
    p2 = omax - omin 
    X = X * (omax - omin)
    X += omin

    return X

def getColumnsData(df, cols):
    print("Retriving", ' '.join(cols), "Columnn(s)")
    return df[cols]

def getRequiredColumns(df):
    res = []
    dateColName = None
    closeColName = None

    for col in df.columns:
        if (('date' in col.lower()) or ('time' in col.lower())):
            dateColName = col
            break

    for col in df.columns:
        if ('open' in col.lower()):
            res.append(col)
            break
    
    for col in df.columns:
        if ('low' in col.lower()):
            res.append(col)
            break

    for col in df.columns:
        if ('high' in col.lower()):
            res.append(col)
            break
    
    for col in df.columns:
        if (('close' in col.lower()) and ('adj' not in col.lower()) and ('prev' not in col.lower())):
            res.append(col)
            closeColName = col
            break
    
    for col in df.columns:
        if (('volume' in col.lower()) or ('turnover' in col.lower())):
            res.append(col)
            break

    return res, dateColName, closeColName
    

def LMS(df, pred_col, next_days, epochs, updateEpochs):
    print("LMS Training for", pred_col)
    
    ndf, omax, omin = minmaxscaler(df[pred_col], 1000, 2000)
    x = ndf.values
    
    tmp = []
    for i in x: tmp.append(i)

    x = np.array(tmp)

    def lmsPred(x,l,u,N):
        xd = np.block([1, x]).T
        y=np.zeros((len(xd),1))

        xn = np.zeros((N+1,1))
        xn = np.matrix(xn)

        wn=np.random.rand(N+1,1)/10 
        
        M=len(xd)
        for epoch in range(epochs):
            updateEpochs(epoch)
            print("epoch ", epoch+1, "/", epochs, sep='')

            for n in range(0,M):
                xn = np.block([[xd[n]], [xn[0:N]]])
                y[n]= np.matmul(wn.T, xn)

                if(n>M-l-1): e = 0;
                else: e=int(x[n]-y[n])

                wn = wn + 2*u*e*xn
                
        return y,wn;

    x_train = x[:-next_days]
    u = 2**(-30);

    l=next_days;
    N=100;

    y,wn = lmsPred(x_train,l,u,N)
    
    x = inverse_scalar(ndf, omax, omin, 1000, 2000)
    y = inverse_scalar(y, omax, omin, 1000, 2000)

    # plotGraph(cols=[x, y], title=pred_col, colors=['black', 'red'])

    json = {
        "inputs": x,
        "outputs": y,
        "actual": x[-l:].values,
        "predicted": y[-l:]
    }

    return json


from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from keras.layers import Dropout

def LSTM_Cell(inputs, init_h, init_c, kernel, recurrent_kernel, bias,
                  mask, time_major, go_backwards, sequence_lengths,
                  zero_output_for_mask):
  
  input_length = inputs.shape[1]

  def operations(cell_inputs, cell_states):
    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_states[0]  # previous memory state
    c_tm1 = cell_states[1]  # previous carry state

    z = K.dot(cell_inputs, kernel)
    z += K.dot(h_tm1, recurrent_kernel)
    z = K.bias_add(z, bias)

    z0, z1, z2, z3 = array_ops.split(z, 4, axis=1)

    i = nn.sigmoid(z0)
    f = nn.sigmoid(z1)
    c = f * c_tm1 + i * nn.tanh(z2)
    o = nn.sigmoid(z3)

    h = o * nn.tanh(c)
    return h, [h, c]

  last_output, outputs, new_states = K.rnn(
      operations,
      inputs, [init_h, init_c],
      constants=None,
      unroll=False,
      time_major=time_major,
      mask=mask,
      go_backwards=go_backwards,
      input_length=input_length,
      zero_output_for_mask=zero_output_for_mask
  )
  
  return (last_output, outputs, new_states[0], new_states[1])

class LSTM(recurrent.DropoutRNNCellMixin, recurrent.LSTM):
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               bias_initializer='zeros',
               unit_forget_bias=True,
               return_sequences=False,
               **kwargs):

    super(LSTM, self).__init__(
        units,
        return_sequences=return_sequences,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        **kwargs)

  def call(self, inputs, mask=None, training=None, initial_state=None):
        row_lengths = inputs.shape[0]
        inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)

        lstm_kwargs = {
            'inputs': inputs,
            'init_h': initial_state[0],
            'init_c': initial_state[1],
            'kernel': self.cell.kernel.read_value(),
            'recurrent_kernel': self.cell.recurrent_kernel.read_value(),
            'bias': self.cell.bias.read_value(),
            'mask': mask,
            'time_major': self.time_major,
            'go_backwards': self.go_backwards,
            'sequence_lengths': row_lengths,
            'zero_output_for_mask': self.zero_output_for_mask
        }
        
        (last_output, outputs, new_h, new_c) = LSTM_Cell(**lstm_kwargs)
            
        output = last_output

        return output


import keras

class EpochPrintingCallback (keras.callbacks.Callback):
    def __init__(self, updateEpochs):
        self.updateEpochs = updateEpochs

    def on_epoch_end(self, epoch, logs=None):
        print(epoch)
        self.updateEpochs(epoch)

import warnings
warnings.filterwarnings('ignore')

import math
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import *
from keras.layers import Dense
from keras.models import Sequential

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def LSTMAlgorithm(fileName, train_size, epochs, updateEpochs):
    df = pd.read_csv('./datasets/' + fileName + '.csv')
    cols, dateColName, trade_close_col = getRequiredColumns(df)


    scaling_data_frame = df.filter(cols)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_Data = scaler.fit_transform(scaling_data_frame)
    scaled_data_frame = pd.DataFrame(data=scaled_Data, index=[df[trade_close_col]], columns=cols)

    stock_close_data = df.filter([trade_close_col])
    stock_close_dataset = stock_close_data.values

    # trainingDataLength = math.ceil( len(stock_close_dataset) * train_size )
    trainingDataLength = len(stock_close_dataset)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaledData = scaler.fit_transform(stock_close_dataset)

    StockTrainData = scaledData[0:trainingDataLength , :]

    Xtrain = []
    Ytrain = []


    for i in range(60, len(StockTrainData)):
        Xtrain.append(StockTrainData[i-60:i, 0])
        Ytrain.append(StockTrainData[i, 0])

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
    # testingData = scaledData[trainingDataLength - 60: , :]

    # Xtest = []
    # Ytest = stock_close_dataset[trainingDataLength:, :]
    # for i in range(60, len(testingData)):
    #   Xtest.append(testingData[i-60:i, 0])

    # Xtest = np.array(Xtest)
    # Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))


    print("\n\nLSTM Algorithm for "+str(epochs)+" epochs")

    neurons = 50
    
    model = Sequential()

    model.add(LSTM(neurons, return_sequences=True, input_shape= (Xtrain.shape[1], 1)))
    model.add(LSTM(neurons, return_sequences=False)) 

    model.add(Dense(25)) 
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse') 

    history_data = model.fit(Xtrain, Ytrain, 
                            batch_size=50, epochs=epochs, validation_split=0.2, 
                            verbose=0, callbacks=[EpochPrintingCallback(updateEpochs=updateEpochs)])
    print("Saving Model--------------------------------------------->")
    
    model.save('pretrained/' + fileName + ".h5")

    # predictions = model.predict(Xtest)
    # predictions = scaler.inverse_transform(predictions)

    # training = stock_close_data[:trainingDataLength]
    # validation = pd.DataFrame(df[trade_close_col][trainingDataLength:], columns=['Close'])

    # validation['Predictions'] = predictions


    # real = validation['Close'].values
    # pred = validation['Predictions'].values
    # n = len(pred)

    # accuracy = 0
    # for i in range(n):
    #     accuracy += (abs(real[i] - pred[i])/real[i])*100

    # print('For', epochs, "epochs")
    # print("Accuracy:", 100 - accuracy/n, end='\n\n')

    return model


def getPredictonsFromModel(fileName, train_size):
    df = pd.read_csv('./datasets/' + fileName + '.csv')
    cols, dateColName, trade_close_col = getRequiredColumns(df)

    model = tf.keras.models.load_model('./pretrained/' + fileName + '.h5')

    scaling_data_frame = df.filter(cols)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_Data = scaler.fit_transform(scaling_data_frame)
    scaled_data_frame = pd.DataFrame(data=scaled_Data, index=[df[trade_close_col]], columns=cols)

    stock_close_data = df.filter([trade_close_col])
    stock_close_dataset = stock_close_data.values

    trainingDataLength = math.ceil( len(stock_close_dataset) * train_size )

    scaler = MinMaxScaler(feature_range=(0,1))
    scaledData = scaler.fit_transform(stock_close_dataset)

    StockTrainData = scaledData[0:trainingDataLength , :]

    Xtrain = []
    Ytrain = []

    for i in range(60, len(StockTrainData)):
        Xtrain.append(StockTrainData[i-60:i, 0])
        Ytrain.append(StockTrainData[i, 0])

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
    testingData = scaledData[trainingDataLength - 60: , :]

    Xtest = []
    Ytest = stock_close_dataset[trainingDataLength:, :]
    for i in range(60, len(testingData)):
        Xtest.append(testingData[i-60:i, 0])

    Xtest = np.array(Xtest)
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))


    # predictions

    predictions = model.predict(Xtest)
    predictions = scaler.inverse_transform(predictions)

    training = stock_close_data[:trainingDataLength]
    validation = pd.DataFrame(df[trade_close_col][trainingDataLength:], columns=['Close'])

    validation['Predictions'] = predictions


    real = validation['Close'].values
    pred = validation['Predictions'].values
    n = len(pred)

    accuracy = 0
    for i in range(n):
        accuracy += (abs(real[i] - pred[i])/real[i])*100

    # print('For', epochs, "epochs")
    accuracyPercentage = 100 - accuracy/n
    # print("Accuracy:", , end='\n\n')

    trainingDates = df[dateColName].iloc[:trainingDataLength]
    trainingDates = list(trainingDates.values)
    trainingData = list(training[trade_close_col].values)
    
    realData = list(real)
    
    predictionDates = df[dateColName].iloc[trainingDataLength:]
    predictionDates = list(predictionDates.values)
    predictionData = list(pred)

    for i in range(len(trainingData)): trainingData[i] = float(trainingData[i])
    for i in range(len(predictionData)): predictionData[i] = float(predictionData[i])

    json = {
        "training": {
            "dates": trainingDates,
            "data": trainingData
        },
        "predictions": {
            "dates": predictionDates,
            "realData": realData,
            "predictedData": predictionData,
            "accuracy": accuracyPercentage
        }
    }

    return json


def getManualPredictionForModel(fileName, train_size, openValue, highValue, lowValue, volumeValue):
    df = pd.read_csv('./datasets/' + fileName + '.csv')
    cols, dateColName, trade_close_col = getRequiredColumns(df)

    close_idx = -1
    for col in df.columns:
        close_idx += 1
        if(col == trade_close_col): break
    
    row = []
    for i in range(df.shape[1]):
        if(i==close_idx): row.append(random.randint(int(float(lowValue)), int(float(highValue))))
        else: row.append(0)
    df.loc[df.shape[0]] = row


    model = tf.keras.models.load_model('./pretrained/' + fileName + '.h5')

    scaling_data_frame = df.filter(cols)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_Data = scaler.fit_transform(scaling_data_frame)
    scaled_data_frame = pd.DataFrame(data=scaled_Data, index=[df[trade_close_col]], columns=cols)

    stock_close_data = df.filter([trade_close_col])
    stock_close_dataset = stock_close_data.values

    trainingDataLength = math.ceil( len(stock_close_dataset) * train_size )-1

    scaler = MinMaxScaler(feature_range=(0,1))
    scaledData = scaler.fit_transform(stock_close_dataset)

    testingData = scaledData[trainingDataLength - 60: , :]

    Xtest = []
    for i in range(60, len(testingData)+1):
        Xtest.append(testingData[i-60:i, 0])

    Xtest = np.array(Xtest)
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))
    # predictions

    predictions = model.predict(Xtest)
    predictions = scaler.inverse_transform(predictions)


    return predictions[-1][0]