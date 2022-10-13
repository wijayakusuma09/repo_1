#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Import Data
df = pd.read_csv('TLKM.JK.csv')
df 


# In[3]:


df.info()


# In[4]:


null_columns=df.columns[df.isnull().any()]
print(df[df["Open"].isnull()][null_columns])


# In[5]:


df = df.drop([636])
df.reset_index(drop=True,inplace=True)
df.head()


# In[7]:


# merubah tipe data object to datetime
df['Date'] = df['Date'].astype('datetime64')

# melihat tipe data dataframe
print(df.dtypes)


# In[8]:


print('waktu terawal dari kolom Date adalah:', df['Date'].min())
df.head()


# In[9]:


# mengurutkan data berdasarkan waktu
df.sort_values('Date', inplace=True, ignore_index=True)
df.head()


# In[34]:


visual_plot =df[['Date','Close', 'Open', 'High']]

plt.figure(figsize=(20,10))

sns.lineplot(y=visual_plot['Open'], color="r", x=visual_plot['Date'])
sns.lineplot(y=visual_plot['Close'], color="g", x=visual_plot['Date'])
sns.lineplot(y=visual_plot['High'], color="cyan", x=visual_plot['Date'])

plt.xlabel('Tahun', fontsize=20)
plt.ylabel('Harga (IDR)', fontsize=20)
plt.legend(['Open','Close','High'], loc='upper right')


# In[12]:


# split data
train_size = int(len(df) * 0.7) # Menentukan banyaknya data train yaitu sebesar 70% data
train = df[:train_size]
test =df[train_size:].reset_index(drop=True)


# In[39]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train[['Close']])

train['scaled'] = scaler.transform(train[['Open']])
test['scaled'] = scaler.transform(test[['Open']])
train['scaled'] = scaler.transform(train[['Close']])
test['scaled'] = scaler.transform(test[['Close']])


# In[40]:


train.head()


# In[41]:


def sliding_window(data, window_size):
    sub_seq, next_values = [], []
    for i in range(len(data)-window_size):
        sub_seq.append(data[i:i+window_size])
        next_values.append(data[i+window_size])
    X = np.stack(sub_seq)
    y = np.array(next_values)
    return X,y


# In[42]:


window_size = 12

X_train, y_train = sliding_window(train[['scaled']].values, window_size)
X_test, y_test = sliding_window(test[['scaled']].values, window_size)


# In[43]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[44]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM #, RNN, GRU 


# In[45]:


def create_model(LSTM_unit=64, dropout=0.2): #jika ingin menggunakan RNN atau GRU ganti LSTM dengan GRU/RNN
    # create model
    model = Sequential()
    model.add(LSTM(units=LSTM_unit, input_shape=(window_size, 1)))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


# In[46]:


LSTM_unit = [16,32,64,128]
dropout = [0.1,0.2]


# In[47]:


from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping

# create model
model = KerasRegressor(build_fn=create_model, epochs=25, validation_split=0.1, batch_size=32)

# define the grid search parameters
LSTM_unit = [16,32,64,128]
dropout=[0.1,0.2]
param_grid = dict(LSTM_unit=LSTM_unit, dropout=dropout)


# In[48]:


grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)


# In[49]:


grid_result = grid.fit(X_train, y_train)


# In[50]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
# Mengambil model terbaik
best_model = grid_result.best_estimator_.model


# In[51]:


history = best_model.history
# grafik loss function MSE

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('loss function MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()


# In[52]:


# grafik metric MAE

plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('metric MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()


# In[53]:


# Prediksi data train
predict_train = scaler.inverse_transform(best_model.predict(X_train))
true_train = scaler.inverse_transform(y_train)

# Prediksi data test
predict_test = scaler.inverse_transform(best_model.predict(X_test))
true_test = scaler.inverse_transform(y_test)


# In[55]:


train['predict'] = np.nan
train['predict'][-len(predict_train):] = predict_train[:,0]

plt.figure(figsize=(15,8))
sns.lineplot(data=train, x='Date', y='Close', label = 'train')
sns.lineplot(data=train, x='Date', y='predict', label = 'predict')


# In[30]:


test['predict'] = np.nan
test['predict'][-len(predict_test):] = predict_test[:,0]

plt.figure(figsize=(15,8))
sns.lineplot(data=test, x='Date', y='Close', label = 'test')
sns.lineplot(data=test, x='Date', y='predict', label = 'predict')


# In[31]:


plt.figure(figsize=(15,8))
sns.lineplot(data=test[-24*30:], x='Date', y='Close', label = 'test')
sns.lineplot(data=test[-24*30:], x='Date', y='predict', label = 'predict')


# In[32]:


# forecasting data selanjutnya
y_test = scaler.transform(test[['Close']])
n_future = 24*7
future = [[y_test[-1,0]]]
X_new = y_test[-window_size:,0].tolist()

for i in range(n_future):
    y_future = best_model.predict(np.array([X_new]).reshape(1,window_size,1))
    future.append([y_future[0,0]])
    X_new = X_new[1:]
    X_new.append(y_future[0,0])

future = scaler.inverse_transform(np.array(future))
date_future = pd.date_range(start=test['Date'].values[-1], periods=n_future+1, freq='H')
# Plot Data sebulan terakhir dan seminggu ke depan
plt.figure(figsize=(15,8))
sns.lineplot(data=test[-24*30:], x='Date', y='Close', label = 'test')
sns.lineplot(data=test[-24*30:], x='Date', y='predict', label = 'predict')
sns.lineplot(x=date_future, y=future[:,0], label = 'future')
plt.ylabel('Close');


# In[ ]:




