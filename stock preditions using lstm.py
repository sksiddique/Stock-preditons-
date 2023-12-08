#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# In[19]:


# Load your stock data (replace 'your_stock_data.csv' with your file)
data = pd.read_csv('stock_data.csv')
data = data['Close'].values.reshape(-1, 1)


# In[20]:


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)


# In[21]:


# Create sequences and labels for training
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


# In[22]:


# Choose sequence length (number of time steps to look back)
sequence_length = 10
X, y = create_sequences(data_scaled, sequence_length)


# In[23]:


# Split the data into training and testing sets
train_size = int(len(data) * 0.80)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# In[24]:


# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')


# In[25]:


# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)


# In[27]:


# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Training Loss: {train_loss:.4f}')
print(f'Testing Loss: {test_loss:.4f}')


# In[28]:


# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


# In[29]:


# Invert predictions to original scale
train_predictions = scaler.inverse_transform(train_predictions)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[30]:


# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(np.arange(len(data)), data, label='Actual Stock Price', color='blue')
plt.plot(np.arange(sequence_length, train_size + sequence_length), train_predictions.flatten(), label='Training Predictions', color='orange')
plt.plot(np.arange(train_size + sequence_length, len(data)), test_predictions.flatten(), label='Testing Predictions', color='green')
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[ ]:




