import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from lstm_pytorch import LSTM
from utils import helpers

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
TIME_STAMPS = 44

df = pd.read_csv("../data/demand_data.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = df.demand.values.astype("float64")
trainSplit = int(len(data) * 0.80)
train_data = data[:-trainSplit]
test_data = data[-trainSplit:]
print(len(train_data))
print(len(test_data))
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
train_data_normalized = train_data_normalized.to(device)


train_inout_seq = helpers.create_inout_sequences(train_data_normalized, TIME_STAMPS)
train_inout_seq[:3]

model = LSTM()
model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)

epochs = 2

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (
            torch.zeros(1, 1, model.hidden_layer).to(device),
            torch.zeros(1, 1, model.hidden_layer).to(device),
        )

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")

print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")
