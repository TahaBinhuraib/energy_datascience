import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from models import *

df = pd.read_csv("../data/demandData.csv")
df = df.drop("Unnamed: 0", axis=1)
df = df.drop("tarih", axis=1)
df = df.drop("saat", axis=1)
data = df
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


trainSplit = int(len(data) * 0.80)

# train test split:
train_data = data[:trainSplit]
test_data = data[trainSplit + 1 :]
print("train length: ", len(train_data))
print("test length: ", len(test_data))


scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_data)

train_data_normalized = pd.DataFrame(
    scaler.transform(train_data), index=train_data.index, columns=train_data.columns
)
test_data_normalized = pd.DataFrame(
    scaler.transform(test_data), index=test_data.index, columns=test_data.columns
)
# sequence initialization:


def create_sequence(input_data: pd.DataFrame, time_stamps):

    sequences = []
    data_size = len(input_data)

    for i in range(data_size - time_stamps):

        sequence = input_data[i : i + time_stamps]
        label_position = i + time_stamps
        label = input_data.iloc[label_position]
        sequences.append((sequence, label))
    return sequences


time_stamps = 44

train_seq = create_sequence(train_data_normalized, time_stamps)
test_seq = create_sequence(test_data_normalized, time_stamps)


class EnergyDataset(Dataset):
    def __init__(self, sequence):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):

        sequence, label = self.sequence[idx]

        return dict(sequence=torch.Tensor(sequence.to_numpy()), label=torch.tensor(label).float())


##trainDataLoader and dataset
train_dataSet = EnergyDataset(train_seq)
train_dataloader = DataLoader(
    dataset=train_dataSet, batch_size=8, shuffle=False, num_workers=2, pin_memory=True
)

##testDataloader and dataset
test_dataSet = EnergyDataset(test_seq)
test_dataloader = DataLoader(dataset=test_dataSet, batch_size=1, shuffle=1)

# check dimensions for concept proof
for batch in train_dataloader:
    seq, lab = batch["sequence"], batch["label"]
    print(seq.shape)
    break


model = LSTM(1)
model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)


import time

epochs = 1

for epoch in range(epochs):
    start = time.time()
    for batch in train_dataloader:
        optimizer.zero_grad()
        seq, labels = batch["sequence"], batch["label"]
        seq = seq.to(device)
        labels = labels.to(device)
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    end = time.time()
    print(f"epoch: {epoch+1} runtime: {end-start}")
    print(f"epoch: {epoch+1} loss: {single_loss.item():10.20f}")

y_pred = []
y_actual = []
with torch.no_grad():
    for data in test_dataloader:
        features, label = data["sequence"], data["label"]
        features = features.to(device)
        y_actual.append(label.cpu().detach().numpy())
        y_predSingle = model(features)
        y_pred.append(y_predSingle.cpu().detach().numpy())


y_pred = np.array(y_pred)
y_actual = np.array(y_actual)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_actual = scaler.inverse_transform(y_actual.reshape(-1, 1))
print("mean absolute Error")


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print(f"Mean absolute Error: {mean_absolute_percentage_error(y_actual,y_pred)}")


plt.plot(y_actual[:100], marker=".", label="true")
plt.plot(y_pred[:100], "r", label="prediction")
plt.ylabel("Target_load")
plt.xlabel("Time")
plt.legend()
plt.show()
