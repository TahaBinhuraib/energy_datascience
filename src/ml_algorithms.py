import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

df = pd.read_csv("../data/crawled_data.csv")

df["spread"] = df["SMF (TL/MWh)"] - df["PTF (TL/MWh)Okuma Yukumlulu"]
df["net_load"] = (
    df["YAL (0) Kodlu (MWh)"]
    + df["YAL (1) Kodlu (MWh)"]
    + df["YAL (2) Kodlu (MWh)"]
    - df["YAL Teslim Edilmeyen (MWh)"]
    - df["YAT (0) Kodlu (MWh)"]
    - df["YAT (1) Kodlu (MWh)"]
    - df["YAT (2) Kodlu (MWh)"]
    + df["YAT Teslim Edilmeyen (MWh)"]
)
# delete Columns

df = df.drop(
    [
        "Saat",
        "Yuk Tahmin Plani (MWh)",
        "Ikili Anlasma (MWh)",
        "PTF (TL/MWh)Okuma Yukumlulu",
        "SAM (MWh)",
        "SSM(MWh)",
        "KGUP (MWh)",
        "SMF (TL/MWh)",
        "YAL (0) Kodlu (MWh)",
        "YAL (1) Kodlu (MWh)",
        "YAL (2) Kodlu (MWh)",
        "YAL Teslim Edilmeyen (MWh)",
        "YAT (0) Kodlu (MWh)",
        "YAT (1) Kodlu (MWh)",
        "YAT (2) Kodlu (MWh)",
        "YAT Teslim Edilmeyen (MWh)",
        "KARAPINAR_Windy Final (%)",
        "KARAPINAR_Final Production (MWh)",
        "KARAPINAR_KGUP (MWh)",
        "KARAPINAR_Actual KGUP (MWh)",
        "KIVANC_Windy Final (%)",
        "KIVANC_Final Production (MWh)",
        "KIVANC_KGUP (MWh)",
        "KIVANC_Actual KGUP (MWh)",
        "TEKSIN_Windy Final (%)",
        "TEKSIN_Final Production (MWh)",
        "TEKSIN_KGUP (MWh)",
        "TEKSIN_Actual KGUP (MWh)",
        "CINGILLI_Windy Final (%)",
        "CINGILLI_Final Production (MWh)",
        "CINGILLI_KGUP (MWh)",
        "CINGILLI_Actual KGUP (MWh)",
        "BUYUKALAN_Windy Final (%)",
        "BUYUKALAN_Final Production (MWh)",
        "BUYUKALAN_KGUP (MWh)",
        "BUYUKALAN_Actual KGUP (MWh)",
        "Balikesir_Windy Final (m/s)",
        "Balikesir_Final Production (MWh)",
        "Balikesir_KGUP (MWh)",
        "Balikesir_Actual KGUP (MWh)",
        "Gokcedag_Windy Final (m/s)",
        "Gokcedag_Final Production (MWh)",
        "Gokcedag_KGUP (MWh)",
        "Gokcedag_Actual KGUP (MWh)",
        "Dinar_Windy Final (m/s)",
        "Dinar_Final Production (MWh)",
        "Dinar_KGUP (MWh)",
        "Dinar_Actual KGUP (MWh)",
        "Ucpinar_Windy Final (m/s)",
        "Ucpinar_Final Production (MWh)",
        "Ucpinar_KGUP (MWh)",
        "Ucpinar_Actual KGUP (MWh)",
        "Geycek_Windy Final (m/s)",
        "Geycek_Final Production (MWh)",
        "Geycek_KGUP (MWh)",
        "Geycek_Actual KGUP (MWh)",
        "spread",
    ],
    axis=1,
)


df["Date"] = pd.date_range(start="20210427", freq="H", periods=len(df))
df.dropna(inplace=True)

df["10_prev"] = df["net_load"].shift(10)
df["11_prev"] = df["net_load"].shift(11)
df["12_prev"] = df["net_load"].shift(12)

rowsData = []
for i, row in df.iterrows():

    data = dict(
        target=row.net_load,
        weekDay=row.Date.dayofweek,
        weekOfYear=row.Date.week,
        hourOfDay=row.Date.hour,
        dayOfMonth=row.Date.day,
        KARAPINAR_Windy=row["KARAPINAR_Windy Estimated Cloudiness Rate (%)"],
        KIVANC_Windy=row["KIVANC_Windy Estimated Cloudiness Rate (%)"],
        TEKSIN_Windy=row["TEKSIN_Windy Estimated Cloudiness Rate (%)"],
        CINGILLI_Windy=row["CINGILLI_Windy Estimated Cloudiness Rate (%)"],
        BUYUKALAN_Windy=row["BUYUKALAN_Windy Estimated Cloudiness Rate (%)"],
        Balikesir_Windy=row["Balikesir_Windy Estimated Wind Rate (m/s)"],
        Gokcedag_Windy=row["Gokcedag_Windy Estimated Wind Rate (m/s)"],
        Dinar_Windy=row["Dinar_Windy Estimated Wind Rate (m/s)"],
        Ucpinar_Windy=row["Ucpinar_Windy Estimated Wind Rate (m/s)"],
        Geycek_Windy=row["Geycek_Windy Estimated Wind Rate (m/s)"],
        TenPrev=row["10_prev"],
        ElevenPrev=row["11_prev"],
        TwelvePrev=row["12_prev"],
    )

    rowsData.append(data)
df_data = pd.DataFrame(rowsData)

df_data.dropna(inplace=True)

# models
def models(models=dict()):
    # non-linear models
    models["knn"] = KNeighborsRegressor(n_neighbors=7)
    models["DecisionTreeRegressor"] = DecisionTreeRegressor()
    models["extraTreeRegressor"] = ExtraTreeRegressor()
    models["SVR"] = SVR()
    # # ensemble models
    n_trees = 100
    models["adaBoost"] = AdaBoostRegressor(n_estimators=n_trees)
    models["baggingRegressor"] = BaggingRegressor(n_estimators=n_trees)
    models["RandomForest"] = RandomForestRegressor(n_estimators=n_trees)
    models["extraTreesRegressor"] = ExtraTreesRegressor(n_estimators=n_trees)
    models["GradientBoosting"] = GradientBoostingRegressor(n_estimators=n_trees)
    return models


models = models()

# We want to predict the next hour:
df_data["target"] = df_data["target"].shift(-1)
df_data.dropna(inplace=True)

train_size = int(len(df_data) * 0.9)
train, test = df_data.iloc[0:train_size], df_data.iloc[train_size : len(df_data)]
print(len(train), len(test))

# featureColumns:
f_columns = [
    "weekDay",
    "weekOfYear",
    "hourOfDay",
    "dayOfMonth",
    "KARAPINAR_Windy",
    "KIVANC_Windy",
    "TEKSIN_Windy",
    "CINGILLI_Windy",
    "BUYUKALAN_Windy",
    "Balikesir_Windy",
    "Gokcedag_Windy",
    "Dinar_Windy",
    "Ucpinar_Windy",
    "Geycek_Windy",
    "TenPrev",
    "ElevenPrev",
]

# scaling and data preprocessing:
def Scale_preprocess(f_columns, train, test):
    f_columns = f_columns
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    global f_transformer
    f_transformer = scaler_X.fit(train[f_columns].to_numpy())
    global target_transformer
    target_transformer = scaler_y.fit(train[["target"]])

    train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
    train["target"] = target_transformer.transform(train[["target"]])

    test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
    test["target"] = target_transformer.transform(test[["target"]])
    return train, test


train, test = Scale_preprocess(f_columns, train, test)

# data:
X = train.iloc[:, 1:]
y = train.iloc[:, 0]
Xtest = test.iloc[:, 1:]
yTest = test.iloc[:, 0]


def ModelsEval(models, Xtrain, yTrain, Xtest, yTest):
    mae = dict()
    for name, model in models.items():
        regressor = model
        regressor.fit(Xtrain.values.reshape(1, -1), yTrain)
        y_pred = regressor.predict(test.iloc[:, 1:])
        y_pred = target_transformer.inverse_transform(y_pred)
        y_real = target_transformer.inverse_transform(yTest)

        plt.plot(
            np.arange(len(Xtrain), len(Xtrain) + len(Xtest)), y_real, marker=".", label="true"
        )
        plt.plot(np.arange(len(Xtrain), len(Xtrain) + len(Xtest)), y_pred, "r", label="prediction")
        plt.title(f"Model name: {name}")
        plt.show()
        maeModel = mean_absolute_error(y_real, y_pred)
        mae[f"{name}"] = maeModel
    return mae


ModelsEval(models, X, y, Xtest, yTest)
