import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from sklearn.metrics import mean_squared_error

color_pal = sns.color_palette()
plt.style.use("fivethirtyeight")

df = pd.read_csv("../../data/raw/PJME_hourly.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime")

df.plot(style=".", figsize=(15, 5), color=color_pal, title="PJME usage in MW")

# use histogram to observe distribution range to determine outliers
df["PJME_MW"].plot(kind="hist", bins=100)

df.query("PJME_MW < 20_000").plot(figsize=(15, 5), style=".")

# remove outliers ( value less than 19000 in PJME_MW)
df = df.query("PJME_MW > 19_000").copy()


# time series cross validation
# ( split into how many datasets for cross validation depends on how large the initial data is)
# training size set as 1 year ( based on hourly data), gap set as 24 hours between validation, test set)
tss = TimeSeriesSplit(n_splits=5, test_size=24 * 365 * 1, gap=24)
df = (
    df.sort_index()
)  # time series data split has to be chronological ( no random split)

# visualize how time series data split works
fig, axs = plt.subplots(5, 1, figsize=(15, 10), sharex=True)  # Corrected function call
fold = 0
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]
    train["PJME_MW"].plot(
        ax=axs[fold],  # Correct variable reference
        label="training set",
        title=f"data train test split fold {fold}",
    )
    test["PJME_MW"].plot(ax=axs[fold], label="test set")  # Correct variable reference
    axs[fold].axvline(  # Correct variable reference
        test.index.min(), color="black", ls="--"
    )
    fold += 1

plt.legend()
plt.show()


# forecasting horizon , short-term FH (less than 3 months), long-term FH (more than 2 year)


# feature creation (hour of the day, day of the week, month of the year and so on)
def create_features(df):
    # create time series features based on time series index
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.day_of_week
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    df["dayofyear"] = df.index.day_of_year
    return df


df = create_features(df)


# lag feature creation (what was the largest (x) days in the past)
def add_lags(df):
    target_map = df["PJME_MW"].to_dict()
    df["lag1"] = (df.index - pd.Timedelta("364 days")).map(target_map)
    df["lag2"] = (df.index - pd.Timedelta("728 days")).map(target_map)
    df["lag3"] = (df.index - pd.Timedelta("1092 days")).map(target_map)
    return df


df = add_lags(df)
df.columns

# train using cross validation

tss = TimeSeriesSplit(n_splits=5, test_size=24 * 365 * 1, gap=24)
df = df.sort_index()

fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]

    features = [
        "hour",
        "dayofweek",
        "month",
        "quarter",
        "year",
        "dayofyear",
        "lag1",
        "lag2",
        "lag3",
    ]
    target = ["PJME_MW"]

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    # turn the hyperparameter to get better score
    reg = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        base_score=0.5,
        booster="gbtree",
        max_depth=3,
        objective="reg:linear",
    )
    reg.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100,
        early_stopping_rounds=50,
    )

    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)

scores


# predict the future
# (1, re-train on all data )
# (2,also need an empty dataframe for furture data ranges )
# (3, run those dates through create_feature function + lag creation)


features = [
    "hour",
    "dayofweek",
    "month",
    "quarter",
    "year",
    "dayofyear",
    "lag1",
    "lag2",
    "lag3",
]
target = ["PJME_MW"]

X_all = df[features]
y_all = df[target]

reg = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    base_score=0.5,
    booster="gbtree",
    max_depth=3,
    objective="reg:linear",
)
reg.fit(
    X_all,
    y_all,
    eval_set=[(X_all, y_all)],
    verbose=100,
    early_stopping_rounds=50,
)

# create future dataframe
df.index.max()

future = pd.date_range("2018-08-03", "2019-08-01", freq="1h")
future_df = pd.DataFrame(index=future)

future_df["isFuture"] = True
df["isFuture"] = False
df_and_future = pd.concat([df, future_df])
df_and_future = create_features(df_and_future)
df_and_future = add_lags(df_and_future)

future_w_features = df_and_future.query("isFuture").copy()

future_w_features["pred"] = reg.predict(future_w_features[features])
future_w_features["pred"].plot(
    figsize=(10, 5), color=color_pal[4], ms=1, lw=1, title="future predictions"
)
plt.show()


# save model
reg.save_model("model.json")

# re-load model
reg_new = xgb.XGBRegressor()
reg_new.load_model("model.json")
