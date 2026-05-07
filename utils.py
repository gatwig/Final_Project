# utils.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import requests


def load_data1(summer_path):
    summer = pd.read_csv(summer_path)
    return summer

def load_data2(winter_path):
    winter = pd.read_csv(winter_path)
    return winter


def standardize_winter(winter_df):
    winter_df["Bait"] = None
    winter_df["Topwater"] = None
    winter_df["Spinning"] = None
    return winter_df


def clean_summer(summer_df):
    # remove foul hook = Yes
    return summer_df[summer_df["Foul_Hook"] != "Yes"].drop(columns=["Foul_Hook"])


def merge_datasets(summer_df, winter_df):
    return pd.concat([summer_df, winter_df], ignore_index=True)


def encode_data(df, label_col="Fish_Species"):
    df = df.copy()
    
    encoders = {}
    for col in ["Lake", "Fish_Species", "Bait", "Topwater", "Spinning"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    return df, encoders


def split_data(df, target="Fish_Species"):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=0)


def run_knn(X_train, X_test, y_train, y_test, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


def run_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

def fetch_wisconsin_weather(start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 43.0731,   # Madison, WI
        "longitude": -89.4012,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "precipitation_sum", "windspeed_10m_max"],
        "timezone": "America/Chicago"
    }

    response = requests.get(url, params=params)
    data = response.json()["daily"]

    weather_df = pd.DataFrame({
        "Date": pd.to_datetime(data["time"]),
        "Temp_Max": data["temperature_2m_max"],
        "Precip": data["precipitation_sum"],
        "Wind_Max": data["windspeed_10m_max"]
    })
    return weather_df


def merge_weather(fish_df, weather_df):
    fish_df["Date"] = pd.to_datetime(fish_df["Date"])
    weather_df["Date"] = pd.to_datetime(weather_df["Date"])
    return pd.merge(fish_df, weather_df, on="Date", how="left")