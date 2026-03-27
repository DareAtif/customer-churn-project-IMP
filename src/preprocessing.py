# Load Data
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # convert TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
    df.dropna(inplace=True)

    # Encode Target
    df['Churn'] = df['Churn'].map({'Yes':1,"No":0})

    # drop ID
    df.drop('customerID',axis=1,inplace=True)

    return df

def encode_data(df):
    df = pd.get_dummies(df,drop_first=True)
    return df

def split_features_target(df):
    X = df.drop("Churn",axis=1)
    y = df['Churn']

    return X,y

def scale_data(X_train,X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled,X_test_scaled,scaler

    print("run successfully")

