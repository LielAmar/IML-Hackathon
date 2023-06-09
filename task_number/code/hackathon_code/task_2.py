import joblib
import numpy as np
import pandas as pd
import re

import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import matplotlib.pyplot as plt

# Currencies as of 8.6.23 12:45
from task_number.code.hackathon_code.task_1 import task_1_prediction

currencies = {
    "AED": 3.673,
    "AFN": 86.517226,
    "ALL": 99.967307,
    "AMD": 385.305018,
    "ANG": 1.801542,
    "AOA": 612.5,
    "ARS": 243.555608,
    "AUD": 1.495317,
    "AWG": 1.8025,
    "AZN": 1.7,
    "BAM": 1.826417,
    "BBD": 2,
    "BDT": 108.036662,
    "BGN": 1.82618,
    "BHD": 0.376975,
    "BIF": 2824.740732,
    "BMD": 1,
    "BND": 1.347098,
    "BOB": 6.914198,
    "BRL": 4.9242,
    "BSD": 1,
    "BTC": 0.000037812887,
    "BTN": 82.558899,
    "BWP": 13.558388,
    "BYN": 2.525657,
    "BZD": 2.016902,
    "CAD": 1.33352,
    "CDF": 2326.572701,
    "CHF": 0.908657,
    "CLF": 0.028699,
    "CLP": 791.88,
    "CNH": 7.1406,
    "CNY": 7.1292,
    "COP": 4215.212601,
    "CRC": 536.796676,
    "CUC": 1,
    "CUP": 25.75,
    "CVE": 102.965915,
    "CZK": 22.032323,
    "DJF": 178.157609,
    "DKK": 6.94539,
    "DOP": 54.751049,
    "DZD": 136.588851,
    "EGP": 30.949796,
    "ERN": 15,
    "ETB": 54.37007,
    "EUR": 0.932236,
    "FJD": 2.22975,
    "FKP": 0.801419,
    "GBP": 0.801419,
    "GEL": 2.615,
    "GGP": 0.801419,
    "GHS": 11.307011,
    "GIP": 0.801419,
    "GMD": 59.445,
    "GNF": 8604.04959,
    "GTQ": 7.834971,
    "GYD": 211.628268,
    "HKD": 7.836742,
    "HNL": 24.702068,
    "HRK": 7.024398,
    "HTG": 139.578133,
    "HUF": 344.398428,
    "IDR": 14887.174315,
    "ILS": 3.662767,
    "IMP": 0.801419,
    "INR": 82.554664,
    "IQD": 1310.828226,
    "IRR": 42312.5,
    "ISK": 140.13,
    "JEP": 0.801419,
    "JMD": 154.94408,
    "JOD": 0.7101,
    "JPY": 139.73203846,
    "KES": 139.2,
    "KGS": 87.3849,
    "KHR": 4126.05406,
    "KMF": 460.499925,
    "KPW": 900,
    "KRW": 1301.967716,
    "KWD": 0.3075,
    "KYD": 0.833824,
    "KZT": 445.823822,
    "LAK": 18161.966732,
    "LBP": 15018.79313,
    "LKR": 292.188326,
    "LRD": 171.100041,
    "LSL": 19.059906,
    "LYD": 4.827007,
    "MAD": 10.212357,
    "MDL": 17.840967,
    "MGA": 4427.326549,
    "MKD": 57.436224,
    "MMK": 2101.324837,
    "MNT": 3519,
    "MOP": 8.082364,
    "MRU": 34.398833,
    "MUR": 46.150001,
    "MVR": 15.35,
    "MWK": 1022.597647,
    "MXN": 17.344193,
    "MYR": 4.6185,
    "MZN": 63.850001,
    "NAD": 19.21,
    "NGN": 461.68,
    "NIO": 36.595558,
    "NOK": 10.98545,
    "NPR": 131.960419,
    "NZD": 1.645829,
    "OMR": 0.385033,
    "PAB": 1,
    "PEN": 3.678981,
    "PGK": 3.552169,
    "PHP": 56.126494,
    "PKR": 287.049298,
    "PLN": 4.184877,
    "PYG": 7249.42017,
    "QAR": 3.641,
    "RON": 4.6241,
    "RSD": 109.292794,
    "RUB": 82.010007,
    "RWF": 1132.533874,
    "SAR": 3.750543,
    "SBD": 8.334167,
    "SCR": 13.247405,
    "SDG": 601.5,
    "SEK": 10.890057,
    "SGD": 1.346624,
    "SHP": 0.801419,
    "SLL": 17665,
    "SOS": 568.334757,
    "SRD": 37.5885,
    "SSP": 130.26,
    "STD": 22823.990504,
    "STN": 22.878594,
    "SVC": 8.755661,
    "SYP": 2512.53,
    "SZL": 19.050287,
    "THB": 34.834,
    "TJS": 10.931596,
    "TMT": 3.51,
    "TND": 3.1105,
    "TOP": 2.368232,
    "TRY": 23.3494,
    "TTD": 6.787132,
    "TWD": 30.724498,
    "TZS": 2365,
    "UAH": 36.912126,
    "UGX": 3737.25347,
    "USD": 1,
    "UYU": 39.008022,
    "UZS": 11460.130376,
    "VES": 26.670432,
    "VND": 23496.241899,
    "VUV": 118.979,
    "WST": 2.72551,
    "XAF": 611.506725,
    "XAG": 0.04208763,
    "XAU": 0.00051321,
    "XCD": 2.70255,
    "XDR": 0.749213,
    "XOF": 611.506725,
    "XPD": 0.00073207,
    "XPF": 111.245345,
    "XPT": 0.00097744,
    "YER": 250.349961,
    "ZAR": 18.924807,
    "ZMW": 20.01214,
    "ZWL": 322
}
means = {'hotel_star_rating': 2.7379305082159027,
         'time_ahead': 30.059532563322083,
         'staying_duration': 1.783537671058656,
         'no_of_people': 2.1357277305794375}


def clean_data(X: pd.DataFrame, y: pd.Series):
    X = X[X["no_of_people"] < 20]
    X = X[X["time_ahead"] >= -1]
    X["time_ahead"][X["time_ahead"] <= 0] = 0
    y = y.loc[X.index]
    # X["request_highfloor"] = X["request_highfloor"].fillna(0)
    # X["request_largebed"] = X["request_largebed"].fillna(0)
    # X["request_twinbeds"] = X["request_twinbeds"].fillna(0)
    # X["request_airport"] = X["request_airport"].fillna(0)

    X.loc[(X["hotel_star_rating"] < 0) |
          (X["hotel_star_rating"] > 5), "hotel_star_rating"] = means["hotel_star_rating"]

    for feature in ["time_ahead", "staying_duration", "no_of_people"]:
        X.loc[(X[feature].isna()) | (X[feature] < 0), feature] = means[feature]

    return X, y


def split_data(df: pd.DataFrame):
    # Divide into X and y
    df["original_selling_amount"] = df.apply(
        lambda x: (1 / currencies[x["original_payment_currency"]]) * x["original_selling_amount"], axis=1)

    cancellation_datetime = df["cancellation_datetime"]
    X, y = df.drop(["original_selling_amount"], axis=1), df["original_selling_amount"]
    y = pd.Series(np.where(cancellation_datetime.isna(), -1, y))

    # Divide into train, dev and test
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.1, random_state=0)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def remove_redundant_features(X):
    X = X.drop(['h_booking_id', 'hotel_id', 'hotel_area_code', 'hotel_chain_code', 'hotel_live_date', 'h_customer_id',
                'customer_nationality', 'guest_is_not_the_customer', 'language', 'guest_nationality_country_name',
                'origin_country_code', 'original_payment_type', 'is_user_logged_in', 'is_first_booking',
                'request_nonesmoke', 'request_latecheckin', 'request_earlycheckin', 'charge_option',
                'cancellation_policy_code', 'original_payment_currency', 'original_payment_method',
                'hotel_country_code', 'checkin_date', 'checkout_date', 'booking_datetime',
                "no_of_adults", "no_of_children",
                "request_highfloor", "request_largebed", "request_twinbeds", "request_airport"], axis=1)

    return X


def create_dummy_features(X):
    X['checkin_month'] = X['checkin_date'].dt.month

    X = pd.get_dummies(X, prefix='checkin_month', columns=['checkin_month'])
    X = pd.get_dummies(X, prefix='hotel_brand_code', columns=['hotel_brand_code'])
    X = pd.get_dummies(X, prefix='accommadation_type_name', columns=['accommadation_type_name'])
    X = pd.get_dummies(X, prefix='hotel_city_code', columns=['hotel_city_code'])

    return X


def create_linear_features(X):
    X['time_ahead'] = np.round((X['checkin_date'] - X['booking_datetime']) / np.timedelta64(1, 'D'))
    X['staying_duration'] = np.round((X['checkout_date'] - X['checkin_date']) / np.timedelta64(1, 'D'))
    X['no_of_people'] = (X['no_of_adults'] + X['no_of_children'])
    # star rating
    # no of adults
    # selling amount

    return X


def fill_na(X):
    # for feature in ["request_airport", "request_twinbeds", "request_largebed", "request_highfloor"]:
    #     X[feature][X[feature].isna()] = 0
    # X[X[feature].isna()] = 0

    X.loc[(X["hotel_star_rating"] < 0) |
          (X["hotel_star_rating"] > 5), "hotel_star_rating"] = means["hotel_star_rating"]

    for feature in ["time_ahead", "staying_duration", "no_of_people"]:
        X.loc[(X[feature].isna()) | (X[feature] < 0), feature] = means[feature]
    return X


def preprocess_train(X, y):
    X = create_linear_features(X)
    X = create_dummy_features(X)
    # X = fill_na(X)
    X = remove_redundant_features(X)

    X, y = clean_data(X, y)

    return X, y


def preprocess_test(X, features):
    X = create_linear_features(X)
    X = create_dummy_features(X)
    X = remove_redundant_features(X)
    X = fill_na(X)

    # Reindexing X - removing columns that are not in the features list
    X = X.reindex(columns=features, fill_value=0)

    return X


def fit_over_dataset():
    df = pd.read_csv("../datasets/agoda_cancellation_train.csv",
                     parse_dates=['booking_datetime', 'checkin_date', 'checkout_date'])

    cancellation_datetime = df["cancellation_datetime"]
    df["original_selling_amount"] = df.apply(
        lambda x: (1 / currencies[x["original_payment_currency"]]) * x["original_selling_amount"], axis=1)

    X, y = df.drop(["original_selling_amount", "cancellation_datetime"], axis=1), df["original_selling_amount"]
    y = pd.Series(np.where(cancellation_datetime.isna(), -1, y))

    # X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(df)

    X_train, y_train = preprocess_train(X, y)

    np.save('X_train_columns_task_2.npy', X_train.columns)

    model = Ridge(alpha=3.5)
    model.fit(X_train, y_train)

    joblib.dump(model, 'ridge3.5.joblib', compress=9)


def run_task_2(input_file, output_file):
    model = joblib.load("./hackathon_code/ridge3.5.joblib")

    df = pd.read_csv(input_file, parse_dates=['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date'])
    selling_cur = df["original_payment_currency"]
    X_before = df.copy()

    X = preprocess_test(df, np.load("./hackathon_code/X_train_columns_task_2.npy", allow_pickle=True))

    y_pred = model.predict(X)
    X_before["original_selling_amount"] = y_pred

    y_cancel = task_1_prediction(X_before)
    y_pred[y_cancel == 0] = -1
    y_pred[y_pred < 0] = -1

    y_pred_curr = pd.DataFrame()
    y_pred_curr["original_selling_amount"] = y_pred.copy()
    y_pred_curr["original_payment_currency"] = selling_cur

    y_pred = y_pred_curr.apply(
        lambda y: -1 if y["original_selling_amount"] == -1 else currencies[y["original_payment_currency"]] * y["original_selling_amount"], axis=1)

    result = pd.DataFrame()
    result["id"] = df["h_booking_id"]
    result["predicted_selling_amount"] = y_pred.astype(int)

    result.to_csv(output_file, index=False)


if __name__ == "__main__":
    np.random.seed(0)

    fit_over_dataset()
