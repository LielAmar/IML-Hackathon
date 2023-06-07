import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def remove_problematic_samples(X, y):
    X = X[~X["cancellation_policy_code"].isna()]
    y = y.loc[X.index]

    return X, y

def clean_data(X: pd.DataFrame, y: pd.Series):
    X = X[X["no_of_adults"] < 20]
    X = X[X["no_of_people"] < 20]
    X = X[X["original_selling_amount"] < 9000]
    X = X[X["time_ahead"] >= -1]
    X[X["time_ahead"] <= 0] = 0
    y = y.loc[X.index]

    # TODO: save means for test preprocess
    means = dict()
    means["hotel_star_rating"] = np.mean(X[(~X["hotel_star_rating"].isna())
                                           & (X["hotel_star_rating"] >= 0)
                                           & (X["hotel_star_rating"] <= 5)
                                           & (X["hotel_star_rating"] % 0.5 == 0)]["hotel_star_rating"])
    X.loc[(X["hotel_star_rating"] < 1) |
          (X["hotel_star_rating"] > 5) |
          (X["hotel_star_rating"] % 0.5 != 0), "hotel_star_rating"] = means["hotel_star_rating"]

    for feature in ["original_selling_amount", "time_ahead", "staying_duration", "no_of_people"]:
        means[feature] = np.mean(X[(~X[feature].isna()) & X[feature] >= 0][feature])
        X.loc[(X[feature].isna()) | (X[feature] < 0), feature] = means[feature]

    return X, y


def split_data(df: pd.DataFrame):
    # Divide into X and y
    X, y = df.drop(["cancellation_datetime"], axis=1), df["cancellation_datetime"]

    # Divide into train, dev and test
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.1, random_state=0)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def remove_redundant_features(X):
    X = X.drop(['h_booking_id', 'hotel_id', 'hotel_area_code', 'hotel_chain_code', 'hotel_live_date', 'h_customer_id',
                'customer_nationality', 'no_of_extra_bed', 'no_of_room', 'language', 'original_payment_currency',
                'request_nonesmoke', 'request_latecheckin', 'request_highfloor', 'request_largebed',
                'request_twinbeds', 'request_airport', 'request_earlycheckin'], axis=1)

    return X.drop(['checkin_date', 'checkout_date', 'booking_datetime', 'charge_option', 'no_of_children',
                   'hotel_country_code', 'origin_country_code'], axis=1)

def create_dummy_features(X):
    X['checkin_month'] = X['checkin_date'].dt.month

    X = pd.get_dummies(X, prefix='checkin_month', columns=['checkin_month'])
    X = pd.get_dummies(X, prefix='hotel_city_code', columns=['hotel_city_code'])
    X = pd.get_dummies(X, prefix='hotel_brand_code', columns=['hotel_brand_code'])
    X = pd.get_dummies(X, prefix='accommadation_type_name', columns=['accommadation_type_name'])
    X = pd.get_dummies(X, prefix='guest_nationality_country_name', columns=['guest_nationality_country_name'])
    X = pd.get_dummies(X, prefix='original_payment_method', columns=['original_payment_method'])
    X = pd.get_dummies(X, prefix='original_payment_type', columns=['original_payment_type'])
    # X = pd.get_dummies(X, prefix='cancellation_policy_code', columns=['cancellation_policy_code'])
    # TODO: handle cancellation_policy_code

    return X

def create_boolean_features(X):
    X['pay_now'] = (X['charge_option'] == "Pay Now")
    X['children'] = (X['no_of_children'] > 0)
    X['same_country_order'] = (X['hotel_country_code'] == X['origin_country_code'])
    # guest is not the customer
    # logged in
    # first booking
    return X

def create_linear_features(X):
    X['time_ahead'] = np.round((X['checkin_date'] - X['booking_datetime']) / np.timedelta64(1, 'D'))
    X['staying_duration'] = np.round((X['checkout_date'] - X['checkin_date']) / np.timedelta64(1, 'D'))
    X['no_of_people'] = (X['no_of_adults'] + X['no_of_children'])
    # star rating
    # no of adults
    # selling amount
    return X

def calculate_worth(time_duration, tuple_info):
    cancelation_days = tuple_info[0]
    percent = tuple_info[1]
    a = max(max(30, cancelation_days*2), time_duration)
    worth = cancelation_days/a * percent
    return worth
    # Use a breakpoint in the code line below to debug your script.

def calculate_total_worth(price ,time_duration , info_tuple_list):
    sum = 0
    two_to_the = 1
    list = []
    for i in range(0, len(info_tuple_list)):
        list.append(calculate_worth(time_duration, info_tuple_list[i]))
    list.sort()
    list.reverse()
    for i in range(0, len(list)):
        sum += two_to_the * list[i]
        two_to_the *= 0.5
    #sum += 2 * two_to_the * calculate_worth(time_duration, info_tuple_list[-1])
    return sum/2

def receive_policy(price, time_duration, nights, policy):
    if(time_duration == 0):
        return 1
        # TODO: change me
    policies = policy.split("_")
    info_list = []
    for pol in policies:
        match = re.search(r"((\d+)D)?(\d+)([PN])", pol)
        if match:
            if match.group(1):
                if pol[-1] == "N":
                    info_list.append((int(match.group(2)), (100 * int(match.group(3))) / nights))
                else:
                    info_list.append((int(match.group(2)), int(match.group(3))))
            else:
                pass
                # TODO: handel no-show;
    return calculate_total_worth(price, time_duration, info_list)

def create_cancellation_policy_feature(X):
    for index, row in X.iterrows():
        time_duration = row["time_ahead"]
        nights = row["staying_duration"]
        policy = row["cancellation_policy_code"]
        price = row["original_selling_amount"]
        X.at[index, "cancellation_policy_code_2"] = receive_policy(price, time_duration, nights, policy)

    return X

def preprocess_train(X, y):
    X, y = remove_problematic_samples(X, y)
    y = ~y.isna()

    X = create_dummy_features(X)
    X = create_boolean_features(X)
    X = create_linear_features(X)
    X = create_cancellation_policy_feature(X)

    X = remove_redundant_features(X)

    X, y = clean_data(X, y)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X.drop(['cancellation_policy_code'], axis=1))
    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    transformed_data = pca.transform(scaled_data)

    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=y, s=30)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Result')
    plt.savefig('pca_result.png')
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)

    df = pd.read_csv("../datasets/agoda_cancellation_train.csv",
                     parse_dates=['booking_datetime', 'checkin_date', 'checkout_date'])

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(df)

    preprocess_train(X_train, y_train)

# TODO: restore columns for test cases
