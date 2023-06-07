import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def clean_data(X: pd.DataFrame):
    mean_per_feature[f] = X.loc[X[f] > 0, f].mean()


def split_data(df: pd.DataFrame):
    # Calculate the ratio of non-cancelled reservations and all reservations
    print(df["cancellation_datetime"].isna().sum() / len(df))

    # Divide into X and y
    X, y = df.drop(["cancellation_datetime"], axis=1), df["cancellation_datetime"]

    # Find a random_state for which the ration is similar to the previously calculated
    # for i in range(100):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    #     print(f"{i}th iteration:", y_train.isna().sum() / len(y_train))
    #     print(f"{i}th iteration:", y_test.isna().sum() / len(y_test))

    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.1, random_state=0)

    return X_train, X_dev, X_test, y_train, y_dev, y_test

def remove_redundant_features(X):
    X = X.drop(['h_booking_id', 'hotel_id', 'hotel_area_code', 'hotel_chain_code', 'hotel_live_date', 'h_customer_id',
                'customer_nationality', 'no_of_extra_bed', 'no_of_room', 'language', 'original_payment_currency',
                'request_nonesmoke', 'request_latecheckin', 'request_highfloor', 'request_largebed',
                'request_twinbeds', 'request_airport', 'request_earlycheckin'])

    return X.drop(['checkin_date', 'checkout_date', 'booking_datetime', 'charge_option', 'no_of_children',
                   'hotel_country_code', 'origin_country_code'])

def create_dummy_features(X):
    X['checkin_month'] = X['checkin_date'].dt.month

    X = pd.get_dummies(X, prefix='checkin_month', columns=['checkin_month'])
    X = pd.get_dummies(X, prefix='hotel_city_code', columns=['hotel_city_code'])
    X = pd.get_dummies(X, prefix='hotel_brand_code', columns=['hotel_brand_code'])
    X = pd.get_dummies(X, prefix='accommadation_type_name', columns=['accommadation_type_name'])
    X = pd.get_dummies(X, prefix='guest_nationality_country_name', columns=['guest_nationality_country_name'])
    X = pd.get_dummies(X, prefix='original_payment_method', columns=['original_payment_method'])
    X = pd.get_dummies(X, prefix='original_payment_type', columns=['original_payment_type'])
    X = pd.get_dummies(X, prefix='cancellation_policy_code', columns=['cancellation_policy_code'])

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
    X['time_ahead'] = (X['checking_date'] - df['booking_datetime']) / np.timedelta64(1, 'D')
    X['staying_duration'] = (X['checkout_date'] - df['checking_date']) / np.timedelta64(1, 'D')
    X['no_of_people'] = (X['no_of_adults'] + X['no_of_children'])
    # star rating
    # no of adults
    # selling amount
    return X

def preprocess_train(X, y):
    X = create_dummy_features(X)
    X = create_boolean_features(X)
    X = create_linear_features(X)

    X = remove_redundant_features(X)

if __name__ == "__main__":
    np.random.seed(0)

    df = pd.read_csv("../datasets/agoda_cancellation_train.csv",
                     parse_dates=['booking_datetime', 'checkin_date', 'checkout_date'])

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(df)

    print(len(X_train["customer_nationality"].unique()))
    preprocess_train(X_train, y_train)


# TODO: restore columns for test cases