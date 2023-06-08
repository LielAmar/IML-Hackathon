import numpy as np
import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt


means = {'hotel_star_rating': 2.7379305082159027,
         'original_selling_amount': 204.4114964134914,
         'time_ahead': 30.059532563322083,
         'staying_duration': 1.783537671058656,
         'no_of_people': 2.1357277305794375}

columns = ['hotel_star_rating', 'guest_is_not_the_customer', 'no_of_adults', 'original_selling_amount',
           'is_user_logged_in', 'is_first_booking', 'checkin_month_6', 'checkin_month_7', 'checkin_month_8',
           'checkin_month_9', 'accommadation_type_name_Apartment', 'accommadation_type_name_Boat / Cruise',
           'accommadation_type_name_Bungalow', 'accommadation_type_name_Capsule Hotel',
           'accommadation_type_name_Chalet', 'accommadation_type_name_Guest House / Bed & Breakfast',
           'accommadation_type_name_Holiday Park / Caravan Park', 'accommadation_type_name_Home',
           'accommadation_type_name_Homestay', 'accommadation_type_name_Hostel', 'accommadation_type_name_Hotel',
           'accommadation_type_name_Inn', 'accommadation_type_name_Lodge', 'accommadation_type_name_Love Hotel',
           'accommadation_type_name_Motel', 'accommadation_type_name_Private Villa', 'accommadation_type_name_Resort',
           'accommadation_type_name_Resort Villa', 'accommadation_type_name_Ryokan',
           'accommadation_type_name_Serviced Apartment', 'accommadation_type_name_Tent',
           'accommadation_type_name_UNKNOWN', 'guest_nationality_country_name_Afghanistan',
           'guest_nationality_country_name_Albania', 'guest_nationality_country_name_Algeria',
           'guest_nationality_country_name_Andorra', 'guest_nationality_country_name_Angola',
           'guest_nationality_country_name_Argentina', 'guest_nationality_country_name_Australia',
           'guest_nationality_country_name_Austria', 'guest_nationality_country_name_Azerbaijan',
           'guest_nationality_country_name_Bahamas', 'guest_nationality_country_name_Bahrain',
           'guest_nationality_country_name_Bangladesh', 'guest_nationality_country_name_Barbados',
           'guest_nationality_country_name_Belarus', 'guest_nationality_country_name_Belgium',
           'guest_nationality_country_name_Benin', 'guest_nationality_country_name_Bhutan',
           'guest_nationality_country_name_Botswana', 'guest_nationality_country_name_Brazil',
           'guest_nationality_country_name_Brunei Darussalam', 'guest_nationality_country_name_Bulgaria',
           'guest_nationality_country_name_Burkina Faso', 'guest_nationality_country_name_Cambodia',
           'guest_nationality_country_name_Cameroon', 'guest_nationality_country_name_Canada',
           'guest_nationality_country_name_Chile', 'guest_nationality_country_name_China',
           'guest_nationality_country_name_Colombia', 'guest_nationality_country_name_Costa Rica',
           "guest_nationality_country_name_Cote D'ivoire", 'guest_nationality_country_name_Croatia',
           'guest_nationality_country_name_Cyprus', 'guest_nationality_country_name_Czech Republic',
           'guest_nationality_country_name_Democratic Republic of the Congo', 'guest_nationality_country_name_Denmark',
           'guest_nationality_country_name_Egypt', 'guest_nationality_country_name_Estonia',
           'guest_nationality_country_name_Faroe Islands', 'guest_nationality_country_name_Fiji',
           'guest_nationality_country_name_Finland', 'guest_nationality_country_name_France',
           'guest_nationality_country_name_French Guiana', 'guest_nationality_country_name_French Polynesia',
           'guest_nationality_country_name_Gambia', 'guest_nationality_country_name_Georgia',
           'guest_nationality_country_name_Germany', 'guest_nationality_country_name_Ghana',
           'guest_nationality_country_name_Greece', 'guest_nationality_country_name_Guam',
           'guest_nationality_country_name_Guatemala', 'guest_nationality_country_name_Guinea',
           'guest_nationality_country_name_Hong Kong', 'guest_nationality_country_name_Hungary',
           'guest_nationality_country_name_Iceland', 'guest_nationality_country_name_India',
           'guest_nationality_country_name_Indonesia', 'guest_nationality_country_name_Iraq',
           'guest_nationality_country_name_Ireland', 'guest_nationality_country_name_Isle Of Man',
           'guest_nationality_country_name_Israel', 'guest_nationality_country_name_Italy',
           'guest_nationality_country_name_Japan', 'guest_nationality_country_name_Jersey',
           'guest_nationality_country_name_Jordan', 'guest_nationality_country_name_Kazakhstan',
           'guest_nationality_country_name_Kenya', 'guest_nationality_country_name_Kuwait',
           'guest_nationality_country_name_Laos', 'guest_nationality_country_name_Latvia',
           'guest_nationality_country_name_Lebanon', 'guest_nationality_country_name_Lithuania',
           'guest_nationality_country_name_Luxembourg', 'guest_nationality_country_name_Macau',
           'guest_nationality_country_name_Malaysia', 'guest_nationality_country_name_Maldives',
           'guest_nationality_country_name_Mali', 'guest_nationality_country_name_Malta',
           'guest_nationality_country_name_Mauritius', 'guest_nationality_country_name_Mexico',
           'guest_nationality_country_name_Monaco', 'guest_nationality_country_name_Mongolia',
           'guest_nationality_country_name_Montenegro', 'guest_nationality_country_name_Morocco',
           'guest_nationality_country_name_Mozambique', 'guest_nationality_country_name_Myanmar',
           'guest_nationality_country_name_Nepal', 'guest_nationality_country_name_Netherlands',
           'guest_nationality_country_name_New Caledonia', 'guest_nationality_country_name_New Zealand',
           'guest_nationality_country_name_Nigeria', 'guest_nationality_country_name_Northern Mariana Islands',
           'guest_nationality_country_name_Norway', 'guest_nationality_country_name_Oman',
           'guest_nationality_country_name_Pakistan', 'guest_nationality_country_name_Palestinian Territory',
           'guest_nationality_country_name_Papua New Guinea', 'guest_nationality_country_name_Peru',
           'guest_nationality_country_name_Philippines', 'guest_nationality_country_name_Poland',
           'guest_nationality_country_name_Portugal', 'guest_nationality_country_name_Puerto Rico',
           'guest_nationality_country_name_Qatar', 'guest_nationality_country_name_Reunion Island',
           'guest_nationality_country_name_Romania', 'guest_nationality_country_name_Russia',
           'guest_nationality_country_name_Saudi Arabia', 'guest_nationality_country_name_Senegal',
           'guest_nationality_country_name_Singapore', 'guest_nationality_country_name_Sint Maarten (Netherlands)',
           'guest_nationality_country_name_Slovakia', 'guest_nationality_country_name_Slovenia',
           'guest_nationality_country_name_South Africa', 'guest_nationality_country_name_South Korea',
           'guest_nationality_country_name_South Sudan', 'guest_nationality_country_name_Spain',
           'guest_nationality_country_name_Sri Lanka', 'guest_nationality_country_name_Sweden',
           'guest_nationality_country_name_Switzerland', 'guest_nationality_country_name_Taiwan',
           'guest_nationality_country_name_Thailand', 'guest_nationality_country_name_Togo',
           'guest_nationality_country_name_Trinidad & Tobago', 'guest_nationality_country_name_Tunisia',
           'guest_nationality_country_name_Turkey', 'guest_nationality_country_name_UNKNOWN',
           'guest_nationality_country_name_Uganda', 'guest_nationality_country_name_Ukraine',
           'guest_nationality_country_name_United Arab Emirates', 'guest_nationality_country_name_United Kingdom',
           'guest_nationality_country_name_United States', 'guest_nationality_country_name_Uruguay',
           'guest_nationality_country_name_Uzbekistan', 'guest_nationality_country_name_Venezuela',
           'guest_nationality_country_name_Vietnam', 'guest_nationality_country_name_Yemen',
           'guest_nationality_country_name_Zambia', 'guest_nationality_country_name_Zimbabwe', 'children',
           'same_country_order', 'existence', 'time_ahead', 'staying_duration', 'no_of_people', 'no_show',
           'cancellation_policy_30', 'cancellation_policy_21', 'cancellation_policy_14', 'cancellation_policy_7',
           'cancellation_policy_3', 'cancellation_policy_1']

# Currencies as of 16.8.18
currencies = {
    "AED": 3.673181,
    "AFN": 72.823258,
    "ALL": 110.58,
    "AMD": 482.840272,
    "ANG": 1.843953,
    "AOA": 270.3925,
    "ARS": 29.724,
    "AUD": 1.3776,
    "AWG": 1.7925,
    "AZN": 1.7025,
    "BAM": 1.72015,
    "BBD": 2,
    "BDT": 84.453473,
    "BGN": 1.71989,
    "BHD": 0.377088,
    "BIF": 1772.564836,
    "BMD": 1,
    "BND": 1.51076,
    "BOB": 6.90853,
    "BRL": 3.905406,
    "BSD": 1,
    "BTC": 0.000158408561,
    "BTN": 70.250498,
    "BWP": 10.8055,
    "BYN": 2.0534,
    "BZD": 2.008866,
    "CAD": 1.31605,
    "CDF": 1626.551914,
    "CHF": 0.997384,
    "CLF": 0.02338,
    "CLP": 669.261521,
    "CNH": 6.868637,
    "CNY": 6.88415,
    "COP": 3047.975024,
    "CRC": 567.12861,
    "CUC": 1,
    "CUP": 25.5,
    "CVE": 97.3,
    "CZK": 22.6324,
    "DJF": 178,
    "DKK": 6.557082,
    "DOP": 49.876505,
    "DZD": 119.029816,
    "EGP": 17.902,
    "ERN": 14.9965,
    "ETB": 27.568,
    "EUR": 0.879356,
    "FJD": 2.116905,
    "FKP": 0.786624,
    "GBP": 0.786624,
    "GEL": 2.482717,
    "GGP": 0.786624,
    "GHS": 4.874117,
    "GIP": 0.786624,
    "GMD": 48.16,
    "GNF": 9036.151169,
    "GTQ": 7.4894,
    "GYD": 209.103315,
    "HKD": 7.84972,
    "HNL": 24.03,
    "HRK": 6.5276,
    "HTG": 67.3475,
    "HUF": 284.843812,
    "IDR": 14344.516583,
    "ILS": 3.67127,
    "IMP": 0.786624,
    "INR": 70.015,
    "IQD": 1191.56269,
    "IRR": 43163.26868,
    "ISK": 108.329847,
    "JEP": 0.786624,
    "JMD": 135.5825,
    "JOD": 0.709503,
    "JPY": 111.01484,
    "KES": 100.790129,
    "KGS": 68.137481,
    "KHR": 4070.169979,
    "KMF": 432.952376,
    "KPW": 900,
    "KRW": 1127.32,
    "KWD": 0.303322,
    "KYD": 0.832898,
    "KZT": 360.032943,
    "LAK": 8518.942724,
    "LBP": 1511,
    "LKR": 160.413162,
    "LRD": 154.549609,
    "LSL": 14.255,
    "LYD": 1.392443,
    "MAD": 9.5652,
    "MDL": 16.632113,
    "MGA": 3324.709248,
    "MKD": 54.135,
    "MMK": 1527.15,
    "MNT": 2442.166667,
    "MOP": 8.0806,
    "MRO": 357.5,
    "MRU": 35.95,
    "MUR": 34.850029,
    "MVR": 15.459996,
    "MWK": 727.203141,
    "MXN": 18.998056,
    "MYR": 4.1055,
    "MZN": 58.989229,
    "NAD": 14.537382,
    "NGN": 361.020294,
    "NIO": 31.871276,
    "NOK": 8.481203,
    "NPR": 112.403423,
    "NZD": 1.518912,
    "OMR": 0.38496,
    "PAB": 1,
    "PEN": 3.312,
    "PGK": 3.311548,
    "PHP": 53.425938,
    "PKR": 122.755692,
    "PLN": 3.78625,
    "PYG": 5750.35,
    "QAR": 3.641064,
    "RON": 4.0961,
    "RSD": 103.760733,
    "RUB": 66.8698,
    "RWF": 877.665,
    "SAR": 3.7507,
    "SBD": 7.88911,
    "SCR": 13.588838,
    "SDG": 17.990205,
    "SEK": 9.194411,
    "SGD": 1.37585,
    "SHP": 0.786624,
    "SLL": 6542.71,
    "SOS": 578.345,
    "SRD": 7.458,
    "SSP": 130.2634,
    "STD": 21050.59961,
    "STN": 21.575,
    "SVC": 8.745692,
    "SYP": 514.97999,
    "SZL": 14.537219,
    "THB": 33.195021,
    "TJS": 9.419687,
    "TMT": 3.509961,
    "TND": 2.769793,
    "TOP": 2.310538,
    "TRY": 5.852199,
    "TTD": 6.736358,
    "TWD": 30.76086,
    "TZS": 2286.489273,
    "UAH": 27.619953,
    "UGX": 3751.002075,
    "USD": 1,
    "UYU": 31.573521,
    "UZS": 7791.674003,
    "VEF": 141572.666667,
    "VND": 23114.085172,
    "VUV": 108.499605,
    "WST": 2.588533,
    "XAF": 576.819847,
    "XAG": 0.06772851,
    "XAU": 0.00084235,
    "XCD": 2.70255,
    "XDR": 0.717117,
    "XOF": 576.819847,
    "XPD": 0.00101,
    "XPF": 104.935106,
    "XPT": 0.00127078,
    "YER": 250.3,
    "ZAR": 14.56358,
    "ZMW": 10.218987,
    "ZWL": 322.355011
}


def split_data(df: pd.DataFrame, include_dev=False):
    # Divide into X and y
    X, y = df.drop(["cancellation_datetime"], axis=1), df["cancellation_datetime"]

    # Divide into train, dev and test
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    if include_dev:
        X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.1, random_state=0)

        return X_train, X_dev, X_test, y_train, y_dev, y_test

    return X_train_dev, X_test, y_train_dev, y_test


def remove_problematic_samples(X, y):
    X = X[~X["cancellation_policy_code"].isna()]
    y = y.loc[X.index]

    return X, y


def create_dummy_features(X):
    X['checkin_month'] = X['checkin_date'].dt.month

    X = pd.get_dummies(X, prefix='checkin_month', columns=['checkin_month'])
    X = pd.get_dummies(X, prefix='accommadation_type_name', columns=['accommadation_type_name'])
    X = pd.get_dummies(X, prefix='guest_nationality_country_name', columns=['guest_nationality_country_name'])

    return X


def create_boolean_features(X):
    X['children'] = (X['no_of_children'] > 0)
    X['same_country_order'] = (X['hotel_country_code'] == X['origin_country_code'])
    return X


def create_linear_features(X):
    X['existence'] = (X['hotel_live_date'] - X['booking_datetime']) / np.timedelta64(1, 'D')
    X['time_ahead'] = (X['checkin_date'] - X['booking_datetime']) / np.timedelta64(1, 'D')
    X['staying_duration'] = (X['checkout_date'] - X['checkin_date']) / np.timedelta64(1, 'D')
    X['no_of_people'] = (X['no_of_adults'] + X['no_of_children'])

    # X['original_selling_amount'] = X.apply(lambda x:
    #                                        (1 / currencies[x["original_payment_currency"]]) * x[
    #                                            "original_selling_amount"], axis=1)
    return X


def create_cancellation_policy_feature(X):
    def calculate_penalty(penalty, x):
        if penalty.endswith("P"):
            return int(penalty.split("P")[0]) / 100

        price_per_night = x["original_selling_amount"] / x["staying_duration"]

        return (price_per_night * int(penalty.split("N")[0])) / x["original_selling_amount"]

    def calculate_no_show(x):
        if not re.search(r"[^D](\d+)([PN])", x["cancellation_policy_code"]):
            return 0

        no_show_penalty = x["cancellation_policy_code"].split("_")[-1]

        if "D" in no_show_penalty:
            return 0

        return calculate_penalty(no_show_penalty, x)

    X['no_show'] = X.apply(lambda x: calculate_no_show(x), axis=1)

    def policy_penalty(x, num):
        all_policies = x["cancellation_policy_code"]
        policies = all_policies.split("_")

        max_penalty = 0
        recent = 365

        for policy in policies:
            if "D" not in policy:
                continue

            data = policy.split("D")
            days, penalty = int(data[0]), data[1]

            if days < num or days > recent:
                continue

            recent = days
            max_penalty = max(calculate_penalty(penalty, x), max_penalty)

        return max_penalty

    X["cancellation_policy_30"] = X.apply(lambda x: policy_penalty(x, 30), axis=1)
    X["cancellation_policy_21"] = X.apply(lambda x: policy_penalty(x, 21), axis=1)
    X["cancellation_policy_14"] = X.apply(lambda x: policy_penalty(x, 14), axis=1)
    X["cancellation_policy_7"] = X.apply(lambda x: policy_penalty(x, 7), axis=1)
    X["cancellation_policy_3"] = X.apply(lambda x: policy_penalty(x, 3), axis=1)
    X["cancellation_policy_1"] = X.apply(lambda x: policy_penalty(x, 1), axis=1)

    return X


def remove_redundant_features(X):
    X = X.drop(['h_booking_id', 'hotel_id', 'hotel_area_code', 'hotel_brand_code', 'hotel_live_date', 'h_customer_id',
                'customer_nationality', 'no_of_extra_bed', 'no_of_room', 'language', 'original_payment_currency',
                'request_nonesmoke', 'request_latecheckin', 'request_highfloor', 'request_largebed',
                'request_twinbeds', 'request_airport', 'request_earlycheckin'], axis=1)

    return X.drop(['checkin_date', 'checkout_date', 'booking_datetime', 'no_of_children',
                   'hotel_country_code', 'origin_country_code', 'cancellation_policy_code',
                   'charge_option', 'hotel_chain_code', 'hotel_city_code',
                   'original_payment_type', 'original_payment_method'], axis=1)


def clean_data(X: pd.DataFrame, y: pd.Series=None, is_test=False):
    if not is_test:
        X = X[X["no_of_adults"] < 20]
        X = X[X["no_of_people"] < 20]
        X = X[X["time_ahead"] >= -1]
        y = y.loc[X.index]

    X[X["time_ahead"] <= 0] = 0

    X.loc[(X["hotel_star_rating"] < 0) |
          (X["hotel_star_rating"] > 5), "hotel_star_rating"] = means["hotel_star_rating"]

    for feature in ["original_selling_amount", "time_ahead", "staying_duration", "no_of_people"]:
        X.loc[(X[feature].isna()) | (X[feature] < 0), feature] = means[feature]

    if is_test:
        return X

    return X, y


def preprocess_train(X, y):
    X, y = remove_problematic_samples(X, y)

    X = create_dummy_features(X)
    X = create_boolean_features(X)
    X = create_linear_features(X)
    X = create_cancellation_policy_feature(X)

    X = remove_redundant_features(X)

    X, y = clean_data(X, y)

    return X, y


def preprocess_test(X, features):
    X = create_boolean_features(X)
    X = create_linear_features(X)
    X = create_dummy_features(X)
    X = create_cancellation_policy_feature(X)

    X = clean_data(X, is_test=True)
    # Reindexing X - removing columns that are not in the features list
    X = X.reindex(columns=features, fill_value=0)

    return X


def run_estimator_testing(X, y, X_dev, y_dev):
    print("Running estimator tester...")

    models = {
        "xg": XGBClassifier(max_depth=3,
                            learning_rate=0.2,
                            n_estimators=500,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            gamma=0.1,
                            reg_alpha=0.1,
                            reg_lambda=0.1,
                            scale_pos_weight=1.0,
                            eval_metric='logloss'),
        # "lda": LinearDiscriminantAnalysis(),
        "lda": LinearDiscriminantAnalysis(n_components=1),
        "forest5": RandomForestClassifier(n_estimators=5),
        "forest50": RandomForestClassifier(n_estimators=50),
        # "forest100": RandomForestClassifier(n_estimators=100),
        # "forest250": RandomForestClassifier(n_estimators=250),
        # "forest500": RandomForestClassifier(n_estimators=500),
        "svm": SVC(),
        "tree2": DecisionTreeClassifier(max_depth=2),
        # "tree3": DecisionTreeClassifier(max_depth=3),
        # "tree4": DecisionTreeClassifier(max_depth=4),
        "tree5": DecisionTreeClassifier(max_depth=5),
        "ada2_2": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=2),
        # "ada2_4": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=4),
        # "ada2_5": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=5),
        "ada2_50": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=50),
        # "ada2_100": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100),
        # "ada2_500": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=500),
        # "ada2_900": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=900),
        # "ada3_2": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=2),
        # "ada3_4": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=4),
        # "ada3_5": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=5),
        # "ada5_2": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=2),
        # "ada5_4": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=4),
        # "ada5_5": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=5),
        # "logistic_l2_1": LogisticRegression(max_iter=5000, penalty="l2", C=0.1),
        "logistic_l2_2": LogisticRegression(max_iter=5000, penalty="l2", C=0.2),
        # "logistic_l2_3": LogisticRegression(max_iter=5000, penalty="l2", C=0.3),
        # "logistic_l2_4": LogisticRegression(max_iter=5000, penalty="l2", C=0.4),
        # "logistic_l2_5": LogisticRegression(max_iter=5000, penalty="l2", C=0.5)
    }

    best_model_score = 0
    best_model_name = None

    X = X.astype(float)

    # f1_scores_macro = []
    # f1_scores_binary = []
    # mse_scores = []
    # accuracy_scores = []

    scores = {}
    errors = ["macro", "binary", "mse", "accuracy"]

    for name, model in models.items():
        print(f"Testing {name}...")

        model.fit(X, ~y.isna())
        y_pred = model.predict(X_dev)

        score = f1_score(y_pred, ~y_dev.isna(), average="macro")
        print(f"Score for {name} is:", score)

        if best_model_score < score:
            best_model_score = score
            best_model_name = name

        scores[name] = []

        scores[name].append(f1_score(y_pred, ~y_dev.isna(), average="macro"))
        scores[name].append(f1_score(y_pred, ~y_dev.isna()))
        scores[name].append(mean_squared_error(y_pred.astype(int), ~y_dev.isna().astype(int)))
        scores[name].append(accuracy_score(y_pred, ~y_dev.isna()))

    print(f"The best found model is {best_model_name} with a score of {best_model_score}")

    X_axis = np.arange(len(errors))
    width = 0.075

    plt.figure(figsize=(20, 9))

    xg = scores["xg"]
    bar_xg = plt.bar(X_axis, xg, width)

    lda = scores["lda"]
    bar_lda = plt.bar(X_axis+width, lda, width)

    forest5 = scores["forest5"]
    bar_forest5 = plt.bar(X_axis+width*2, forest5, width)

    forest50 = scores["forest50"]
    bar_forest50 = plt.bar(X_axis+width*3, forest50, width)

    svm = scores["svm"]
    bar_svm = plt.bar(X_axis+width*4, svm, width)

    tree2 = scores["tree2"]
    bar_tree2 = plt.bar(X_axis+width*5, tree2, width)

    tree5 = scores["tree5"]
    bar_tree5 = plt.bar(X_axis+width*6, tree5, width)

    ada2_2 = scores["ada2_2"]
    bar_ada2_2 = plt.bar(X_axis+width*7, ada2_2, width)

    ada2_50 = scores["ada2_50"]
    bar_ada2_50 = plt.bar(X_axis+width*8, ada2_50, width)

    logistic_l2_2 = scores["logistic_l2_2"]
    bar_logistic_l2_2 = plt.bar(X_axis+width*9, logistic_l2_2, width)

    plt.xlabel("Error Types")
    plt.ylabel('Score')
    plt.title("Model Score as function of Error Types")

    plt.xticks(X_axis + width, ['F1 Macro', 'F1 Binary', 'MSE', 'Accuracy'])
    plt.legend((bar_xg, bar_lda, bar_forest5, bar_forest50, bar_svm, bar_tree2, bar_tree5, bar_ada2_2, bar_ada2_50, bar_logistic_l2_2),
               ('XGBoost', 'LDA', 'Forest (5)', 'Forest (50)', 'SVM', 'Tree (2)', 'Tree (5)', 'AdaBoost (2)', 'AdaBoost (50)', 'Logistic Regression + L2'))

    plt.savefig("fig.png")


def fit_over_dataset():
    df = pd.read_csv("../datasets/agoda_cancellation_train.csv",
                     parse_dates=['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date'])

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(df, include_dev=True)
    # X_train, X_test, y_train, y_test = split_data(df, include_dev=False)

    X_train, y_train = preprocess_train(X_train, y_train)
    X_train = X_train.astype(float)

    X_dev = preprocess_test(X_dev, X_train.columns.tolist())
    # X_test = preprocess_test(X_test, columns)

    run_estimator_testing(X_train, y_train, X_dev, y_dev)

    # Chosen model: Forest with 50 estimators
    # model = RandomForestClassifier(n_estimators=50)
    # model = XGBClassifier(max_depth=3,
    #                       learning_rate=0.2,
    #                       n_estimators=500,
    #                       subsample=0.8,
    #                       colsample_bytree=0.8,
    #                       gamma=0.1,
    #                       reg_alpha=0.1,
    #                       reg_lambda=0.1,
    #                       scale_pos_weight=1.0,
    #                       eval_metric='logloss')
    # model.fit(X_train, ~y_train.isna())
    #
    # joblib.dump(model, 'xg500.joblib', compress=9)
    #
    # y_pred = model.predict(X_test)
    #
    # score = f1_score(y_pred, ~y_test.isna(), average="macro")
    # print("score is: ", score)


def run_task_1(input_file, output_file):
    model = joblib.load("./hackathon_code/xg500.joblib")

    df = pd.read_csv(input_file, parse_dates=['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date'])

    X = preprocess_test(df, columns)

    y_pred = model.predict(X)

    result = pd.DataFrame()
    result["id"] = df["h_booking_id"]
    result["cancellation"] = y_pred.astype(int)

    result.to_csv(output_file, index=False)


if __name__ == "__main__":
    np.random.seed(0)

    fit_over_dataset()
