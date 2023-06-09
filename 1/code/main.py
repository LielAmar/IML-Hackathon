import sys
import pandas as pd

from hackathon_code.task_1 import *
from hackathon_code.task_2 import *

if __name__ == "__main__":

    if len(sys.argv) == 3:
        df1 = pd.read_csv(sys.argv[1], parse_dates=['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date'])
        df2 = pd.read_csv(sys.argv[2], parse_dates=['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date'])

        run_task_1(df1, "../predictions/agoda_cancellation_prediction.csv")
        run_task_2(df2, "../predictions/agoda_cost_of_cancellation.csv")


        # Task 1 graphs:
        # create_estimators_comparison_graph(df1)
        # create_train_vs_test_graph(df1)

        # Task 2 graphs:
        # create_ridge_lasso_graph(df2)
