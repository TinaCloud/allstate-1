__author__ = 'brandonkelly'

import pandas as pd
import os
import numpy as np
import itertools

base_dir = os.environ['HOME'] + '/Projects/Kaggle/allstate/'
data_dir = base_dir + 'data/'


def make_response_map():
    """ Construct the mapping that goes from the values of {A, B, C, D, E, F, G} to a unique integer value. """

    cat_values = {'A': (0, 1, 2), 'B': (0, 1), 'C': (1, 2, 3, 4), 'D': (1, 2, 3), 'E': (0, 1), 'F': (0, 1, 2, 3),
                  'G': (1, 2, 3, 4)}

    unique_combinations = list(itertools.product(*cat_values.values()))

    response_map = dict()
    class_label = 0
    for combo in unique_combinations:
        identifier = ''
        for s in combo:
            identifier += str(s)
        response_map[identifier] = class_label
        class_label += 1

    return response_map


def make_static_predictors(df):
    # ordinal and continous predictors
    cols = ['car_age', 'risk_factor', 'age_oldest', 'age_youngest', 'duration_previous', 'cost', 'homeowner',
            'married_couple']

    predictors = df[cols]

    # rename to use the prefix "is_" for consistency among binary-valued predictors
    predictors.rename(columns={'homeowner': 'is_homeowner', 'married_couple': 'is_married_couple'}, inplace=True)

    # convert time to continuous values
    print 'Converting time values...'
    predictors['time'] = 0.0
    for customer in predictors.index:
        hour, minute = df.ix[customer]['time'].split(':')
        tvalue = float(hour) + float(minute) / 60.0
        predictors.set_value(customer, 'time', tvalue)

    # convert car_value from letters to numbers
    predictors['car_value'] = 0
    predictors['car_value'][df['car_value'] == 'a'] = 1
    predictors['car_value'][df['car_value'] == 'b'] = 2
    predictors['car_value'][df['car_value'] == 'c'] = 3
    predictors['car_value'][df['car_value'] == 'd'] = 4
    predictors['car_value'][df['car_value'] == 'f'] = 5
    predictors['car_value'][df['car_value'] == 'g'] = 6
    predictors['car_value'][df['car_value'] == 'h'] = 7
    predictors['car_value'][df['car_value'] == 'i'] = 8

    # turn non-binary categorical predictors into binary-valued predictors
    for state in df['state'].unique():
        predictors['is_' + state] = df['state'] == state

    predictors['is_C_previous_nan'] = df['C_previous'].isnull()
    predictors['is_C_previous_1'] = df['C_previous'] == 1
    predictors['is_C_previous_2'] = df['C_previous'] == 2
    predictors['is_C_previous_3'] = df['C_previous'] == 3
    predictors['is_C_previous_4'] = df['C_previous'] == 4

    for gs in df['group_size']:
        predictors['is_group_size_' + str(gs)] = df['group_size'] == gs

    predictors['is_Mon'] = df['day'] == 0
    predictors['is_Tue'] = df['day'] == 1
    predictors['is_Wed'] = df['day'] == 2
    predictors['is_Thu'] = df['day'] == 3
    predictors['is_Fri'] = df['day'] == 4
    predictors['is_Sat'] = df['day'] == 5
    predictors['is_Sun'] = df['day'] == 6

    return predictors


def add_dynamic_predictors(df, predictors, last_shopping_pt):
    """ Create additional predictors based on customer shopping history. """
    customer_ids = df.index.get_level_values(0).unique()
    predictors['n_shopping'] = 0  # number of plan customer looked at before buying
    predictors['first_plan'] = 0  # label of plan customer first looked at
    predictors['last_plan'] = 0  # label of last plan customer looked at before purchasing
    predictors['fraction_last_plan'] = 0.0  # fraction of time customer looked at the last plan
    predictors['most_common_plan'] = 0  # label of plan customer most frequently looked at
    cat_values = {'A': (0, 1, 2), 'B': (0, 1), 'C': (1, 2, 3, 4), 'D': (1, 2, 3), 'E': (0, 1), 'F': (0, 1, 2, 3),
                  'G': (1, 2, 3, 4)}

    for category in cat_values.keys():
        for label in cat_values[category]:
            # fraction of times customer looked at this category label (e.g., A = {0, 1, 2}, etc.)
            predictors['fraction_' + category + str(label)] = 0.0

    rmap = make_response_map()

    print 'Constructing dynamic features for customer number (out of', len(customer_ids), ')'
    for i, customer in enumerate(customer_ids):
        print i+1, '...'
        this_df = df.ix[customer]
        plans = []
        category_count = dict()
        for category in cat_values.keys():
            for label in cat_values[category]:
                category_count[category + str(label)] = 0

        for shopping_pt in range(1, last_shopping_pt[i] + 1):
            plan_id = ''
            for category in cat_values.keys():
                # get unique planID for this sequence of category values
                plan_id += str(this_df.ix[shopping_pt][category])
                # count the number of times each category ('A', 'B', etc.) has a certain value
                label = this_df.ix[shopping_pt][category]
                category_count[category + str(label)] += 1

            plans.append(rmap[plan_id])

        predictors.set_value(customer, 'n_shopping', last_shopping_pt[i])
        predictors.set_value(customer, 'first_plan', plans[0])
        predictors.set_value(customer, 'last_plan', plans[-1])
        predictors.set_value(customer, 'fraction_last_plan', plans.count(plans[-1]) / float(len(plans)))
        # add predictors for each category
        for key in category_count.keys():
            predictors.set_value(customer, 'fraction_' + key, category_count[key] / float(len(plans)))
        # find most commonly-looked at plan
        unique_plans = np.unique(plans)
        pcounts = [plans.count(p) for p in unique_plans]
        predictors.set_value(customer, 'most_common_plan', unique_plans[np.argmax(pcounts)])

    return predictors


def compress_dtypes(df):

    for c in df.columns:
        if df[c].dtype == np.int64:
            if df[c].max() == 1:
                df[c] = df[c].astype(np.bool)
            elif df[c].max() < 256:
                df[c] = df[c].astype(np.uint8)
            else:
                df[c] = df[c].astype(np.uint16)
        elif df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)

    return df


def make_predictors(df, do_train=True):
    """Turn the raw data into a dataframe containing the predictors."""

    df.set_index(['customer_ID', 'shopping_pt'], inplace=True)

    customer_ids = df.index.get_level_values(0).unique()
    # find (customer ID, last shopping point) pairs
    print 'Getting last shopping point for each customer...'
    if do_train:
        last_idx = [(cid, df.ix[cid].index[-2]) for cid in customer_ids]
    else:
        last_idx = [(cid, df.ix[cid].index[-1]) for cid in customer_ids]

    # use the features from the shopping point before the one where the customer made a purchase, since some of the
    # predictors change
    last_df = df.ix[last_idx].reset_index(level=1)

    predictors = make_static_predictors(last_df)

    predictors = add_dynamic_predictors(df, predictors, last_df['shopping_pt'].values)

    cat_values = {'A': (0, 1, 2), 'B': (0, 1), 'C': (1, 2, 3, 4), 'D': (1, 2, 3), 'E': (0, 1), 'F': (0, 1, 2, 3),
                  'G': (1, 2, 3, 4)}

    print 'Adding planIDs to original dataframe...'
    if do_train:
        # add planIDs to the original dataframe for the training set
        rmap = make_response_map()
        df['planID'] = 0
        for i, customer in enumerate(customer_ids):
            this_df = df.ix[customer]
            for shopping_pt in this_df.index:
                plan_id = ''
                for category in cat_values.keys():
                    # get unique planID for this sequence of category values
                    plan_id += str(this_df.ix[shopping_pt][category])
                df.set_value((customer, shopping_pt), 'planID', rmap[plan_id])

    # compress data types
    print 'Compressing data types...'
    predictors = compress_dtypes(predictors)
    df = compress_dtypes(df)

    return df, predictors


def make_response(df):
    # create the response mapping. assumes that df has already been passed through make_predictors
    customer_ids = df.index.get_level_values(0).unique()
    purchased_idx = [(cid, df.ix[cid].index[-1]) for cid in customer_ids]
    purchased_df = df.ix[purchased_idx].reset_index(level=1)

    return purchased_df['planID']


if __name__ == "__main__":

    df = pd.read_csv(data_dir + 'train.csv')

    print 'Building training set...'
    df, predictors = make_predictors(df, do_train=True)

    response = make_response(df)

    df.to_hdf(data_dir + 'training_set.h5', 'df')
    predictors.to_hdf(data_dir + 'training_predictors.h5', 'df')
    response.to_csv(data_dir + 'training_response.csv')

    del df, predictors, response

    df = pd.read_csv(data_dir + 'test_v2.csv')

    print 'Building test set...'
    df, predictors = make_predictors(df, do_train=False)
    df.to_hdf(data_dir + 'test_set.h5', 'df')
    predictors.to_hdf(data_dir + 'test_predictors.h5', 'df')
