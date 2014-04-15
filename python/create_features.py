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


def add_static_predictors(df):
    # rename to use the prefix "is_" for consistency among binary-valued predictors
    df.rename(columns={'homeowner': 'is_homeowner', 'married_couple': 'is_married_couple'}, inplace=True)

    # convert time to continuous values
    print 'Converting time values...'

    def time_to_float(tstr):
        hour, minute = tstr.split(':')
        tvalue = float(hour) + float(minute) / 60.0
        return tvalue

    df['time'] = df['time'].map(time_to_float)

    print 'Converting car_value from string to integer.'

    # convert car_value from letters to numbers
    uvals = np.sort(df['car_value'].unique())
    carv_mapping = {cv: v for cv, v in zip(uvals, range(1, len(uvals)+1))}
    df['car_value'] = df['car_value'].map(carv_mapping)

    print 'Binarizing state values...'
    # turn non-binary categorical predictors into binary-valued predictors
    for state in df['state'].unique():
        df['is_' + state] = df['state'] == state

    print 'Binarizing C_previous values...'

    df['is_C_previous_nan'] = df['C_previous'].isnull()
    df['is_C_previous_1'] = df['C_previous'] == 1
    df['is_C_previous_2'] = df['C_previous'] == 2
    df['is_C_previous_3'] = df['C_previous'] == 3
    df['is_C_previous_4'] = df['C_previous'] == 4

    print 'Binarizing group size values...'

    for gs in np.sort(df['group_size'].unique()):
        print gs, '...'
        df['is_group_size_' + str(gs)] = df['group_size'] == gs

    print 'Binarizing day values...'

    df['is_Mon'] = df['day'] == 0
    df['is_Tue'] = df['day'] == 1
    df['is_Wed'] = df['day'] == 2
    df['is_Thu'] = df['day'] == 3
    df['is_Fri'] = df['day'] == 4
    df['is_Sat'] = df['day'] == 5
    df['is_Sun'] = df['day'] == 6

    print 'Finding missing data...'
    fill_values = {'risk_factor': df['risk_factor'].mean(), 'duration_previous': df['duration_previous'].mean()}
    df['is_risk_factor_missing'] = pd.isnull(df['risk_factor'])
    df['is_duration_previous_missing'] = pd.isnull(df['duration_previous'])
    df = df.fillna(value=fill_values)

    return df


def add_dynamic_predictors(df, last_shopping_pt):
    """ Create additional predictors based on customer shopping history. """
    customer_ids = df.index.get_level_values(0).unique()
    df['planID'] = 0  # unique identifier for plan customer looked at
    df['fraction_last_plan'] = 0.0  # fraction of time customer looked at the last plan
    df['most_common_plan'] = 0  # label of plan customer most frequently looked at
    cat_values = {'A': (0, 1, 2), 'B': (0, 1), 'C': (1, 2, 3, 4), 'D': (1, 2, 3), 'E': (0, 1), 'F': (0, 1, 2, 3),
                  'G': (1, 2, 3, 4)}

    for category in cat_values.keys():
        df['fraction_changed_' + category] = 0.0
        for label in cat_values[category]:
            # fraction of times customer looked at this category label (e.g., A = {0, 1, 2}, etc.)
            df['fraction_' + category + str(label)] = 0.0
            df['is_last_value_' + category + str(label)] = False

    rmap = make_response_map()

    print 'Constructing dynamic features for customer number (out of', len(customer_ids), ')'
    for i, customer in enumerate(customer_ids):
        print i+1, '...'
        this_df = df.ix[customer]
        plans = []
        category_count = dict()
        category_change_count = dict()
        for category in cat_values.keys():
            category_change_count[category] = 0.0
            for label in cat_values[category]:
                category_count[category + str(label)] = 0.0

        for shopping_pt in range(1, last_shopping_pt[i] + 1):
            plan_id = ''
            for category in cat_values.keys():
                # get unique planID for this sequence of category values
                plan_id += str(this_df.ix[shopping_pt][category])
                # count the number of times each category ('A', 'B', etc.) has a certain value
                label = this_df.ix[shopping_pt][category]
                category_count[category + str(label)] += 1
                df.set_value((customer, shopping_pt), 'is_last_value_' + category + str(label), True)
                if shopping_pt > 1:
                    # check for a change in plan
                    if label != previous_label:
                        category_change_count[category] += 1
                previous_label = label

            plans.append(rmap[plan_id])
            df.set_value((customer, shopping_pt), 'planID', plans[-1])
            df.set_value((customer, shopping_pt), 'fraction_last_plan', plans.count(plans[-1]) / float(len(plans)))
            # find most commonly-looked at plan
            unique_plans = np.unique(plans)
            pcounts = [plans.count(p) for p in unique_plans]
            df.set_value((customer, shopping_pt), 'most_common_plan', unique_plans[np.argmax(pcounts)])

            # add predictors for each category
            for key in category_count.keys():
                df.set_value((customer, shopping_pt), 'fraction_' + key, category_count[key] / float(len(plans)))
            if shopping_pt > 1:
                for category in category_change_count.keys():
                    df.set_value((customer, shopping_pt), 'fraction_changed_' + category,
                                 category_change_count[category] / float(len(plans)-1))

    return df


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


def make_predictors(df):
    """Turn the raw data into a dataframe containing the predictors."""

    # TODO: deal with missing values

    df.set_index(['customer_ID', 'shopping_pt'], inplace=True)

    customer_ids = df.index.get_level_values(0).unique()
    # find (customer ID, last shopping point) pairs
    print 'Getting last shopping point for each customer...'

    # use the features from the shopping point before the one where the customer made a purchase, since some of the
    # predictors change
    df = add_static_predictors(df)

    last_shopping_pnt = [df.ix[cid].index[-1] for cid in customer_ids]

    df = add_dynamic_predictors(df, last_shopping_pnt)

    # compress data types
    print 'Compressing data types...'
    df = compress_dtypes(df)

    return df


def make_response(df):
    # create the response mapping. assumes that df has already been passed through make_predictors
    customer_ids = df.index.get_level_values(0).unique()
    purchased_idx = [(cid, df.ix[cid].index[-1]) for cid in customer_ids]
    purchased_df = df.ix[purchased_idx].reset_index(level=1)

    return purchased_df['planID']


if __name__ == "__main__":

    df = pd.read_csv(data_dir + 'train.csv')

    print 'Building training set...'
    df = make_predictors(df)

    response = make_response(df)

    df.to_hdf(data_dir + 'training_set.h5', 'df')
    response.to_csv(data_dir + 'training_response.csv')

    del df, response

    df = pd.read_csv(data_dir + 'test_v2.csv')

    print 'Building test set...'
    df = make_predictors(df)
    df.to_hdf(data_dir + 'test_set.h5', 'df')
