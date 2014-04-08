__author__ = 'brandonkelly'

import pandas as pd
import os
import numpy as np
import itertools

base_dir = os.environ['HOME'] + '/Projects/Kaggle/allstate/'
data_dir = base_dir + 'data/'


def make_response_map(df):
    """ Construct the mapping that goes from the values of {A, B, C, D, E, F, G} to a unique integer value. """
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    nclasses = 1
    for cat in categories:
        nclasses *= len(set(df[cat]))

    combos = [set(df[c]) for c in categories]
    unique_combinations = list(itertools.product(*combos))

    response_map = dict()
    class_label = 0
    for combo in unique_combinations:
        identifier = ''
        for s in combo:
            identifier += str(s)
        response_map[identifier] = class_label
        class_label += 1

    return response_map


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

    # ordinal and continous predictors
    cols = ['customer_ID', 'day', 'car_age', 'risk_factor', 'age_oldest', 'age_youngest',
            'duration_previous', 'cost', 'homeowner', 'married_couple']

    predictors = last_df[cols]

    # rename to use the prefix "is_" for consistency among binary-valued predictors
    predictors.rename(columns={'homeowner': 'is_homeowner', 'married_couple': 'is_married_couple'}, inplace=True)

    # convert time to continuous values
    print 'Converting time values...'
    predictors['time'] = 0.0
    for idx in xrange(len(predictors.index)):
        hour, minute = last_df.ix[idx]['time'].split(':')
        tvalue = float(hour) + float(minute) / 60.0
        predictors.set_value(idx, 'time', tvalue)

    # convert car_value from letters to numbers
    predictors['car_value'] = 0
    predictors['car_value'][last_df['car_value'] == 'a'] = 1
    predictors['car_value'][last_df['car_value'] == 'b'] = 2
    predictors['car_value'][last_df['car_value'] == 'c'] = 3
    predictors['car_value'][last_df['car_value'] == 'd'] = 4
    predictors['car_value'][last_df['car_value'] == 'f'] = 5
    predictors['car_value'][last_df['car_value'] == 'g'] = 6
    predictors['car_value'][last_df['car_value'] == 'h'] = 7
    predictors['car_value'][last_df['car_value'] == 'i'] = 8

    # turn non-binary categorical predictors into binary-valued predictors
    for state in df['state'].unique():
        predictors['is_' + state] = last_df['state'] == state

    predictors['is_C_previous_nan'] = last_df['C_previous'].isnull()
    predictors['is_C_previous_1'] = last_df['C_previous'] == 1
    predictors['is_C_previous_2'] = last_df['C_previous'] == 2
    predictors['is_C_previous_3'] = last_df['C_previous'] == 3
    predictors['is_C_previous_4'] = last_df['C_previous'] == 4

    for gs in last_df['group_size']:
        predictors['is_group_size_' + str(gs)] = last_df['group_size'] == gs

    predictors['is_Mon'] = last_df['day'] == 0
    predictors['is_Tue'] = last_df['day'] == 1
    predictors['is_Wed'] = last_df['day'] == 2
    predictors['is_Thu'] = last_df['day'] == 3
    predictors['is_Fri'] = last_df['day'] == 4
    predictors['is_Sat'] = last_df['day'] == 5
    predictors['is_Sun'] = last_df['day'] == 6


    # create additional predictors based on customer shopping history
    predictors['n_shopping'] = last_df['shopping_pt']  # number of plan customer looked at before buying
    predictors['first_plan'] = 0  # label of plan customer first looked at
    predictors['last_plan'] = 0  # label of last plan customer looked at before purchasing
    predictors['fraction_last_plan'] = 0.0  # fraction of time customer looked at the last plan
    predictors['most_common_plan'] = 0  # label of plan customer most frequently looked at
    for category in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        for label in set(last_df[category]):
            # fraction of times customer looked at this category label (e.g., A = {0, 1, 2}, etc.)
            predictors['fraction_' + category + '_' + str(label)] = 0.0

    rmap = make_response_map(df)
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    print 'Constructing features for customer number'
    for i, customer in enumerate(customer_ids):
        print i+1, '...'
        this_df = df.ix[customer]
        plans = []
        for shopping_pt in this_df.index[:-1]:
            plan_id = ''
            for category in categories:
                plan_id += str(this_df.ix[shopping_pt][category])
            plans.append(rmap[plan_id])
        predictors.set_value(customer, 'first_plan', plans[0])
        predictors.set_value(customer, 'last_plan', plans[-1])
        predictors.set_value(customer, 'fraction_last_plan', plans.count(plans[-1]) / float(len(plans)))
        pcounts = []
        unique_plans = np.unique(list)
        pcounts = [plans.count(p) for p in unique_plans]
        predictors.set_value(customer, 'most_common_plan', unique_plans[pcounts.argmax()])


    # compress data types


def make_response(df):
    # create the response mapping
    response_map = dict()  # map the categorical responses to unique set of integer class labels



if __name__ == "__main__":
    df = pd.read_csv(data_dir + 'train_csv')
