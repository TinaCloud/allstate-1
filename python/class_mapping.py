__author__ = 'brandonkelly'

import itertools


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


def inverse_response_map():
    """ Construct the inverse mapping that goes from the planID to the sequence of ABCDEFG."""
    response_map = make_response_map()
    inverse_map = dict()
    category_index = dict()
    cat_values = {'A': (0, 1, 2), 'B': (0, 1), 'C': (1, 2, 3, 4), 'D': (1, 2, 3), 'E': (0, 1), 'F': (0, 1, 2, 3),
                  'G': (1, 2, 3, 4)}
    for i, key in enumerate(cat_values.keys()):
        # categories are not in alphabetical order, so grab their indices
        category_index[key] = i
    for options, plan_id in response_map.items():
        # need to sort the plan option labels in alphabetical order
        option_labels = ''
        for option in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            option_labels += options[category_index[option]]
        inverse_map[str(plan_id)] = option_labels

    return inverse_map