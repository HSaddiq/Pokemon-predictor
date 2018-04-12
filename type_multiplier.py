import numpy as np

# have separated into separate functions to reduce overhead
def initialise_type_advantage_matrix():
    type_matrix = np.genfromtxt("data/type_chart.csv", delimiter=',')
    type_matrix = np.delete(type_matrix, (0), axis=0)
    type_matrix = np.delete(type_matrix, (0), axis=1)
    return type_matrix


def intialise_type_dictionary():
    type_dictionary = {
        "Normal": 0,
        "Fire": 1,
        "Water": 2,
        "Electric": 3,
        "Grass": 4,
        "Ice": 5,
        "Fighting": 6,
        "Poison": 7,
        "Ground": 8,
        "Flying": 9,
        "Psychic": 10,
        "Bug": 11,
        "Rock": 12,
        "Ghost": 13,
        "Dragon": 14,
        "Dark": 15,
        "Steel": 16,
        "Fairy": 17
    }
    return type_dictionary


def find_type_multiplier(attacking_type_array, defending_type_array,
                         type_dictionary, type_advantage_matrix):
    multiplier = 1

    for attack_type in attacking_type_array:
        for defend_type in defending_type_array:
            multiplier *= type_advantage_matrix[type_dictionary[attack_type]][
                type_dictionary[defend_type]]

    return multiplier
