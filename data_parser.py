import csv
import numpy as np
import type_multiplier


# produces csv file of cleaned data, with features of each pokemon and label of
def data_cleaner():
    with open('data/pokemon.csv', 'rt') as csvfile:
        pokemon_reader = csv.reader(csvfile, delimiter=',')
        for row in pokemon_reader:
            print(row[0])


def generate_pokemon_matrix():
    pokemon_matrix = np.genfromtxt("data/pokemon.csv", delimiter=',',
                                   dtype=None, encoding=None)
    return pokemon_matrix


# takes row of combats, returns a list appropriate for appending to cleaned_data

def generate_element_csv(row, pokemon_matrix):
    first_pokemon_attr = [i for i in pokemon_matrix[row[0] - 1]]
    second_pokemon_attr = [i for i in pokemon_matrix[row[1] - 1]]
    winner_id = row[2]

    output_list = []

    # finding the attacking type multiplier

    attack_array = list(filter(None, first_pokemon_attr[2:4]))
    defense_array = list(filter(None, second_pokemon_attr[2:4]))

    attacking_type_multiplier = type_multiplier.find_type_multiplier(
        attack_array, defense_array)
    output_list.append(attacking_type_multiplier)

    # append relevant first pokemon attribute
    output_list += first_pokemon_attr[4:10]

    # finding the defending type multiplier
    defending_type_multiplier = type_multiplier.find_type_multiplier(
        defense_array, attack_array)
    output_list.append(defending_type_multiplier)

    # append relevant second pokemon attributes
    output_list += second_pokemon_attr[4:10]

    # append outcome
    if row[0] == row[2]:
        winner = 0

    else:
        winner = 1

    output_list.append(winner)

    return output_list


pokemon_matrix = generate_pokemon_matrix()

if __name__ == "__main__":
    with open('data/cleaned_data.csv', 'a', newline='') as cleaned_csvfile:
        with open('data/combats.csv', 'rt') as raw_csvfile:
            pokemon_reader = csv.reader(raw_csvfile, delimiter=',')
            writer = csv.writer(cleaned_csvfile)
            next(raw_csvfile)
            for row in pokemon_reader:
                row = [int(i) for i in row]
                attributes = generate_element_csv(row, pokemon_matrix)
                writer.writerow(attributes)

        raw_csvfile.close()
    cleaned_csvfile.close()
