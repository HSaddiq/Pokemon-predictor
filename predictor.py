import pandas as pd
import tensorflow as tf
import numpy as np
from type_multiplier import find_type_multiplier
import data_parser as data_parser


def generate_input_data(pokemon_name_1, pokemon_name_2):
    pokemon_matrix = data_parser.generate_pokemon_matrix()

    for i in range(pokemon_matrix.shape[0]):
        # check if name is one of chosen values
        if pokemon_matrix[i][1] == pokemon_name_1:
            pokemon_1_data = list(pokemon_matrix[i])
            pokemon_1_data = pokemon_1_data[2:-2]

        elif pokemon_matrix[i][1] == pokemon_name_2:
            pokemon_2_data = list(pokemon_matrix[i])
            pokemon_2_data = pokemon_2_data[2:-2]

    pokemon_1_types = [type for type in pokemon_1_data[:2] if type != ""]
    pokemon_2_types = [type for type in pokemon_2_data[:2] if type != ""]

    output = [find_type_multiplier(pokemon_1_types,
                                   pokemon_2_types)] + pokemon_1_data[
                                                       2:] + [
                 find_type_multiplier(
                     pokemon_2_types, pokemon_1_types)] + pokemon_2_data[2:]

    return np.array([output])


def predict(pokemon_name_1, pokemon_name_2):
    # initialising tensorflow
    # import previously generated model
    saver = tf.train.import_meta_graph(
        "./trained_model/trained_predictor.meta")

    sess = tf.Session()

    saver.restore(sess, "./trained_model/trained_predictor")

    # generate input data line using names of pokemon
    input = generate_input_data(pokemon_name_1, pokemon_name_2)

    input = pd.DataFrame(input)
    input = input.as_matrix()

    prediction = sess.run("y3:0", feed_dict={"x:0": input})

    return prediction
