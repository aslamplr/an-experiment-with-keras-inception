from python.utils.tensorflow import export_graph
from keras.models import load_model

def main():
    keras_model = load_model('../saved_models/inception_v3_full.h5')
    export_graph(keras_model,
                 num_output=1,
                 prefix_output_node_names_of_final_network='output_node',
                 output_dir='../tensorflow_model',
                 output_graph_filename='constant_graph_weights.pb',
                 write_graph_def_ascii_flag=True,
                 write_graph_def_filename='only_the_graph_def.pb.ascii')


if __name__ == '__main__':
    main()