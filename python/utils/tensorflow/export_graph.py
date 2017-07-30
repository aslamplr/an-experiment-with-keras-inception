import tensorflow as tf
from keras import backend as K
from keras.models import load_model


def export_graph(keras_model=load_model('../saved_models/inception_v3_full.h5'),
                 num_output=1,
                 prefix_output_node_names_of_final_network='output_node',
                 output_dir='tensorflow_model/',
                 output_graph_filename='constant_graph_weights.pb',
                 write_graph_def_ascii_flag=True,
                 write_graph_def_filename='only_the_graph_def.pb.ascii'):
    K.set_learning_phase(0)
    model = keras_model

    pred = [None] * num_output
    pred_node_names = [None] * num_output

    for i in range(num_output):
        pred_node_names[i] = prefix_output_node_names_of_final_network + str(i)
        pred[i] = tf.identity(model.output[i], name=pred_node_names[i])

    print('output node names are - ', pred_node_names)

    sess = K.get_session()

    if write_graph_def_ascii_flag:
        filename = write_graph_def_filename
        tf.train.write_graph(sess.graph.as_graph_def(), output_dir, filename, as_text=True)
        print('saved the graph definition in ascii format at - ', filename)

    from tensorflow.python.framework import graph_util, graph_io
    # Freeze the graph, convert all weight varibales to constants...
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_dir, output_graph_filename, as_text=False)
    print('saved the constant graph (ready for inference) at - ', output_graph_filename)
