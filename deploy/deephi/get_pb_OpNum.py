import argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np


# method-1: from .pb
# note: if "Incompleted shape" occurs, refer to method-2
def parse_pb(filename):
    with tf.Session() as sess:
        output_graph_def = tf.GraphDef()
        with open(filename, "rb") as f:
            # import graph from .pb file
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
            sess.graph.as_default()
            init = sess.run(tf.global_variables_initializer())
            #tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            #for tensor_name in tensor_name_list:
            #    print(tensor_name)
            graph = sess.graph

            # save model as .pbtxt file
            # tf.train.write_graph(sess.graph, '.','test.py.pbtxt')
            '''
            output = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
            img_val = np.random.rand(1,299,299,3)
            sess.run([output], feed_dict={"input:0": img_val})
            '''
            parameters = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
            print ('total parameters: {}'.format(parameters.total_parameters))
            flops = tf.profiler.profile(sess.graph, options = tf.profiler.ProfileOptionBuilder.float_operation())
            print('FLOPs: {}', format(flops.total_float_ops))


# method-2: from .pbtxt
# note: the input shape must be assigned, especially the batch_size (set to be 1 from -1)
def parse_pbtxt(filename):
    from tensorflow.core.framework import graph_pb2 as gpb
    from google.protobuf import text_format as pbtf

    gdef = gpb.GraphDef()
    with open(filename, 'r') as fh:
        graph_str = fh.read()
        pbtf.Parse(graph_str, gdef)
        tf.import_graph_def(gdef)
        sess = tf.Session()
        parameters = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        print ('total parameters: {}'.format(parameters.total_parameters))
        flops = tf.profiler.profile(sess.graph, options = tf.profiler.ProfileOptionBuilder.float_operation())
        print('FLOPs: {}', format(flops.total_float_ops))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_file', help='Model file to be parsed.')

    args = parser.parse_args()
    ext = args.graph_file.split(".")[-1]
    if ext == 'pb':
        parse_pb(args.graph_file)
    elif ext == 'pbtxt':
        parse_pbtxt(args.graph_file)
    else:
        raise Exception