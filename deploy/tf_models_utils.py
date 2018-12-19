import os

import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format


def convert_pb_to_pbtxt(pb_file, output_path, pbtxt_file):
    with gfile.FastGFile(pb_file, 'rb') as f:
        graph_def = tf.GraphDef()

        graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name='')

        tf.train.write_graph(graph_def, output_path, pbtxt_file, as_text=True)
    return


def convert_pbtxt_to_pb(pbtxt_file, output_path, pb_file):
    """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.

    Args:
      filename: The name of a file containing a GraphDef pbtxt (text-formatted
        `tf.GraphDef` protocol buffer data).

    """
    with tf.gfile.FastGFile(pbtxt_file, 'r') as f:
        graph_def = tf.GraphDef()

        file_content = f.read()

        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, output_path, pb_file, as_text=False)
    return


if __name__ == '__main__':
    outdir = '/root/models/detection/results/tf-models/'
    convert_pb_to_pbtxt(os.path.join(outdir, 'frozen_inference_graph.pb'), outdir, 'frozen_graph.pbtxt')