import argparse

from tensorflow.python.tools import freeze_graph
from tensorpack import *
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.tfutils.export import ModelExport

from deploy.tf_models_utils import convert_pb_to_pbtxt
from detection.config.tensorpack_config import finalize_configs, config as cfg
from detection.tensorpacks.train import ResNetFPNModel, ResNetC4Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training.', required=True)
    parser.add_argument('--outdir', help='Path to export directory', required=True)
    parser.add_argument('--freeze', action='store_true',
                        help='Whether to freeze the graph, which in turn leads to different format of output pb file.')
    parser.add_argument('--pbtxt', action='store_true',
                        help='Also output a readable pbtxt file along with pb file.')
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in tensorpack_config.py",
                        nargs='+')

    if get_tf_version_tuple() < (1, 6):
        # https://github.com/tensorflow/tensorflow/issues/14657
        logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN if you're unlucky.")

    args = parser.parse_args()

    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    finalize_configs(is_training=False)

    # if args.predict or args.visualize:
    #     cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

    if not args.freeze:
        e = ModelExport(MODEL, MODEL.get_inference_tensor_names()[0], MODEL.get_inference_tensor_names()[1])
        e.export(args.load, args.outdir)
    else:
        print("Freezing the graph as a pb file")
        pb_file = os.path.join(args.outdir, 'frozen_graph.pb')
        freeze_graph.freeze_graph(
            input_saved_model_dir=args.outdir,
            output_graph=pb_file,
            output_node_names=','.join(MODEL.get_inference_tensor_names()[1]),
            input_graph=None,
            input_saver=None,
            input_binary=None,
            input_checkpoint=None,
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            clear_devices=True,
            initializer_nodes=''
        )

        if args.pbtxt:
            convert_pb_to_pbtxt(pb_file, args.outdir, 'frozen_graph.pbtxt')

