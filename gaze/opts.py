import argparse
from gaze.model import model_names


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        required=True,
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--result_path',
        default='results',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--dataset',
        default='gazecapture',
        type=str,
        help='Used dataset (gazecapture)')
    parser.add_argument(
        '--lr',
        default=1e-3,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument(
        '--mean_dataset',
        default='gazecapture',
        type=str,
        help=
        'dataset for mean values of mean subtraction (gazecapture)')
    parser.add_argument(
        '--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=5,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--batch_size', default=256, type=int, help='Batch Size')
    parser.add_argument(
        '--face_size', default=224, type=int, help='Size of face bounding box to be sized to')
    parser.add_argument(
        '--n_epochs',
        default=30,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained models indicated by resume_path is loaded.'
    )
    parser.add_argument(
        '--checkpoint',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--pretrain', action='store_true', help='Use pretrained models')
    parser.set_defaults(pretrain=False)
    parser.add_argument(
        '--ft_begin_index',
        default=0,
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--save_interval',
        default=3,
        type=int,
        help='Trained models is saved at every this epochs.')
    parser.add_argument(
        '--model',
        default='facegaze',
        type=str,
        help='facegaze | eyegaze')
    parser.add_argument(
        '--conv',
        default='resnet18',
        type=str,
        help=model_names())
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_file", type=str, default=None, help="log file to log output to")
    parser.add_argument("--log_dir", type=str, help="log directory for Tensorboard log output")
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)

    args = parser.parse_args()
    if args.log_dir is None:
        args.log_dir = args.result_path

    return args
