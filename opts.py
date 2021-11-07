# Arguments definition.

import argparse, os


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32, help='input image batch size. Defaults to 32.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate. Defaults to 0.0001.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay. Defaults to 0.9.')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs. Defaults to 100.')
parser.add_argument('--im_size', type=int, default=32, help='input image size of the network model. Defaults to 256.')
parser.add_argument('--num_class', type=int, default=2, help='number of classes. Defaults to 2.')
parser.add_argument('--save_logs_path', '--save_checkpoint_path', type=str, default='./logs', help='where checkpoints and logs saved.')
parser.add_argument('--load_logs_path', '--load_checkpoint_path', type=str, default='./logs', help='where checkpoints and logs loaded from.')
parser.add_argument('--weights_path', type=str, help='path of the saved model weights file.')
parser.add_argument('--data_path', type=str, default='', help='where dataset saved. See loader.py for dataset storage structure.')
parser.add_argument('--pred_save_path', type=str, default='', help='where predictions saved.')
parser.add_argument('--model_name', type=str, default='', help='model name. See models.py for optional model names.')
parser.add_argument('--interval', type=int, default=3, help='step size for traversing the image to be predicted. Defaults to 3.')
parser.add_argument('--rect_size', type=int, default=30, help="square's side length. The square is drawn for non maximum suppression. Defaults to 30.")
parser.add_argument('--load_pred_path', type=str, default='', help="directory where the predictions are stored.")
parser.add_argument('--data_format', type=str, default='', help="data format which determines the slide format and cell type.")
parser.add_argument('--threshold', type=float, default=1, help="Threshold of the density map, is used to get cell area mask.")
parser.add_argument('--demo_save_path', type=str, default='', help="directory storing demos.")
parser.add_argument('--pred_path', type=str, default='', help="directory storing predictions.")
parser.add_argument('--plan_group', type=int, default=0, help="plan group number, used for chooseing a group of slides.")
parser.add_argument('--pred_save_name', type=str, help="path to save the prediction. The suffix name or file format is '.npy'.")
parser.add_argument('--multiGPU', action='store_true', help="using this parameter will incuring the construction/loading multi-GPU model. This requires the number of GPUs is not less than two.")
parser.add_argument('--gpus', type=int, default=2, help="number of GPUs to be used. If `multiGPU` is `False`, this argument is ignored. Default: 2.")

opt = parser.parse_args()

args = vars(opt)
print('----------------- Options -------------------')
for k, v in sorted(args.items()):
    print('{}: {}'.format(str(k), str(v)))
print('------------------- End ---------------------')

if opt.save_logs_path != '' and not os.path.isdir(opt.save_logs_path):
    os.mkdir(opt.save_logs_path)
