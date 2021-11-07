"""
Save the non-multi-GPU model.
Refer this [link](https://github.com/keras-team/keras/issues/11253#issuecomment-482467792) for more details.
"""

import os
import numpy as np
from keras.utils import multi_gpu_model

import utils
from opts import opt
from loader import data_generator2


def save_nmg_model():
	"""
	Do the saving.

	Returns:
		{ndarray} -- its shape is N*M, where `N` is the number of image samples, `M` equals to the number of classes.
	"""
	model_name = opt.model_name
	im_size = opt.im_size
	num_class = opt.num_class
	save_logs_path = opt.save_logs_path
	gpus = opt.gpus

	weights_path = os.path.join(save_logs_path, model_name, 'fine_tune_model', "weights-best-acc_multi-gpu.hdf5")
	if not os.path.isfile(weights_path):
		raise ValueError("Cannot find finle '{}'.".format(weights_path))

	# print("Constructing models ...")
	# model = utils.get_original_model(model_name, im_size, num_class)
	# gpu_model = multi_gpu_model(model, gpus=2, cpu_relocation=True)

	print("Constructing multi-GPU model ...")
	gpu_model = utils.get_gpu_model(model_name, im_size, num_class, gpus)

	print("Loading saved model ...")
	gpu_model.load_weights(weights_path)

	old_model = gpu_model.layers[-2]   # get single GPU model weights
	model_save_name = os.path.join(save_logs_path, model_name, 'fine_tune_model', "weights-best-acc_non-multi-gpu.hdf5")
	print("Saving model ...")
	old_model.save_weights(model_save_name)

	return


if __name__ == "__main__":
	save_nmg_model()
	























