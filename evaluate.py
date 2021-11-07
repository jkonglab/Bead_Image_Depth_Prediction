# Evaluation.

import os

import utils
from opts import opt
from loader import data_generator2


def evaluate():
	data_path = opt.data_path
	model_name = opt.model_name
	im_size = opt.im_size
	num_class = opt.num_class
	batch_size = opt.batch_size
	multiGPU = opt.multiGPU
	save_logs_path = opt.save_logs_path
	gpus = opt.gpus

	if multiGPU:
		print("Constructing multi-GPU model ...")
		model = utils.get_gpu_model(model_name, im_size, num_class, gpus)
		weights_path = os.path.join(save_logs_path, model_name, 'fine_tune_model', "weights-best-acc_multi-gpu.hdf5")
	else:
		print("Constructing non-multi-GPU model ...")
		model = utils.get_original_model(model_name, im_size, num_class)
		weights_path = os.path.join(save_logs_path, model_name, 'fine_tune_model', "weights-best-acc_non-multi-gpu.hdf5")
	
	print("Loading saved model ...")
	model.load_weights(weights_path)

	model.compile(
		optimizer='rmsprop',
		loss='categorical_crossentropy',
		metrics=['accuracy']
	)

	print("Making test data generator ...")
	(_, _, test_generator), (_, _, _) = data_generator2(
		data_path=data_path,
		im_size=im_size,
		batch_size=batch_size,
		op='Test'
	)

	print("Start evaluating ...")
	res = model.evaluate_generator(
		generator=test_generator,
		verbose=1,
	)

	print(res)


if __name__ == "__main__":
	evaluate()











