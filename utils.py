import os

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.layers import Dense
from keras.utils import multi_gpu_model

import models


def get_original_model(model_name, im_size, num_class):
	"""
	Load the trained model's weights. Return a trained model. 

	Arguments:
		model_name {str} -- model's name, see models.py for details.
		im_size {int} input image size.
		num_class {int} number of classes.

	Returns:
		{Keras Network Model} -- trained model. 
	"""
	base_model, _ = getattr(models, model_name)(im_size)
	x = base_model.output
	x = Dense(1024, activation='relu')(x)
	predictions = Dense(num_class, activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=predictions)
	return model


def get_gpu_model(model_name, im_size, num_class, gpus):
	"""
	Load the trained model's weights. Return a trained model. 

	Arguments:
		model_name {str} -- model's name, see models.py for details.
		im_size {int} input image size.
		num_class {int} number of classes.
		gpus {int} -- number of GPUs to be used.

	Returns:
		{Keras Network Model} -- trained model. 
	"""
	model = get_original_model(model_name, im_size, num_class)
	gpu_model = multi_gpu_model(model, gpus=gpus, cpu_relocation=True)
	return gpu_model


def get_img_files(img_path):
	"""
	Get all the image file names in directory img_path.

	Arguments: 
		img_path {str} -- path of the image directory.

	Returns:
		list of str -- all the image names. 
		int -- the total number of the image files in the directory. 
	"""
	img_formats = ['JPG','PNG','TIF','BMP','TIFF']
	s_files = os.listdir(img_path)
	img_files = [x for x in s_files if x.split('.')[-1].upper() in img_formats]
	img_num = len(img_files)
	return img_files, img_num


def get_callbacks(save_path, batch_size, multiGPU, callback_cat):
	"""
	Return keras model callbacks. 

	Arguments: 
		save_path {str} -- saveing path of callbacks.
		batch_size {int} -- batch size.
		multiGPU {bool} -- if True, use save multi-GPU model, else save non-multi-GPU model.
		callback_cat {list of str} -- callback categories. Options are: 'ModelCheckpointImprove', 'ModelCheckpointBest', 'CSVLogger', 'TensorBoard', 'ReduceLROnPlateau'.

	Returns: 
		{list of callback} -- list of selected callbacks. 
	"""
	if not os.path.isdir(save_path):
		os.makedirs(save_path)

	callbacklist = []

	if 'ModelCheckpointImprove' in callback_cat:
		# Checkpoints: improvment.
		if multiGPU:
			file_path = os.path.join(save_path, "weights-improvement-{epoch:03d}-{val_acc:.3f}_multi-gpu.hdf5")
		else:
			file_path = os.path.join(save_path, "weights-improvement-{epoch:03d}-{val_acc:.3f}_non-multi-gpu.hdf5")
		checkpoint_imprv = ModelCheckpoint(
			filepath=file_path,
			monitor='val_acc', 
			verbose=1,
			save_best_only=True, 
			mode='max')
		callbacklist.append(checkpoint_imprv)

	if 'ModelCheckpointBestAcc' in callback_cat:
		# Checkpoints: best model.
		if multiGPU:
			file_path = os.path.join(save_path,"weights-best-acc_multi-gpu.hdf5")
		else:
			file_path = os.path.join(save_path,"weights-best-acc_non-multi-gpu.hdf5")
		checkpoint_best_acc = ModelCheckpoint(
			filepath=file_path, 
			monitor='val_acc', 
			verbose=1,
			save_best_only=True, 
			mode='max')
		callbacklist.append(checkpoint_best_acc)
	
	if 'ModelCheckpointBestLoss' in callback_cat:
		# Checkpoints: best model.
		if multiGPU:
			file_path = os.path.join(save_path,"weights-best-loss_multi-gpu.hdf5")
		else:
			file_path = os.path.join(save_path,"weights-best-loss_non-multi-gpu.hdf5")
		checkpoint_best_loss = ModelCheckpoint(
			filepath=file_path, 
			monitor='val_loss',
			verbose=1,
			save_best_only=True, 
			mode='min')
		callbacklist.append(checkpoint_best_loss)

	if 'CSVLogger' in callback_cat:
		# CSVLogger.
		file_name = os.path.join(save_path,"training_history.csv")
		csv_logger = CSVLogger(
			filename=file_name, 
			append=False)
		callbacklist.append(csv_logger)

	if 'TensorBoard' in callback_cat:
		# TensorBoard
		tensor_board = TensorBoard(
			log_dir=save_path,
			batch_size=batch_size,
			write_graph=True)
		callbacklist.append(tensor_board)

	if 'ReduceLROnPlateau' in callback_cat:
		# ReduceLROnPlateau
		reduce_lr_on_plateau = ReduceLROnPlateau(
			monitor='val_loss', 
			factor=0.2,
			patience=5,
			min_lr=1.0e-12)
		callbacklist.append(reduce_lr_on_plateau)

	if 'EarlyStopping' in callback_cat:
		# EarlyStopping
		early_stopping = EarlyStopping(
			monitor='val_loss',
			patience=10,
			verbose=1
		)
		callbacklist.append(early_stopping)

	return callbacklist











