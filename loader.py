# Load training dataset as image generators.
import os

import numpy as np
import pandas as pd
from cv2 import cv2
from skimage.transform import resize

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from utils import get_img_files


def data_generator(data_path, im_size, batch_size=50):
	"""
	Get image array and their corresponding label array from the directory.

	Data path structure is as following:

	train_data/
		train/
			class1/
			class2/
			...
		val/
			class1/
			class2/
			...
		test/
			class1/
			class2/
			...
		class-labels.xlsx
	
	In each subsubfolder, there are many images. The spreadsheet 'class-labels.xlsx' contains the class name list.

	Arguments:
		data_path {str} -- data path.
		im_size {int} -- length of the input image. Suppose the input image is square.
		batch_size {int} -- training and validation data batch size. Default: 50. 

	Returns:
		{DirectoryIterator} -- yielding tuples of (x, y) where x is a numpy array containing a batch of images with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels. Training data generator.
		{DirectoryIterator} -- validation data generator.
		{DirectoryIterator} -- testing data generator.
	"""
	# create data generator
	# this is the augmentation configuration we will use for training
	train_datagen = ImageDataGenerator(
		rescale=1./255,
		vertical_flip=True,
		horizontal_flip=True)

	# this is the augmentation configuration we will use for testing:
	# only rescaling
	test_datagen = ImageDataGenerator(rescale=1./255)

	# list of classes
	lab_df = pd.read_excel(os.path.join(data_path, 'class-labels.xlsx'), index_col=0)
	classes = lab_df.loc[:,'LabelText'].to_list()

	train_generator = train_datagen.flow_from_directory(
		directory=os.path.join(data_path, 'train'),
		target_size=(im_size, im_size),
		color_mode='rgb',
		classes=classes,
		class_mode='categorical',
		batch_size=batch_size,
		shuffle=True,
	)

	val_generator = test_datagen.flow_from_directory(
		directory=os.path.join(data_path, 'val'),
		target_size=(im_size, im_size),
		color_mode='rgb',
		classes=classes,
		class_mode='categorical',
		batch_size=batch_size,
		shuffle=False,
	)
	
	test_generator = test_datagen.flow_from_directory(
		directory=os.path.join(data_path, 'test'),
		target_size=(im_size, im_size),
		color_mode='rgb',
		classes=classes,
		class_mode='categorical',
		batch_size=batch_size,
		shuffle=False,
	)

	return train_generator, val_generator, test_generator


def data_generator2(data_path, im_size, batch_size=50, op='Train'):
	"""
	Get image array and their corresponding label array into a big matrix and then encapsule them into an iterator.

	Data path structure is as following:

	train_data/
		train/
			class1/
			class2/
			...
		val/
			class1/
			class2/
			...
		test/
			class1/
			class2/
			...
		class-labels.xlsx
	
	In each subsubfolder, there are many images. The spreadsheet 'class-labels.xlsx' contains the class name list.

	Arguments:
		data_path {str} -- data path.
		im_size {int} -- length of the input image. Suppose the input image is square.
		batch_size {int} -- training and validation data batch size. Default: 50.
		op {str} -- operation label. If 'Train', return training and validation data generator, if 'Test', return testing data generator. Default: 'Train'.

	Returns:
		{tuple of Iterator} -- Each Iterator yields tuples of (x, y) where x is a numpy array of image data (in the case of a single image input) or a list of numpy arrays (in the case with additional inputs) and y is a numpy array of corresponding labels. The tuple is (TrainingGenerator, ValidationGenerator, TestingGenerator).
		{tuple of int} -- the tuple is (TrainingImage#, ValidationImage#, TestingImage#).
	"""
	ops = {'train':'Train', 'test':'Test'}

	# create data generator
	# this is the augmentation configuration we will use for training
	train_datagen = ImageDataGenerator(
		rescale=1./255,
		vertical_flip=True,
		horizontal_flip=True)

	# this is the augmentation configuration we will use for testing:
	# only rescaling
	test_datagen = ImageDataGenerator(rescale=1./255)

	# list of classes
	lab_df = pd.read_excel(os.path.join(data_path, 'class-labels.xlsx'), index_col=0)

	if op==ops['train']:
		train_img, train_lab = get_img_lab(os.path.join(data_path, 'train'), lab_df, im_size)
		val_img, val_lab = get_img_lab(os.path.join(data_path, 'val'), lab_df, im_size)
		test_img, test_lab = np.zeros((1,1,1,1)), np.zeros((1,1))

	elif op==ops['test']:
		train_img, train_lab = np.zeros((1,1,1,1)), np.zeros((1,1))
		val_img, val_lab = np.zeros((1,1,1,1)), np.zeros((1,1))
		test_img, test_lab = get_img_lab(os.path.join(data_path, 'test'), lab_df, im_size)

	else:
		raise ValueError("Options for 'op' are: {}. Current 'op' is '{}'.".format([ops[x] for x in ops], op))

	train_generator = train_datagen.flow(
		x=train_img,
		y=train_lab,
		batch_size=batch_size,
		shuffle=True,
	)

	val_generator = test_datagen.flow(
		x=val_img,
		y=val_lab,
		batch_size=batch_size,
		shuffle=False,
	)

	test_generator = test_datagen.flow(
		x=test_img,
		y=test_lab,
		batch_size=batch_size,
		shuffle=False,
	)

	train_num = train_img.shape[0]
	val_num = val_img.shape[0]
	test_num = test_img.shape[0]
	
	print('Train image array shape: {}'.format(train_img.shape))
	print('Validation image array shape: {}'.format(val_img.shape))
	print('Test image array shape: {}'.format(test_img.shape))

	return (train_generator, val_generator, test_generator), (train_num, val_num, test_num)
	


def get_img_lab(fpath, classes, im_size):
	"""
	Get the images and their corresponding labels.

	Arguments:
		fpath {str} -- path of the data folder.
		classes {DataFrame} -- DataFrame of all the classes. Each class has a subfolder, where some images are stored. The class subfolder name is in column 'LabelText'. Each class has an integer label in column 'LabelInteger'.
		im_size {int} -- input image shape will be im_size*im_size*C, where `C` is the image channel. Resize all the loaded image to this shape.
	
	Returns:
		{ndarray} -- image array of rank 4. Its shape is N*H*W*C, where `N` is image number, `H` is image height, `W` is image width, `C` is image channel.
		{ndarray} -- one-hot label array of rank 2. Its shape is N*M, where `N` is image number, the length of `M` equals to the number of classes.
	"""
	im_shape = (im_size, im_size)
	imgs = []
	labs = []
	
	for ind in classes.index:
		subfolder = classes.at[ind, 'LabelText']
		lab = classes.at[ind, 'LabelInteger']

		subpath = os.path.join(fpath, subfolder)
		if os.path.isdir(subpath):
			img_names,_ = get_img_files(subpath)

			for img_name in img_names:
				img = cv2.imread(os.path.join(subpath,img_name), cv2.IMREAD_COLOR)[:,:,::-1]
				if img.shape[:2] != im_shape:
					img = resize(img, im_shape)
				imgs.append(img)
				labs.append(lab)
	
	img_arr = np.array(imgs)
	lab_arr = to_categorical(np.array(labs))

	return img_arr, lab_arr


def get_lab_ground_truth(data_path):
	"""
	Get the grond truth of the lab.

	The Dataset structure's detail please refer to the documentation of function `data_generator2`.

	Arguments:
		data_path {str} -- data path.
	
	Return:
		{list} -- each element of the list is a string representing the class name.
		{ndarray} -- each row of the numpy array represents one sample's categorical one-hot label.
	"""
	# list of classes
	lab_df = pd.read_excel(os.path.join(data_path, 'class-labels.xlsx'), index_col=0)

	classes = []
	labs = []
	for ind in lab_df.index:
		subfolder = lab_df.at[ind, 'LabelText']  # class name string
		lab = lab_df.at[ind, 'LabelInteger']  # class integer label

		subpath = os.path.join(data_path, 'test', subfolder)
		if os.path.isdir(subpath):
			img_names,_ = get_img_files(subpath)
			for _ in img_names:
				classes.append(subfolder)
				labs.append(lab)
	
	lab_arr = to_categorical(np.array(labs))

	return classes, lab_arr
















