# Library of models.

from keras.applications import ResNet50 as RN50


def ResNet50(im_size):
	base_model = RN50(
		weights='imagenet',
		include_top=False,
		pooling='avg',
		input_shape=(im_size,im_size,3)
	)
	top_conv_level = 1
	return base_model, top_conv_level








