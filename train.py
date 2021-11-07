# Training.
import os

from keras.models import Model
from keras.layers import Dense
from keras.utils import multi_gpu_model

import models, utils
from opts import opt
from loader import data_generator2
from save_non_multi_gpu_model import save_nmg_model


epochs = opt.epochs
batch_size = opt.batch_size
data_path = opt.data_path
save_logs_path = opt.save_logs_path
num_class = opt.num_class
model_name = opt.model_name
im_size = opt.im_size
multiGPU = opt.multiGPU
gpus = opt.gpus

print("Constructing model ...")
base_model, top_conv_level = getattr(models, model_name)(im_size)
x = base_model.output
# add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- we have 2 classes
predictions = Dense(num_class, activation='softmax', kernel_initializer='he_normal')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
	layer.trainable = False

# multiple GPUs support
if multiGPU:
	parallel_model = multi_gpu_model(model, gpus=gpus)
	print("Training using {} GPUs ...".format(gpus))

# data generator
print("Making training and validation data generator ...")
(train_generator, val_generator, test_generator), (train_num, val_num, test_num) = data_generator2(
	data_path=data_path,
	im_size=im_size,
	batch_size=batch_size
)

# compile the model (should be done *after* setting layers to non-trainable)
print("Compiling model ...")
parallel_model.compile(
	optimizer='rmsprop',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)
model.summary()


# callbacks
callbacklist = utils.get_callbacks(
	save_path=os.path.join(save_logs_path, model_name, 'origin_model'),
	batch_size=batch_size,  # 'TensorBoard' callback requires
	multiGPU=multiGPU,
	callback_cat = [
		'ModelCheckpointBestAcc',
		'CSVLogger',
		'TensorBoard',
		'ReduceLROnPlateau',
		'EarlyStopping',
	]
)

# train the model on the new data for a few epochs
print("Start training ...")
parallel_model.fit_generator(
	generator=train_generator,
	steps_per_epoch=train_num//batch_size,  # Each class has 150 images, batch size is 50, we have 39 classes.
	epochs=epochs,
	validation_data=val_generator,
	validation_steps=val_num//batch_size,  # Each class has 50 images, batch size is 50, we have 39 classes.
	callbacks=callbacklist
)

# train the top few conv blocks, i.e. freeze
# The first few top_conv_level layers and unfreeze the rest. 
# The number of frozen blocks is dependent of base models. 
# See models.py for details. 
for layer in model.layers[:top_conv_level]:
   layer.trainable = False
for layer in model.layers[top_conv_level:]:
   layer.trainable = True

# multiple GPUs support
if multiGPU:
	parallel_model = multi_gpu_model(model, gpus=gpus)
	print("Training using {} GPUs ...".format(gpus))

# recompile the model for these modifications to take effect
print("Compiling model ...")
parallel_model.compile(
	optimizer='rmsprop',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)
model.summary()

# callbacks
callbacklist = utils.get_callbacks(
	save_path=os.path.join(save_logs_path, model_name, 'fine_tune_model'),
	batch_size=batch_size,  # 'TensorBoard' callback requires
	multiGPU=multiGPU,
	callback_cat=[
		'ModelCheckpointBestAcc',
		'CSVLogger',
		'TensorBoard',
		'ReduceLROnPlateau',
		'EarlyStopping',
	]
)

# train the model again
print("Start training ...")
parallel_model.fit_generator(
	generator=train_generator,
	steps_per_epoch=train_num//batch_size,  # Each class has 150 images, batch size is 50, we have 39 classes.
	epochs=epochs,
	validation_data=val_generator,
	validation_steps=val_num//batch_size,  # Each class has 50 images, batch size is 50, we have 39 classes.
	callbacks=callbacklist
)

# convert multi-gpu model to single-gpu model
if multiGPU:
	print("Start converting model ...")
	save_nmg_model()












