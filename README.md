# Classify the z-axis position of bead image

## Introduction

This project classifies the z-axis position of the bead image based on its image feature. The backbone network is ResNet50.

We randomly select 150 images without replacement from each class as training images and 50 images as validation images. All the rest images are testing data.

The model is composed of a pre-trained ResNet50 on ImageNet and two dense layers. The model training has two rounds. In the first round, the ResNet50 backbone is locked, only the two dense layers' weights are tunable. In the second round, the ResNet50 layers are tunable, too. This process is fine-tuning.

The training data is used to update the model's parameters, and the validation data is used to monitor and validate the performance of the model. The testing data do not participate in the training of the model. It is only used to test the performance of the trained model. The testing loss and accuracy is [0.06399855672756713, 0.9783851786125503].

Since the original image is too small to pass through the deep network and maintain enough information, we resize them into 256X256. The pre-trained Resnet only accept RGB image as input, we convert the original gray-scale image into RGB image.

## Functions

File | Function
-----|---------
evaluate.py | Evaluate the performance of the trained model on the testing data.
loader.py | Data loader.
models.py | Model library.
opts.py | Arguments parser of the program.
predict.py | Excute prediction on the testing data.
save_non_multi_gpu_model.py | Transform the saved multi-gpu model into a single-gpu model.
train.py | Do the training.
utils.py | Utilities.
predict_evaluate.ipynb | Analyze and visualize the predictions.
vis_img_test.ipynb | Make the dataset.

## Folders

Folder | Content
-------|--------
analyze_hdf5 | Scripts used to analyze hdf5 format files when I was finding a solution to convert a multi-gpu model into a single-gpu model.
logs | Saved models and prediction results.
source_images | Original multi-frame tif source images.
train_data | Training, validation and testing datasets.
