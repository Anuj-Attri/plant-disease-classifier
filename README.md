# Plant Disease Classifier

> Training a machine learning model to classify diseases plants.

## Data:

Obtained from Kaggle's [New Plant Disease dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset).

The data has been augmented, as seen in the `plant-disease-classifier.ipynb` file.

## Methods:

The method used is transfer learning on [AlexNet](https://en.wikipedia.org/wiki/AlexNet) and the other on [Inception V3](https://en.wikipedia.org/wiki/Inceptionv3).-

`ImageDataAugmentor` was also used to create an augmented dataset for increased training statistics.

## AlexNet:
> Accuracy on Validation set: `95.064%`
> Multiclass Cross-Entropy Loss: `0.15`
> Optimizer: [RMSprop](https://golden.com/wiki/RMSprop)

`AlexNet Model's training has been inspired by the owner of the data himself`

## InceptionV3:
> Accuracy on Validation set: ~86%
> Optimizer: RMSprop

## The models in the given repository include:
* `inceptionv3_rmsprop.h5`
* `AlexNet_model_rmsprop.h5`

## How to use the models?
> Run the `disease-pred.py` file to test the models on custom images.
