from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import json


def predictor(model, image_path):

    new_img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    with open('classes.json') as f:
        classes = json.load(f)

    print("True Label:", image_path.replace("../input/new-plant-diseases-dataset/test/test/", ""))
    prediction = model.predict(img)

    decoded = prediction.flatten()
    j = decoded.max()
    for index, item in enumerate(decoded):
        if item == j:
            class_name = classes[index]

    print("Predicted Label:", class_name)
    # #ploting image with predicted class name
    # plt.figure(figsize = (4,4))
    # plt.imshow(new_img)
    # plt.axis('off')
    # plt.title(class_name)
    # plt.show()


def worker():

    image_path = input("Enter the path to the image: ")
    model = load_model("Models/inceptionV3_rmsprop_augmented.h5")
    predictor(model, image_path)


worker()

