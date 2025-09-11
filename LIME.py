# Install dependencies if not already installed
!pip install tensorflow scikit-image matplotlib lime

# Import libraries
import os
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Print the version of TensorFlow and Keras
print('TensorFlow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)

# Load the pre-trained InceptionV3 model with ImageNet weights
model = InceptionV3(weights='imagenet')

# Path to your image file
img_path = '104.jpg'  # Make sure this file is uploaded in your environment

# Load and preprocess the image
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.inception_v3.preprocess_input(x)

# Predict and decode the results
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

# Display the original image
plt.imshow(img)
plt.axis('off')
plt.title('Original Image')
plt.show()

# LIME explanation
def predict_fn(images):
    images = tf.keras.applications.inception_v3.preprocess_input(images)
    return model.predict(images)

# Initialize the LIME explainer
explainer = lime_image.LimeImageExplainer()

# Explain the image prediction
explanation = explainer.explain_instance(
    np.array(img), 
    predict_fn, 
    top_labels=5, 
    hide_color=0, 
    num_samples=1000
)

# Get the explanation for the top predicted class
top_label = preds[0].argmax()
temp, mask = explanation.get_image_and_mask(
    top_label, 
    positive_only=True, 
    num_features=5, 
    hide_rest=False
)

# Display the explanation with highlighted important regions
plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.axis('off')
plt.title('LIME Explanation')
plt.show()