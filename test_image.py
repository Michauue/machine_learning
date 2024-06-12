import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

model_path = 'final_model.h5'

model = tf.keras.models.load_model(model_path)

image_path = 'CNN_zadanie_images/test/0.png'
# image_path = 'CNN_zadanie_images/images/train/car/0006.jpg'

img_size = (150, 150)

img = image.load_img(image_path, target_size=img_size)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  

prediction = model.predict(img_array)
predicted_value = prediction[0][0]

print(f'Wartość predykcji: {predicted_value}')

threshold = 0.5
if predicted_value > threshold:
    predicted_class = 'Nie Samochód'
else:
    predicted_class = 'Samochód'

plt.imshow(Image.open(image_path))
plt.title(predicted_class)
plt.axis('off')
plt.show()
