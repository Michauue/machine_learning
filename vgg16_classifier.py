from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# VGG16 jako klasyfikator i ekstraktor cech 

# Wczytanie i przeskalowanie obrazów z katalogu treningowego
datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.2
)

train_generator = datagen.flow_from_directory(
    'CNN_zadanie_images/images/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    'CNN_zadanie_images/images/validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

xepochs = 5

# Definicja modelu VGG-16
def create_vgg16(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Tworzenie modelu VGG16
input_shape = (150, 150, 3)
vgg16_model = create_vgg16(input_shape)

# Trening modelu VGG16
checkpoint_path_vgg16 = "best_model_vgg16.h5"
checkpoint_vgg16 = ModelCheckpoint(checkpoint_path_vgg16, monitor='val_accuracy', save_best_only=True, mode='max')

history_vgg16 = vgg16_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=xepochs,
    callbacks=[checkpoint_vgg16]
)

# Ładowanie wag najlepszego modelu VGG16
vgg16_model.load_weights(checkpoint_path_vgg16)

# Ocena modelu VGG16
loss_vgg16, accuracy_vgg16 = vgg16_model.evaluate(validation_generator)
print(f'VGG16 model loss: {loss_vgg16}')
print(f'VGG16 model accuracy: {accuracy_vgg16}')

# Funkcja do ekstrakcji cech z obrazów przy użyciu VGG-16
def extract_features(model, generator, steps):
    features = []
    labels = []
    for i in range(steps):
        x_batch, y_batch = next(generator)
        features_batch = model.predict(x_batch)
        features.append(features_batch)
        labels.append(y_batch)
    return np.vstack(features), np.hstack(labels)

# Użycie warstw konwolucyjnych VGG-16 do ekstrakcji cech
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

# Ekstrakcja cech z danych treningowych
train_features, train_labels = extract_features(feature_extractor, train_generator, train_generator.samples // train_generator.batch_size)

# Ekstrakcja cech z danych walidacyjnych
validation_features, validation_labels = extract_features(feature_extractor, validation_generator, validation_generator.samples // validation_generator.batch_size)

# Klasyfikator regresji logistycznej
classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
classifier.fit(train_features.reshape(train_features.shape[0], -1), train_labels)

# Ocena klasyfikatora na danych walidacyjnych
val_predictions = classifier.predict(validation_features.reshape(validation_features.shape[0], -1))
accuracy = accuracy_score(validation_labels, val_predictions)
print(f'Feature extractor + Logistic Regression accuracy: {accuracy}')
