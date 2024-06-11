import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

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

xepochs = 50

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

# Funkcja tworząca model CNN
def make_convnet(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Tworzenie modelu
model = make_convnet(input_shape)

# Przygotowanie klasy zapisującej epokę, w której powstał najlepszy model
class EpochSaver(Callback):
    def __init__(self):
        super(EpochSaver, self).__init__()
        self.best_epoch = 0
        self.best_val_accuracy = -np.Inf
    
    def on_epoch_end(self, epoch, logs=None):
        current_val_accuracy = logs.get('val_accuracy')
        if current_val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_val_accuracy
            self.best_epoch = epoch + 1

    def on_train_end(self, logs=None):
        print(f'Best model saved from epoch {self.best_epoch} with val_accuracy {self.best_val_accuracy:.4f}')

# Przygotowanie do zapisu najlepszego modelu na podstawie walidacji
checkpoint_path = "best_model.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')

# Stworzenie instancji EpochSaver do zapisu, z której epoki pochodzi najlepszy model
epoch_saver = EpochSaver()

# Trenowanie modelu
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=xepochs,
    callbacks=[checkpoint, epoch_saver]
)

# Ocena modelu na danych walidacyjnych
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
print(f'Validation loss: {val_loss}')
print(f'Validation accuracy: {val_accuracy}')

# Załadowanie najlepszego modelu
model.load_weights(checkpoint_path)

# Zapis wyuczonych wag modelu
model.save('final_model.h5')

# Ocena modelu
loss, accuracy = model.evaluate(validation_generator)
print(f'Final model loss: {loss}')
print(f'Final model accuracy: {accuracy}')

# Wizualizacja historii treningu
def plot_history(history1, history2, model1_name, model2_name):
    plt.figure(figsize=(12, 5))

    # Wykres dokładności
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['accuracy'], label=f'{model1_name} Training Accuracy')
    plt.plot(history1.history['val_accuracy'], label=f'{model1_name} Validation Accuracy')
    plt.plot(history2.history['accuracy'], label=f'{model2_name} Training Accuracy')
    plt.plot(history2.history['val_accuracy'], label=f'{model2_name} Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Wykres straty
    plt.subplot(1, 2, 2)
    plt.plot(history1.history['loss'], label=f'{model1_name} Training Loss')
    plt.plot(history1.history['val_loss'], label=f'{model1_name} Validation Loss')
    plt.plot(history2.history['loss'], label=f'{model2_name} Training Loss')
    plt.plot(history2.history['val_loss'], label=f'{model2_name} Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Wizualizacja historii treningu dla modelu VGG16 i własnego modelu na jednym wykresie
plot_history(history, history_vgg16, 'My Model', 'VGG16')

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
classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
classifier.fit(train_features.reshape(train_features.shape[0], -1), train_labels)

# Ocena klasyfikatora na danych walidacyjnych
val_predictions = classifier.predict(validation_features.reshape(validation_features.shape[0], -1))
accuracy = accuracy_score(validation_labels, val_predictions)
print(f'Feature extractor + Logistic Regression accuracy: {accuracy}')

# Wizualizacja wyekstrahowanych cech przy użyciu t-SNE
def plot_tsne(features, labels, title):
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))
    for label in np.unique(labels):
        indices = np.where(labels == label)
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=f'Class {label}')

    plt.legend()
    plt.title(title)
    plt.show()

# Wizualizacja cech wyekstrahowanych przez VGG16 z danych treningowych
plot_tsne(train_features.reshape(train_features.shape[0], -1), train_labels, 'VGG16 Features (Train)')

# Wizualizacja cech wyekstrahowanych przez VGG16 z danych walidacyjnych
plot_tsne(validation_features.reshape(validation_features.shape[0], -1), validation_labels, 'VGG16 Features (Validation)')

# Funkcja pozwalająca na wizualizację cech pierwszej warstwy modelów
def visualize_conv_layer(model, layer_name, image):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(image)
    
    num_filters = intermediate_output.shape[-1]
    size = intermediate_output.shape[1]
    display_grid = np.zeros((size, size * num_filters))
    
    for i in range(num_filters):
        x = intermediate_output[0, :, :, i]
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 64
        x += 128
        x = np.clip(x, 0, 255).astype('uint8')
        display_grid[:, i * size : (i + 1) * size] = x
    
    scale = 20. / num_filters
    plt.figure(figsize=(scale * num_filters, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

# Wczytanie przykładowego obrazu
sample_image = next(train_generator)[0][0]
sample_image = np.expand_dims(sample_image, axis=0)

# Wizualizacja cech dla własnego modelu
visualize_conv_layer(model, 'conv2d', sample_image)

# Wizualizacja cech dla modelu VGG16
visualize_conv_layer(vgg16_model, 'block1_conv1', sample_image)

plt.show()
