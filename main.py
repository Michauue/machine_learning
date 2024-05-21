from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# Definicja generatora obrazów z dodatkowymi argumentami
datagen = ImageDataGenerator(
    rescale=1.0/255,                # Skalowanie wartości pikseli do zakresu [0, 1]
    shear_range=0.2,                # Losowe ścinanie obrazów (pochylenie kształtu obrazu) o maksymalny kąt 20 stopni
    zoom_range=0.2,                 # Losowe powiększanie lub pomniejszanie obrazów do 20% ich oryginalnej wielkości
    horizontal_flip=True,           # Losowe poziome odbicie obrazów
    vertical_flip=True,             # Losowe pionowe odbicie obrazów
    rotation_range=40,              # Losowy obrót obrazów w zakresie -/+ 40 stopni
    width_shift_range=0.2,          # Losowe przesunięcie szerokości obrazów o 20% szerokości
    height_shift_range=0.2,         # Losowe przesunięcie wysokości obrazów o 20% wysokości
    brightness_range=[0.8, 1.2],    # Losowe zmiany jasności obrazów w zakresie od 80% do 120% oryginalnej jasności
    channel_shift_range=0.2         # Losowe zmiany intensywności kanałów o wartości do 20% oryginalnej wartości
)

# Wczytanie i przeskalowanie obrazów z katalogu treningowego
train_generator = datagen.flow_from_directory(
    'CNN_zadanie_images/images/train',         # Ścieżka do katalogu z danymi treningowymi
    target_size=(150, 150), # Rozmiar obrazów (szerokość, wysokość)
    batch_size=32,          # Wielkość partii
    class_mode='binary',    # Tryb klasyfikacji binarnej
    shuffle=True            # Losowość próbek
)

# Wczytanie i przeskalowanie obrazów z katalogu walidacyjnego
validation_generator = datagen.flow_from_directory(
    'CNN_zadanie_images/images/validation',    # Ścieżka do katalogu z danymi walidacyjnymi
    target_size=(150, 150), # Rozmiar obrazów (szerokość, wysokość)
    batch_size=32,          # Wielkość partii
    class_mode='binary',    # Tryb klasyfikacji binarnej
    shuffle=True            # Losowość próbek
)

# Funkcja tworząca model CNN
def make_convnet(input_shape):
    model = Sequential()
    
    # Pierwsza warstwa konwolucyjna
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Druga warstwa konwolucyjna
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Trzecia warstwa konwolucyjna
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Spłaszczenie wyników
    model.add(Flatten())
    
    # Warstwa gęsta z dropoutem
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    # Warstwa wyjściowa
    model.add(Dense(1, activation='sigmoid'))
    
    # Kompilacja modelu
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Parametry wejściowe
input_shape = (150, 150, 3)

# Tworzenie modelu
model = make_convnet(input_shape)

# Przygotowanie do zapisu najlepszego modelu na podstawie walidacji
checkpoint_path = "best_model.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')

# Trenowanie modelu
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50,
    callbacks=[checkpoint]
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