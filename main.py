from keras.preprocessing.image import ImageDataGenerator

# Definicja generatora obrazów
datagen = ImageDataGenerator(
    rescale=1.0/255,        # Skalowanie wartości pikseli do zakresu [0, 1]
    shear_range=0.2,        # Losowe ścinanie obrazów
    zoom_range=0.2,         # Losowe powiększanie obrazów
    horizontal_flip=True    # Losowe poziome odbicie obrazów
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