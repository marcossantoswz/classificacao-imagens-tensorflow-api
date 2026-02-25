import tensorflow as tf
from tensorflow.keras import layers, models

IMAGE_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 8

train_data = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

test_data = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

print(train_data.class_names)

model = models.Sequential([
    # normaliza TODAS as imagens (treino, validação, teste)
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(200, 200, 3)),

    # Extrai padrões básicos
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    # Extrai padrões mais complexos
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    # Prepara para decisão
    layers.Flatten(),

    # Camada de decisão
    layers.Dense(128, activation='relu'),

    # 3 classes
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

model.save("model.keras")
