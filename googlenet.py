import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Conv2D, Activation, BatchNormalization, add, GlobalAveragePooling2D, Dense,MaxPooling2D, Flatten, Dropout,Lambda,multiply
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from keras.preprocessing import image
from keras import backend as K
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.applications import VGG16


def create_googlenet_cnn_model(input_shape, num_classes):
    # Input layer
    input = Input(shape=input_shape)

    vgg16_model = VGG16( include_top=False, weights='imagenet', input_tensor=input )

    # Freeze the layers of the VGG16 model
    for layer in vgg16_model.layers:
        layer.trainable = False

    x = vgg16_model.output
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()( x )
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    # First Convolutional block
    x = Conv2D(64, (7, 7), strides=(2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    # Second Convolutional block
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    # Third Convolutional block
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    # Dense layers with Dropout
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)

    # Output layer
    output = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=input, outputs=output)

    return model


# Example usage:
input_shape = (224, 224, 3)
num_classes = 2
train_dir = 'training'

# Define the image size and batch size
img_size = (224, 224)
batch_size = 16

# Define the data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'

)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

model = create_googlenet_cnn_model(input_shape, num_classes)

optimizer = Adam(learning_rate=0.001,beta_1 = 0.9,beta_2 = 0.999,epsilon = 1e-7)
# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define the callbacks
checkpoint = ModelCheckpoint('googlenet_checkpoint.h5', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001)

# Fit the model
history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=13,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop,reduce_lr])

# Print the accuracy
test_loss, test_acc = model.evaluate(validation_generator)
print('Validation accuracy:', test_acc)

model.save('googlenet.h5')



# Accessing the metrics
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plotting the metrics
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'b-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()