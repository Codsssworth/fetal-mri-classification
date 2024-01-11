import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,Input,add,GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler
from keras.preprocessing import image
from keras.datasets import cifar100
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.regularizers import l2
import pickle

num_classes=2
input = Input(shape=(224, 224, 3))
base_model = Xception(include_top=False, weights='imagenet', pooling='avg')(input)

base_model.trainable = False

# Batch normalization layer
x = BatchNormalization()(base_model)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x) # Adding another Dropout layer
x = Flatten()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.5))(x)
x = Dropout(0.5)(x) # Adding another Dropout layer
outputs = Dense(2, activation='softmax')(x)


# Compile the model
model = Model(inputs=input, outputs=outputs)

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
    subset='validation'
)


optimizer = Adam(learning_rate=0.001,beta_1 = 0.9,beta_2 = 0.999,epsilon = 1e-7)
# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define the callbacks
checkpoint = ModelCheckpoint('xceptionbased_checkpoint.h5', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001)

# Fit the model
history = model.fit(
    train_generator,
    epochs=100,
    steps_per_epoch=13,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop,reduce_lr])

# Print the accuracy
test_loss, test_acc = model.evaluate(validation_generator)
print('Validation accuracy:', test_acc)

model.save('xception.h5')

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