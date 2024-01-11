import numpy as np
from tensorflow import keras
from keras.layers import Input, Conv2D, Activation, BatchNormalization, Add, Multiply, MaxPooling2D, Dense,Dropout,Flatten
from keras.preprocessing import image
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from keras.regularizers import l2
import matplotlib.pyplot as plt


def attention_block(inputs, ch):
    x = Conv2D(ch, 1, activation='relu', padding='same')(inputs)
    x = Conv2D(ch, 3, activation='relu', padding='same')(x)
    x = Conv2D(ch, 1, padding='same')(x)
    g = Conv2D(ch, 1, padding='same')(inputs)
    g = Conv2D(ch, 1, activation='sigmoid', padding='same')(g)
    x = Multiply()([x, g])
    x = Add()([x, inputs])
    return x

num_classes = 2


inputs = Input(shape=(224, 224, 3))

x = Conv2D(32, 3, padding="same", kernel_regularizer=l2(0.1))(inputs)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = attention_block(x, 32)
x = attention_block(x, 32)
x = attention_block(x, 32)

x = Conv2D(64, 3, strides=2, padding="same", kernel_regularizer=l2(0.1))(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = attention_block(x, 64)
x = attention_block(x, 64)
x = attention_block(x, 64)

x = Conv2D(128, 3, strides=2, padding="same", kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = attention_block(x, 128)
x = attention_block(x, 128)
x = attention_block(x, 128)

x = Conv2D(256, 3, strides=2, padding="same", kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = attention_block(x, 256)
x = attention_block(x, 256)
x = attention_block(x, 256)

x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)  # Add dropout for regularization

x = MaxPooling2D()(x)
x =  Flatten()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
outputs = Dense(num_classes, activation='softmax')(x)

model = keras.models.Model(inputs, outputs)

# Define the data directories
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
checkpoint = ModelCheckpoint('attetionbased_checkpoint.h5', save_best_only=True)
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

model.save('attentionbased.h5')

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