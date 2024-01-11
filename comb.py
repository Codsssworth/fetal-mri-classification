import tensorflow as tf
from tensorflow import keras
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def lr_schedule(epoch):
    if epoch < 100:
        return 0.1
    elif epoch < 150:
        return 0.01
    else:
        return 0.001

# Define the data directories
train_dir = 'training'
val_dir = 'validation'

# Define the image size and batch size
img_size = (224, 224)
batch_size = 16

# Define the data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Define the feature-based model
inputs = tf.keras.Input(shape=(224, 224, 3))

# Load the VGG16 model with pretrained weights from imagenet
vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)

# Freeze the layers of the VGG16 model
for layer in vgg16_model.layers:
    layer.trainable = False

x = vgg16_model.output
x = tf.keras.layers.Flatten()(x)

# Add a fully connected layer with 1024 neurons
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Add a second fully connected layer with 256 neurons
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Add a final output layer with a sigmoid activation function
outputs = tf.keras.layers.Dense(2, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the callbacks
checkpoint = ModelCheckpoint('combined_checkpoint.h5', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Fit the model
history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=13,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop])

model.save('combine_model.h5')
# Print the accuracy
test_loss, test_acc = model.evaluate(validation_generator)
print('Validation accuracy:', test_acc)



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