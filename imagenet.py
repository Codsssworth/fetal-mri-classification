import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout,Activation
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping


def create_cnn_model(input_shape, num_classes):
    # Input layer
    input = Input(shape=input_shape)

    # Pre-trained base model with ImageNet weights
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=input)

    # Freeze the base model
    base_model.trainable = False

    # Batch normalization layer
    x = BatchNormalization()(base_model.output)

    # Additional layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    # x = Dense( 32 )( x )
    # x = BatchNormalization()( x )
    # x = Activation( 'relu' )( x )
    # x = Dropout( 0.5 )( x )
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense( 512, activation='relu', kernel_regularizer=l2( 0.01 ) )( x )
    x = BatchNormalization()( x )
    x = Dropout( 0.5 )( x )
    x = BatchNormalization()( x )

    output = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=input, outputs=output)

    return model

# Example usage:
input_shape = (224, 224, 3)
img_size=(224,224)
num_classes = 2
batch_size = 16



train_dir = 'training'


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

model = create_cnn_model(input_shape, num_classes)

optimizer = Adam(learning_rate=0.001,beta_1 = 0.9,beta_2 = 0.999,epsilon = 1e-7)
# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define the callbacks
checkpoint = ModelCheckpoint('imagenet_checkpoint.h5', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001)

# Fit the model
history = model.fit(
    train_generator,
    epochs=20,
    steps_per_epoch=13,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop,reduce_lr])

# Print the accuracy
test_loss, test_acc = model.evaluate(validation_generator)
print('Validation accuracy:', test_acc)

model.save('imagenet.h5')


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