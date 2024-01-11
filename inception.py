import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dropout, Flatten, Dense, AveragePooling2D, BatchNormalization
from sklearn.utils.class_weight import compute_class_weight
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt

def InceptionV2():
    input_shape = (224, 224, 3)
    inputs = Input( shape=input_shape )

    # Stem
    conv1 = Conv2D( 32, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu' )( inputs )
    conv2 = Conv2D( 32, kernel_size=(3, 3), padding='valid', activation='relu' )( conv1 )
    conv3 = Conv2D( 64, kernel_size=(3, 3), padding='same', activation='relu' )( conv2)
    pool1 = MaxPooling2D( pool_size=(3, 3), strides=(2, 2) )( conv3 )
    batch_norm1 = BatchNormalization()( pool1 )
    conv4 = Conv2D( 80, kernel_size=(1, 1), padding='same', activation='relu' )( batch_norm1 )
    # den1 = Dense( 32, activation='relu' )( conv4 )
    # bn=BatchNormalization()(den1)
    # dp=Dropout(0.5)(bn)
    conv5 = Conv2D( 192, kernel_size=(3, 3), padding='valid', activation='relu' )( conv4 )
    pool2 = MaxPooling2D( pool_size=(3, 3), strides=(2, 2) )( conv5 )

    den1 = Dense( 512,  activation='relu' )( pool2 )
    bn=BatchNormalization()(den1)
    dp=Dropout(0.5)(bn)

    # Inception Blocks
    inception1 = InceptionBlock( dp, [64, 64, 64, 64, 96, 96, 32] )
    inception2 = InceptionBlock( inception1, [64, 64, 96, 64, 96, 96, 64] )
    pool3 = MaxPooling2D( pool_size=(3, 3), strides=(2, 2) )( inception2 )
    batch_norm2 = BatchNormalization()( pool3 )
    inception3 = InceptionBlock( batch_norm2, [192, 192, 192, 192, 192, 192, 192] )

    # Output
    pool4 = AveragePooling2D( pool_size=(7, 7) )( inception3 )
    dropout = Dropout( 0.4 )( pool4 )
    flatten = Flatten()( dropout )
    output = Dense( units=2, activation='sigmoid' )( flatten )

    # Create the model
    model = Model( inputs=inputs, outputs=output )
    return model


def InceptionBlock(x, filters):
    branch1x1 = Conv2D( filters[0], (1, 1), padding='same', activation='relu' )( x )
    branch1x1 = BatchNormalization()( branch1x1 )

    branch3x3 = Conv2D( filters[1], (1, 1), padding='same', activation='relu' )( x )
    branch3x3 = BatchNormalization()( branch3x3 )
    branch3x3 = Conv2D( filters[2], (3, 3), padding='same', activation='relu' )( branch3x3 )
    branch3x3 = BatchNormalization()( branch3x3 )

    branch5x5 = Conv2D( filters[3], (1, 1), padding='same', activation='relu' )( x )
    branch5x5 = BatchNormalization()( branch5x5 )
    branch5x5 = Conv2D( filters[4], (5, 5), padding='same', activation='relu' )( branch5x5 )
    branch5x5 = BatchNormalization()( branch5x5 )

    branch_pool = MaxPooling2D( (3, 3), strides=(1, 1), padding='same' )( x )
    branch_pool = Conv2D( filters[5], (1, 1), padding='same', activation='relu' )( branch_pool )
    branch_pool = BatchNormalization()( branch_pool )

    # Concatenate the output of each branch
    x = concatenate( [branch1x1, branch3x3, branch5x5, branch_pool] )

    return x


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
    subset='validation')


model = InceptionV2()
optimizer = Adam(learning_rate=0.001,beta_1 = 0.9,beta_2 = 0.999,epsilon = 1e-7)
# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define the callbacks
checkpoint = ModelCheckpoint('inception_check.h5', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001)

# Fit the model
history = model.fit(
    train_generator,
    epochs=100,
    steps_per_epoch=13,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop,reduce_lr])


model.save('inceptionv2.h5')



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