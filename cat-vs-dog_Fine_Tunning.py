from keras import applications 
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout,Flatten, Dense

# Path of model weight file
weights_path='../keras/vgg_16_weights.h5' 
top_model_weights_path='bottle_neck_fc_model.h5' 
img_width,img_height=150,150

train_data_dir='data/train'
validation_data_dir='data/validation'
nb_train_samples=2000
nb_validation_samples=800
epochs=10
batch_size=16

# Build VGG16 Model
model = applications.VGG16(weights='imagenet',include_top=False,input_shape=(img_height,img_width,3))

# Set first 25 model to non-trainable
for layer in model.layers[:25]:
    layer.trainable=False

model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(lr=1e-4,momentum=0.9),metrics=['accuracy'])
print(model.summary())

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir, 
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)
