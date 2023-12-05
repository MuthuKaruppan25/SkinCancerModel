import keras
from keras.layers import GlobalAveragePooling2D,MaxPooling2D,Conv2D,GlobalMaxPooling2D
from keras.layers import Dense,Input
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,Sequential
from keras.src.callbacks import EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.applications.efficientnet_v2 import EfficientNetV2L
from keras.applications.efficientnet import EfficientNetB2
from keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import EarlyStopping
from keras.layers.experimental import preprocessing
from keras import layers
from optuna import trial
import optuna

import tensorflow as tf


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    fill_mode='wrap',
                                   # Randomly shift images vertically by up to 20% of the height
                                   shear_range=0.2,  # Randomly apply shearing transformations


                                   )

train_generator =train_datagen.flow_from_directory(r"C:\cancer\train",target_size=(224,224),color_mode='rgb',batch_size=32,class_mode='categorical',shuffle=True)

test_generator= train_datagen.flow_from_directory(r"C:\cancer\test",target_size=(224,224),color_mode='rgb',batch_size=32,class_mode='categorical',shuffle=True)


#data agumentation
data_aug=Sequential(
    [
        preprocessing.RandomFlip('horizontal_and_vertical'),
        preprocessing.RandomHeight(0.2),
        preprocessing.RandomWidth(0.2),
        preprocessing.RandomZoom(0.2),
        preprocessing.RandomRotation(0.2 )
    ],name="agumentation"
)



NO_CLASSES = len(train_generator.class_indices.values())


base_model = EfficientNetB2(include_top=False,input_shape=(224,224,3)) #basemodel




base_model.trainable=False
co=0
# don't train the first 19 layers - 0..18
for layer in base_model.layers[200:]:
    print(co)
    layer.trainable = True
    co+=1
base_model.summary()


resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(224, 224),
  layers.Rescaling(1./255)
])


models=keras.Sequential(
    [
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(23, activation='softmax')
    ]
)





early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

models.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

history=models.fit(train_generator,validation_steps=len(test_generator),validation_data=test_generator,steps_per_epoch=len(train_generator),batch_size = 32,epochs = 30,
                  callbacks=[early_stopping])
models.save('main3_' + 'model.h5')
models.save('main3_' + 'model.keras')
models.evaluate(test_generator)




import matplotlib.pyplot as plt


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


plot_hist(history)

