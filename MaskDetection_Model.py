import numpy as np


from tensorflow import keras
from keras import layers
from keras.preprocessing.image import image_dataset_from_directory



#Step 1: Get Training and Test Data for Model from Dataset 



training_dataset = image_dataset_from_directory(
    directory = "DataSet/Train",
    image_size = (48, 48),
    color_mode = "grayscale",
    batch_size = 32,
    shuffle = True,
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
)


test_dataset = image_dataset_from_directory(
    directory = "DataSet/Test",
    image_size = (48, 48),
    color_mode = "grayscale",
    batch_size = 32,
    shuffle = False,
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
)



#Step 2: CNN Model 

#CNN From KERAS
inputs = keras.Input(shape=(48, 48, 1))

#Standardization -> Put image values to [0,1] instead of [0,255]
cnn_component = layers.Rescaling(1./255)(inputs)

#First Layer
cnn_component = layers.Conv2D(filters = 64, kernel_size = 3)(inputs)
cnn_component = layers.BatchNormalization()(cnn_component)
cnn_component = layers.Activation("relu")(cnn_component)
cnn_component = layers.Dropout(0.2)(cnn_component)
cnn_component = layers.MaxPooling2D(pool_size=2)(cnn_component)


#Second Layer
cnn_component = layers.Conv2D(filters = 128, kernel_size = 3)(cnn_component)
cnn_component = layers.BatchNormalization()(cnn_component)
cnn_component = layers.Activation("relu")(cnn_component)
cnn_component = layers.Dropout(0.2)(cnn_component)
cnn_component = layers.MaxPooling2D(pool_size=2)(cnn_component)

#Third Layer
cnn_component = layers.Conv2D(filters = 256, kernel_size = 3)(cnn_component)
cnn_component = layers.BatchNormalization()(cnn_component)
cnn_component = layers.Activation("relu")(cnn_component)
cnn_component = layers.Dropout(0.2)(cnn_component)
cnn_component = layers.MaxPooling2D(pool_size=2)(cnn_component)



#Fully Connected Layers
cnn_component = layers.Flatten()(cnn_component)
cnn_component = layers.Dense(256, activation = "elu")(cnn_component)
outputs = layers.Dense(7, activation = "softmax")(cnn_component)

cnn_model = keras.Model(inputs=inputs, outputs=outputs)


#Detailed Summary on the CNN Model
#Step 3: Compile, Train and Test Model with Accuracy as the primary evaluation metric
cnn_model.summary()

cnn_model.compile(optimizer= keras.optimizers.Adam(learning_rate = 0.0001), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

cnn_model.fit(training_dataset, epochs = 20, batch_size = 32)

testing_loss, testing_accuracy = cnn_model.evaluate(test_dataset)
print(f"test accuracy: {testing_accuracy}")




#NOTE: CURRENT CNN: @ EPOCH 20: 87% TRAIN ACCURACY, LOSS: 0.3661, TEST ACCURACY: 0.572


#Step 4: Save the Model with its current weight values
cnn_model.save("fer_model_v1")
