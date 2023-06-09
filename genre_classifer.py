from sklearn.model_selection import train_test_split
import numpy as np
import json
import tensorflow.keras as keras

DATASET_PATH = "data_10.json"

def load_data(dataset_path):
  with open(dataset_path, "r") as fp:
    data = json.load(fp)

  # convert lists into numpy arrays
  inputs = np.array(data["mfcc"])
  labels = np.array(data["labels"])

if __name__ == "__main__":
  # load data
  inputs, targets = load_data(DATASET_PATH)

  # train/test split
  inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.2)

  # build network architecture
  model = keras.Sequential([
    # input layer
    keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

    # first hidden layer
    keras.layers.Dense(512, activation='relu'),

    # second hidden layer
    keras.layers.Dense(512, activation='relu'),

    # third hidden layer
    keras.layers.Dense(512, activation='relu'),

    # output layer
    keras.layers.Dense(10, activation='softmax')
  ])

  # compile network
  optimizer = keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  # summarize model
  model.summary()

  # train network
  model.fit(inputs_train, targets_train, 
            validation_data=(inputs_test, targets_test),
            epochs=50,
            batch_size=32)
  