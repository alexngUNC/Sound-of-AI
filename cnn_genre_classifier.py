import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

DATA_PATH = 'data.json'

def load_data(data_path):
  """
  Loads the training dataset from the json file.

    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
  """

  with open(data_path, "r") as fp:
    data = json.load(fp)

  X = np.array(data['mfcc'])
  y = np.array(data['labels'])
  return X, y


def prepare_datasets(test_size, validation_size):
  # load data
  X, y = load_data(DATA_PATH)

  # create train/test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

  # create train/val split
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size)

  # convert to 4D array for each sample (num_samples, 130, 13, 1) 
  return X_train[..., np.newaxis], X_val[..., np.newaxis], X_test[..., np.newaxis], y_train, y_val, y_test


def build_model(input_shape):
  # create model
  model = keras.Sequential()

  # 1st conv layer
  model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
  model.add(keras.layers.BatchNormalization())

  # 2nd conv layer
  model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
  model.add(keras.layers.BatchNormalization())

  # 3rd conv layer
  model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
  model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
  model.add(keras.layers.BatchNormalization())

  # flatten output and feed into dense layer
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(64, activation='relu'))
  model.add(keras.layers.Dropout(0.3))

  # output layer
  model.add(keras.layers.Dense(10, activation='softmax'))

  return model


def predict(model, X, y):
  # add extra first dim
  X = X[np.newaxis, ...]
  pred = model.predict(X)
  # pred = [[0.1, 0.2, ...]]
  # extract index with max value
  pred_index = np.argmax(pred, axis=1) # e.g. [4]
  print(f'Expected index: {y} | Predicted index: {pred_index}')


if __name__ == '__main__':
  # create train, val, and test sets
  X_train, X_val, X_test, y_train, y_val, y_test = prepare_datasets(test_size=0.25, validation_size=0.2)

  # build the CNN net
  input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
  model = build_model(input_shape)

  # compile the network
  optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  # train the CNN
  model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=30)

  # evaluate CNN on test set
  test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
  print('Test accuracy: {}'.format(test_accuracy))

  # make predictions on a sample
  X = X_test[100]
  y = y_test[100]
  predict(model, X, y)