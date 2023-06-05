import tensorflow as tf
from random import random
import numpy as np
from sklearn.model_selection import train_test_split

def generate_dataset(num_samples, test_size):
  x = np.array([[random() / 2 for _ in range(2)] for _ in range(num_samples)])
  y = np.array([[i[0] + i[1]] for i in x])

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
  return x_train, x_test, y_train, y_test

if __name__ == "__main__":
  x_train, x_test, y_train, y_test = generate_dataset(2000, 0.2)

  model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
    tf.keras.layers.Dense(1, activation="sigmoid")
  ])

  optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.1)
  model.compile(optimizer=optimizer, loss="mse")

  model.fit(x_train, y_train, epochs=100)

  print("\nModel evaluation:")
  model.evaluate(x_test)

  # make prediction
  data = np.array([[0.1, 0.2], [0.2, 0.2]])
  preds = model.predict(data)

  print("\nPredictions:")
  for d, p in zip(data, preds):
    print("{} + {} = {}".format(d[0], d[1], p[0]))