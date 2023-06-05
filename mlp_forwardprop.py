import numpy as np
from random import random

class MLP:
  def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2) -> None:
    self.num_inputs = num_inputs
    self.num_hidden = num_hidden
    self.num_outputs = num_outputs

    layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
    
    # initiate random weights
    self.weights = []
    for i in range(len(layers)-1):
      w = np.random.rand(layers[i], layers[i+1])
      self.weights.append(w)

    activations = []
    for i in range(len(layers)):
      a = np.zeros(layers[i])
      activations.append(a)
    self.activations = activations

    derivatives = []
    for i in range(len(layers)-1):
      d = np.zeros((layers[i], layers[i+1]))
      derivatives.append(d)
    self.derivatives = derivatives


  def forward_porpogate(self, inputs):
    activations = inputs
    self.activations[0] = inputs

    for i, w in enumerate(self.weights):
      # calc net inputs
      net_inputs = np.dot(activations, w)

      # calc activations
      activations = self._sigmoid(net_inputs)
      self.activations[i+1] = activations

    return activations
  
  def back_propogate(self, error, verbose=False):
    for i in reversed(range(len(self.derivatives))):
      activations = self.activations[i+1]
      delta = error * self._sigmoid_derivative(activations)
      delta_reshaped = delta.reshape(delta.shape[0], -1).T
      current_activations = self.activations[i]
      current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
      self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
      error = np.dot(delta, self.weights[i].T)
      if verbose:
        print("Derivatives for W{}: {}".format(i, self.derivatives[i]))
    return error
  
  def gradient_descent(self, learning_rate):
    for i in range(len(self.weights)):
      weights = self.weights[i]
      derivatives = self.derivatives[i]
      weights += derivatives * learning_rate


  def train(self, inputs, targets, epochs, learning_rate=0.1):
    
    for i in range(epochs):
      sum_error = 0
      for input, target in zip(inputs, targets):
        # perform forward prop
        output = self.forward_porpogate(input)

        # calc error 
        error = target - output

        # backprop
        self.back_propogate(output, verbose=False)

        # gradient descent
        self.gradient_descent(learning_rate)

        sum_error += self._mse(target, output)
      print(f"Error: {sum_error / len(inputs)} at epoch {i+1}")

  def _mse(self, target, output):
    return np.average((target - output)**2)


  def _sigmoid_derivative(self, x):
    return x * (1.0 - x)
  

  def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  

if __name__ == "__main__":
  # create an MLP
  mlp = MLP(2, [5], 1)

  # create data
  inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
  targets = np.array([[i[0] + i[1]] for i in inputs])

  mlp.train(inputs, targets, 50, 0.1)

  # dummy data
  input = np.array([0.3, 0.1])
  target = np.array([0.4])
  output = mlp.forward_porpogate(input)


  # print resutls
  # print("The network output is: {}".format(inputs))
  # print("The network output is: {}".format(output))
  print("Our network belives that {} + {} is equal to {}".format(input[0], input[1], output[0]))