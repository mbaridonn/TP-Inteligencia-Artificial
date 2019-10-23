import numpy as np
import pandas

class Excel_Reader(object):
  def __init__(self):
    df = pandas.read_excel('sample.xlsx')
    self.training_values = df.values

  def apply_conditions(self):
    self.training_values[self.training_values == 'Hombre'] = 1
    self.training_values[self.training_values == 'Mujer'] = 2
    self.training_values[self.training_values == 'Otros'] = 3
    self.training_values[self.training_values == 'CABA'] = 1
    self.training_values[self.training_values == 'GBA'] = 2
    self.training_values[self.training_values == 'Cordoba'] = 3
    self.training_values[self.training_values == 'Entre Rios'] = 4
    self.training_values[self.training_values == 'Mendoza'] = 5
    self.training_values[self.training_values == 'Neuquen'] = 6
    self.training_values[self.training_values == 'Secundario'] = 1
    self.training_values[self.training_values == 'Terciario'] = 2
    self.training_values[self.training_values == 'Universitario'] = 3

  def get_xAll(self):
    return np.delete(self.training_values, 5, 1)
  
  def get_y(self):
    return self.training_values[:, 5]

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 5
    self.hiddenSize = 3
    self.outputSize = 1

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (5x3) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propagate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def predict(self):
    print ("Sueldo predecido en base a los casos de entrenamiento:")
    print (str(int(self.forward(xPredicted) * sueldoMasAlto)) + " pesos")

class Main(object):
  def initialize_parameters(self):
    print("Bienvenido al predictor de sueldos \n")
    print("Para cada opción, ingrese el número correspondiente \n")

    print("Ingrese su genero:")
    print("1 -  Hombre")
    print("2 -  Mujer")
    print("3 -  Otros \n")
    genero = int(input())

    print("\nIngrese su edad:")
    edad = int(input())

    print("\nIngrese su provincia:")
    print("1 - CABA")
    print("2 - GBA")
    print("3 - Cordoba")
    print("4 - Entre Rios")
    print("5 - Mendoza")
    print("6 - Neuquen")
    provincia = int(input())

    print("\nIngrese la cantidad de años de experiencia en IT:")
    años_experiencia = int(input())

    print("\nIngrese su  nivel de estudios:")
    print("1 - Secundario")
    print("2 - Terciario")
    print("3 - Universitario")
    nivel_estudios = int(input())

    return [[genero, edad, provincia, años_experiencia, nivel_estudios]]

main = Main()
row_to_add = main.initialize_parameters()

#read excel
excel_reader = Excel_Reader()
excel_reader.apply_conditions()

xAll = np.array((np.append(excel_reader.get_xAll(), row_to_add, axis=0)), dtype=float)

rows_number = excel_reader.get_xAll().shape[0]
y = np.array((excel_reader.get_y().reshape(rows_number,1)), dtype=float)

sueldoMasAlto = np.amax(y, axis=0)

# scale units
xAll = xAll/np.amax(xAll, axis=0) # scaling input data
y = y/np.amax(y, axis=0) # scaling output data

# split data
X = np.split(xAll, [rows_number])[0] # training data
xPredicted = np.split(xAll, [rows_number])[1] # testing data

unscale = lambda x: x * sueldoMasAlto

NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
  print ("# " + str(i) + "\n")
  print ("Output real: \n" + str(unscale(y)))
  output = np.trunc(unscale(NN.forward(X)))
  print ("Output predecido: \n" + str(output))
  print ("Error: " + str(np.mean(np.square(y - NN.forward(X)))))
  print ("\n")
  NN.train(X, y)

NN.predict()