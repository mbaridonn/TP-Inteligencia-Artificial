import numpy as np
import xlrd
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
    self.training_values[self.training_values == 'Catamarca'] = 3
    self.training_values[self.training_values == 'Chaco'] = 4
    self.training_values[self.training_values == 'Chubut'] = 5
    self.training_values[self.training_values == 'Cordoba'] = 6
    self.training_values[self.training_values == 'Corrientes'] = 7
    self.training_values[self.training_values == 'Entre Rios'] = 8
    self.training_values[self.training_values == 'Formosa'] = 9
    self.training_values[self.training_values == 'Jujuy'] = 10
    self.training_values[self.training_values == 'La Pampa'] = 11
    self.training_values[self.training_values == 'La Rioja'] = 12
    self.training_values[self.training_values == 'Mendoza'] = 13
    self.training_values[self.training_values == 'Misiones'] = 14
    self.training_values[self.training_values == 'Neuquen'] = 15
    self.training_values[self.training_values == 'Rio Negro'] = 16
    self.training_values[self.training_values == 'Salta'] = 17
    self.training_values[self.training_values == 'San Juan'] = 18
    self.training_values[self.training_values == 'San Luis'] = 19
    self.training_values[self.training_values == 'Santa Cruz'] = 20
    self.training_values[self.training_values == 'Santa Fe'] = 21
    self.training_values[self.training_values == 'Santiago del Estero'] = 22
    self.training_values[self.training_values == 'Tierra del Fuego'] = 23
    self.training_values[self.training_values == 'Secundario'] = 1
    self.training_values[self.training_values == 'Terciario'] = 2
    self.training_values[self.training_values == 'Universitario'] = 3
    self.training_values[self.training_values == 'Posgrado'] = 4

  def get_training_values(self):
    return self.training_values

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

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  def predict(self):
    print ("Predicted data based on trained weights: ")
    #print ("Input (scaled): \n" + str(xPredicted))
    #print ("Input: \n" + str(hours_studied) + ", " + str(hours_slept))
    #print ("Output: \n" + str(self.forward(xPredicted)))
    print ("Output: \n" + str(self.forward(xPredicted)[0][0] * 110000))



#testing data
genero = 1
edad = 22
provincia = 1
años_experiencia = 2
nivel_estudios = 1

#read excel

# X = (hours studying, hours sleeping), y = score on test
xAll = np.array(([1,28,3,3,3], [2,30,1,5,4], [3,26,2,7,4], [genero, edad, provincia, años_experiencia, nivel_estudios]), dtype=float) # input data
y = np.array(([80000], [120000], [110000]), dtype=float)

# scale units
xAll = xAll/np.amax(xAll, axis=0) # scaling input data
y = y/np.amax(y, axis=0) # scaling output data

# split data
X = np.split(xAll, [3])[0] # training data
xPredicted = np.split(xAll, [3])[1] # testing data



NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
  print ("# " + str(i) + "\n")
  print ("Input (scaled): \n" + str(X))
  print ("Actual Output: \n" + str(y))
  print ("Predicted Output: \n" + str(NN.forward(X)))
  print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  print ("\n")
  NN.train(X, y)

NN.saveWeights()
NN.predict()


excel_reader = Excel_Reader()
excel_reader.apply_conditions()
print("Training Values:")
print(excel_reader.get_training_values())

print("\n xAll:")
print(xAll);
print("\n prueba: ")
prueba = excel_reader.get_training_values()
row_to_add = np.array([genero, edad, provincia, años_experiencia, nivel_estudios])
prueba = np.vstack((prueba,row_to_add))
print(prueba)

'''
xAll = np.array(([1,28,3,3,3], [2,30,1,5,4], [3,26,2,7,4], [genero, edad, provincia, años_experiencia, nivel_estudios]), dtype=float) # input data
'''