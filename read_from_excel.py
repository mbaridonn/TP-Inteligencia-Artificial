import numpy as np
import xlrd
import pandas

df = pandas.read_excel('sample.xlsx')

number_of_rows = df.shape[0]
number_of_columns = df.shape[1]

training_values = df.values

print(training_values)

training_values[training_values == 'Hombre'] = 1
training_values[training_values == 'Mujer'] = 2
training_values[training_values == 'Otros'] = 3

training_values[training_values == 'CABA'] = 1
training_values[training_values == 'GBA'] = 2
training_values[training_values == 'Catamarca'] = 3
training_values[training_values == 'Chaco'] = 4
training_values[training_values == 'Chubut'] = 5
training_values[training_values == 'Cordoba'] = 6
training_values[training_values == 'Corrientes'] = 7
training_values[training_values == 'Entre Rios'] = 8
training_values[training_values == 'Formosa'] = 9
training_values[training_values == 'Jujuy'] = 10
training_values[training_values == 'La Pampa'] = 11
training_values[training_values == 'La Rioja'] = 12
training_values[training_values == 'Mendoza'] = 13
training_values[training_values == 'Misiones'] = 14
training_values[training_values == 'Neuquen'] = 15
training_values[training_values == 'Rio Negro'] = 16
training_values[training_values == 'Salta'] = 17
training_values[training_values == 'San Juan'] = 18
training_values[training_values == 'San Luis'] = 19
training_values[training_values == 'Santa Cruz'] = 20
training_values[training_values == 'Santa Fe'] = 21
training_values[training_values == 'Santiago del Estero'] = 22
training_values[training_values == 'Tierra del Fuego'] = 23

training_values[training_values == 'Secundario'] = 1
training_values[training_values == 'Terciario'] = 2
training_values[training_values == 'Universitario'] = 3
training_values[training_values == 'Posgrado'] = 4

print(training_values)