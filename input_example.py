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

row_to_add = [[genero, edad, provincia, años_experiencia, nivel_estudios]]