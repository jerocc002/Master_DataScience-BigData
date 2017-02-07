# -*- coding: utf-8 -*-
# Ejercicios de repaso de NumPy
# - Siguen la misma estructura que los ejemplos de código que tenemos
# disponibles en las transparencias.
# - Las soluciones se pueden enviar usando este mismo archivo, ya que es un
# archivo python listo para ser ejecutado en nuestro entorno.
# - Os recuerdo mi correo para cualquier duda y para el envío de las
# soluciones:
# gmunoz4@us.es

# Importación de la librería numpy
import numpy as np

##
# Ejercicio 1
##
# a) Crea un objeto ndarray con números del 0 al 9. El tipo de los datos es
# tipo entero de 32 de bits (int32).
data = np.arange(10,dtype='int32')
data

# b) Una vez tenemos nuestro ndarray, le cambiamos el tipo a float64.
data_float = data.astype(np.float64)
data_float

# c) Ahora, necesitamos crear un ndarray con todos sus elementos igual a 1 del
# mismo tamaño que el array creado anteriormente.
data_ones = np.ones_like(data)
data_ones

# d) La última operación es sumar los arrays creados.
data+data_float+data_ones

##
# Ejercicio 2
##
# a) Creamos un objeto ndarray con números del 0 al 20. El tipo de los datos
# es tipo float64.
d = np.arange(21,dtype='float64')
d

# b) Necesitamos aplicar las siguientes funciones unarias a nuestro array:
#   - Calcular la raíz cuadrada de cada elemento
#   - Calcular el cuadrado de cada elemento
#   - Preguntaremos si los elementos de nuestro array son números o no
d**0.5
d*2
np.isnan(d)

# c) Necesitamos aplicar las siguientes funciones binarias a nuestro array.
# Para ello crearemos otro objeto ndarray con sus elementos igual a 1 y del
# mismo tamaño que el anterior:
#   - Sumaremos los elementos de ambos arrays
#   - Multiplicaremos los elementos de ambos arrays
#   - Sumaremos uno al array con todos sus elementos igual a 1.
#   - Elevaremos los elementos del primer array a los elementos de este nuevo
#     array.
d1 = np.ones_like(d)
d+d1
d*d1
d1+1
d**(d1+1)

##
# Ejercicio 3
##
# a) Creamos un objeto ndarray con números del 0 al 10. El tipo de los datos
# es tipo float64.
a = np.arange(11,dtype='float64')
a

# b) Calcularemos la media de los elementos de nuestro array de dos maneras:
#    - Usando el método que nos ofrece nuestro array
#    - Usando el método que nos ofrece la librería numpy
sum(a)/len(a)
np.mean(a)

# c) Calcularemos ahora la suma acumulada de todos los elementos de nuestro
# array.
np.cumsum(a)

# d) Crearemos a continuación un array con los valores:
values = ['Python', 'R', 'datos', 'R', 'ciencia', 'libreria', 'Python']
av = np.array(values)
av

#   - Aplicar una función para eliminar elementos duplicados
av = np.unique(av)
av

#   - Comprobar si los elementos de nuestro array existen en este otro array:
new_values = ['Python', 'R']
nv = np.array(new_values)
nv
np.in1d(av,new_values,assume_unique=True)

##
# Ejercicio 4
##
# a) En este ejercicio trabajaremos con las distintas formas de indexación que
# numpy nos ofrece. Comenzamos creando un array con elementos del 0 al 8.
b = np.arange(9)

# b) Mostraremos el valor del elemento en la posición 3. Seguidamente
# modificaremos los valores desde la posición 4 a la 6 con el valor 20.
b[2]
b[3:6] = 20
b

# c) Modificamos nuestro array para que sea una matriz 3 x 3. Mostramos los
# valores accediendo a la segunda fila y hasta la segunda columna.
b = np.reshape(b, (3, 3))
b[1:,:2]

# d) En este apartado haremos uso del array:
science = np.array(['Python', 'R', 'datos', 'R', 'ciencia', 'libreria',
                    'Python', 'Python', 'R'])
# Mostramos los valores de nuestro array que en 'values' son igual a 'Python'.
# El array a usar debe ser el del apartado a).
np.where(science == 'Python')

