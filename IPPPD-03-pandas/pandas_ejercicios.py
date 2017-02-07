# -*- coding: utf-8 -*-
# Ejercicios de repaso de Pandas
# - Siguen la misma estructura que los ejemplos de código que tenemos
# disponibles en las transparencias.
# - Las soluciones se pueden enviar usando este mismo archivo, ya que es un
# archivo python listo para ser ejecutado en nuestro entorno.
# - Os recuerdo mi correo para cualquier duda y para el envío de las
# soluciones:
# gmunoz4@us.es

# Importación de la librería numpy
import numpy as np
# Importación de la librería pandas
import pandas as pd
# Importación de los objetos Series y DataFrame
from pandas import Series, DataFrame

##
# Ejercicio 1
##
# a) Crea un objeto Series con los valores:
values = [-1, 2, -3, 5]
Ser1 = Series(values)

# Mostrar el objeto creado y el índice.
Ser1
Ser1.index

# b) Crea un nuevo objeto Series con los mismos valores pero cambiando el
# índice. El nuevo índice tendrá los valores:
index_values = ['a', 'b', 'c', 'd']
Ser2 = Series(values, index=index_values)

# Mostrar el objeto creado y el índice.
Ser2
Ser2.index

# c) Mostrar el valor en la posición 0. Obtener ese mismo valor haciendo uso
# del índice del objeto.
Ser2[0]
Ser2['a']

# d) A continuación creamos un objeto DataFrame usando el diccionario:
data = {'equipo': ['Betis', 'Sevilla', 'Madrid', 'Barcelona', 'Valencia'],
        'titulos': [3, 12, 80, 81, 22],
        'socios': [43800, 35000, 86000, 180000, 39500]}
DF1 = DataFrame(data)

# e) Mostrar el valor de la columna 'equipo' y de la fila 3.
DF1['equipo']
DF1.ix[2]

##
# Ejercicio 2
##
# a) Seguiremos trabajando con el DataFrame creado en el ejercicio 1.
# Cambiaremos el índice de las filas a los valores:
new_index = ['one', 'two', 'three', 'four', 'five']
DF1 = DataFrame(DF1,index=new_index)
DF1

# b) Mostrar nuestro DataFrame eliminando la columna 'socios' y la fila 'one'.
DF1.drop('socios', axis=1).drop('one',axis=0)

# c) Haciendo uso de alguna de las técnicas de indexación vistas, mostrar los
# valores de las columnas 'equipo' y 'titulos' de la fila 'two'.
DF1.ix['two', ['equipo','titulos']]

##
# Ejercicio 3
##
# a) Seguiremos trabajando con el DataFrame creado en el ejercicio 1.
# Mostraremos el resultado de aplicar la siguiente función a las columnas
# 'socios' y 'titulos':
f = lambda x: x.max()

DF1 = DataFrame(data)
DF1.drop('equipo',axis=1).apply(f)

# b) Aplicar la misma función al DataFrame completo.
DF1.apply(f)

# c) Mostrar nuestro DataFrame ordenando el índice tanto de manera ascendente
# como descendente.
DF1.sort_index()
DF1.sort_index(ascending=False)

# d) Seleccionando la columna 'socios', mostrar el resultado de ordenar por
# valores.
DF1['socios'].order()

##
# Ejercicio 4
##
# a) Seguiremos trabajando con el DataFrame creado en el ejercicio 1. Mostrar
# un resumen de la información del mismo.
DF1.describe()

# b) Para aumentar la información de nuestros datos, concatenar a nuestro
# DataFrame un objeto DataFrame con la siguiente información:
new_data = {'equipo': ['Atletico de Madrid'],
            'titulos': [29],
            'socios': [48008]}

DF1.append(DataFrame(new_data),ignore_index=True)

# c) Crear una nueva columna 'posicion' con los siguientes datos:
posicion_values = ['13', np.nan, '3', np.nan, '5', np.nan]

DF2 = DF1.join(DataFrame({'posicion':posicion_values}))
DF2

# d) Mostrar la posicion de los elementos que son NA en nuestro DataFrame.
DF2[DF2['posicion'].isnull()]

# e) Mostrar nuestro DataFrame sin las filas con elementos NA.
DF2[DF2['posicion'].notnull()]

