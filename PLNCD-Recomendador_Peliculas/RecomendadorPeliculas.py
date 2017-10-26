

##########################################################################
#####  Sistema de recomendación de películas
#####      PLNCD
#####          Jose F Quesada
##########################################################################

##########################################################################
### Paso 1: Leer ficheros con las sinopsis de las películas
##########################################################################

###
### Utilizaremos el dataset disponible en
###
###    http://www.cs.cmu.edu/~ark/personas/
###
### En concreto el fichero
###
###    http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
###
### que una vez descomprimido genera los siguientes ficheros
###
### character.metadata.tsv
### movie.metadata.tsv
### name.clusters.txt
### plot_summaries.txt
### README.txt
### tvtropes.clusters.txt
###
### De esta distribución utilizaremos el fichero
###
###     movie.metadata.tsv
###
### cada línea de este fichero contiene los metadatos básicos de
### una película, separados por un tabulador, como muestra el siguiente
### ejemplo:
###
### 975900	/m/03vyhn	Ghosts of Mars	2001-08-24	14010832	98.0	{"/m/02h40lc": "English
### Language"}	{"/m/09c7w0": "United States of America"}	{"/m/01jfsb": "Thriller", "/m/06n90": "S
### cience Fiction", "/m/03npn": "Horror", "/m/03k9fj": "Adventure", "/m/0fdjb": "Supernatural", "/m/02kdv5l
### ": "Action", "/m/09zvmj": "Space western"}
###
### El primer elemento indica el código de película (975900). Tras un código
### de referencia aparece el título (Ghosts of Mars) y a continuación la fecha.
###
### El segundo fichero que utilizaremos es
###
###     plot_summaries.txt
###
### que contiene la sinopsis o resumen de cada película. Cada línea comienza
### con el código de película, y a continuación el texto correspondiente.
###


def leerPeliculas(maxPeliculas = 0):
    ## Creamos un diccionario inicial para contener todas
    ## las películas que vamos a leer
    ##
    ## Cada entrada de este diccionario será una película indexada
    ## en el diccionario por el código de película

    peliculas = {}

    ## Fase 1: Comenzamos leyendo los metadatos
    ficheroMetadatos = open("MovieSummaries/movie.metadata.tsv", "r")
    contadorMetadatos = 0

    for pelicula in ficheroMetadatos:
        contadorMetadatos += 1

        metadatos = pelicula.split('\t')

        # metadatos[0] -> Código de la película
        codigoPelicula = metadatos[0]
        # metadatos[1] -> Referencia (ignorar)
        # metadatos[2] -> Título
        tituloPelicula = metadatos[2]
        # metadatos[3] -> Fecha
        fechaPelicula = metadatos[3]

        pelicula = {}
        pelicula['codigo'] = codigoPelicula
        pelicula['titulo'] = tituloPelicula
        pelicula['fecha'] = fechaPelicula

        peliculas[codigoPelicula] = pelicula

    ## Fase 2: Leemos las sinopsis de las películas y las
    ## vinculamos a la entrada del diccionario correspondiente

    ficheroSinopsis = open("MovieSummaries/plot_summaries.txt","r")
    contadorSinopsis = 0

    for lineaSinopsis in ficheroSinopsis:
        # print(pelicula)
        contadorSinopsis += 1

        datosSinopsis = lineaSinopsis.split('\t')
        # datosSinopsis[0] -> Código de película
        codigoPelicula = datosSinopsis[0]
        # datosSinopsis[1] -> Sinopsis de la película
        resumenPelicula = datosSinopsis[1]

        pelicula = peliculas.get(codigoPelicula, 0)

        if (pelicula != 0):
            pelicula['resumen'] = resumenPelicula
            peliculas[codigoPelicula] = pelicula

    ## Fase 3: Creamos un nuevo diccionario que solo tenga las películas
    ## para cuyos metadatos hayamos encontrado un resumen

    peliculasCompletas = {}
    contadorCompletas = 0
    for peliculaCodigo in peliculas:
        pelicula = peliculas[peliculaCodigo]

        resumen = pelicula.get('resumen',0)
        if (resumen != 0):
            contadorCompletas += 1
            peliculasCompletas[peliculaCodigo] = pelicula

            if ((maxPeliculas > 0) & (contadorCompletas >= maxPeliculas)):
                break;

    print("Total Metadatos = ", contadorMetadatos)
    print("Total Sinopsis  = ", contadorSinopsis)
    print("Total Peliculas = ", contadorCompletas)

    return peliculasCompletas



##########################################################################
### Paso 2: Preprocesado y limpieza de los resúmenes de las películas
##########################################################################

import nltk

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")

def obtenerNombresPropios(nombres, texto):
    # Recorremos todas las oraciones de un texto (resumen de una película)

    for frase in nltk.sent_tokenize(texto):
        #
        # nltk.word_tokenize devuelve la lista de palabras que forman
        #    la frase (tokenización)
        #
        # nltk.pos_tag devuelve el part of speech (categoría) correspondiente
        #    a la palabra introducida
        #
        # nltk.ne_chunk devuelve la etiqueta correspondiente al part of
        #    speech
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(frase))):
            try:
                if chunk.label() == 'PERSON':
                    for c in chunk.leaves():
                        if str(c[0].lower()) not in nombres:
                            nombres.append(str(c[0]).lower())
            except AttributeError:
                pass
    return nombres

def preprocesarPeliculas(peliculas):
    print("Preprocesando películas")
    nombresPropios = []

    for elemento in peliculas:
        print("Preproceso: ",elemento)

        pelicula = peliculas[elemento]

        ## Eliminación de signos de puntuación usando tokenizer
        resumen = pelicula['resumen']
        texto = ' '.join(tokenizer.tokenize(resumen))
        pelicula['texto'] = texto

        nombresPropios = obtenerNombresPropios(nombresPropios, texto)

    ignoraPalabras = stopWords
    ignoraPalabras.union(nombresPropios)

    palabras = [[]]
    for elemento in peliculas:
        pelicula = peliculas[elemento]

        texto = pelicula['texto']
        textoPreprocesado = []
        for palabra in tokenizer.tokenize(texto):
            textoPreprocesado.append(stemmer.stem(palabra.lower()))
            if (palabra.lower() not in ignoraPalabras):
                palabras.append([(stemmer.stem(palabra.lower()))])

        pelicula['texto'] = ' '.join(textoPreprocesado)
        
    return palabras

##########################################################################
### Paso 3: Creación de la colección de textos
##########################################################################

from gensim import corpora, models, similarities
    
def crearColeccionTextos(peliculas):
    print("Creando colección global de resúmenes")
    textos = []
    
    for elemento in peliculas:
        pelicula = peliculas[elemento]
        texto = pelicula['texto']
        lista = texto.split(' ')

        textos.append(lista)

    return textos



##########################################################################
### Paso 4: Creación del diccionario de palabras
##########################################################################
###
### El diccionario está formado por la concatenación de todas las
### palabras que aparecen en alguna sinopsis (modo texto) de alguna
### de las peliculas
###
### Básicamente esta función mapea cada palabra única con su identificador
###
### Es decir, si tenemos N palabras, lo que conseguiremos al final
### es que cada película sea representada mediante un vector en un
### espacio de N dimensiones

def crearDiccionario(textos):
    print("Creación del diccionario global")
    return corpora.Dictionary(textos)


##########################################################################
### Paso 5: Creación del corpus de resúmenes preprocesados
##########################################################################
###
### Crearemos un corpus con la colección de todos los resúmenes
### previamente pre-procesados y transformados usando el diccionario
###

def crearCorpus(diccionario):
    print("Creación del corpus global con los resúmenes de todas las películas")
    return [diccionario.doc2bow(texto) for texto in textos]

'''
peliculas   = leerPeliculas(10)
palabras    = preprocesarPeliculas(peliculas)
textos      = crearColeccionTextos(peliculas)
diccionario = crearDiccionario(textos)
corpus      = crearCorpus(diccionario)
'''

### En este momento podemos revisar el contenido de la información
### obtenida.

### Consideremos por ejemplo la película "Mary Poppins" cuyo
### código en el Data Set es 77856

### >>> peliculas['77856']['titulo']
### 'Mary Poppins'
### >>> peliculas['77856']['fecha']
### '1964-08-27'
### >>> peliculas['77856']['resumen'][:200]
###'The film opens with Mary Poppins  perched in a cloud high above London in spring 1910."... It\'s grand to be an Englishman in 1910 / King Edward\'s on the throne; it\'s the age of men! ..." George Banks\''
### >>> peliculas['77856']['texto'][:200]
### 'the film open with mari poppin perch in a cloud high abov london in spring 1910 it s grand to be an englishman in 1910 king edward s on the throne it s the age of men georg bank open song the life i l'

### Esta película ocupa el índice 8 (novena película) entre las que hemos leído
###
### Esto nos indica que la entrada 8 de corpus contiene 554 tokens (palabras distintas)
###
### >>> len(corpus[8])
### 554
###
### Si observamos las primeras 20 entradas de corpus[8], podemos determinar
### que la palabra de índice 0 aparece 1 vez en el texto resumen de esta
### película, mientras que la palabra 2 aparece 117 veces
###
### >>> corpus[8][:20]
### [(0, 1), (1, 18), (2, 117), (4, 1), (5, 36), (8, 1), (11, 12), (12, 42), (14, 16), (15, 4), (21, 44), (22, 1), (23, 7), (25, 2), (31, 1), (34, 14), (35, 1), (37, 1), (39, 1), (53, 4)]
###
### Para ver la palabra exacta a la que se refiere cada índice podemos
### utilizar el diccionario
### >>> diccionario[0]
### 'set'
### >>> diccionario[2]
### 'the'
### >>> diccionario[34]
### 'with'

### La siguiente instrucción nos permite obtener la lista de tuplas
### con la palabra, su índice y el número de repeticiones asociadas
### a esta película
###
### >>> [(diccionario[n],n,m) for (n,m) in corpus[8]]
###
### >>> diccionario[1028]
### 'supercalifragilisticexpialidoci'

##########################################################################
### Paso 6: Creación del modelo tf-idf
##########################################################################

def crearTfIdf(corpus):
    print("Creación del Modelo Espacio-Vector Tf-Idf")
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf

'''
peliculas   = leerPeliculas(50)
palabras    = preprocesarPeliculas(peliculas)
textos      = crearColeccionTextos(peliculas)
diccionario = crearDiccionario(textos)
corpus      = crearCorpus(diccionario)
pel_tfidf   = crearTfIdf(corpus)



### >>> print(corpus[8][:20])
### [(0, 1), (1, 18), (2, 117), (4, 1), (5, 36), (8, 1), (11, 12), (12, 42), (14, 16), (15, 4), (21, 44), (22, 1), (23, 7), (25, 2), (31, 1), (34, 14), (35, 1), (37, 1), (39, 1), (53, 4)]
### >>> print(pel_tfidf[8][:20])
### [(0, 0.00762910434797757), (1, 0.025451519972774662), (4, 0.010198378012720392), (8, 0.00762910434797757), (11, 0.008011540113419483), (14, 0.01068205348455931), (15, 0.005655893327283258), (22, 0.00762910434797757), (23, 0.015820766458248765), (25, 0.020396756025440783), (31, 0.005806175672270604), (34, 0.009346796798989396), (35, 0.00762910434797757), (37, 0.00762910434797757), (39, 0.00762910434797757), (53, 0.012947608030111121), (57, 0.005806175672270604), (60, 0.03051641739191028), (61, 0.0323690200752778), (65, 0.017568809361799154)]
'''


##########################################################################
### Paso 7: Creación del modelo LSA (Latent Semantic Analysis)
##########################################################################

import gensim
import numpy as np

### Valores clave para controlar el proceso
TOTAL_TOPICOS_LSA = 50
UMBRAL_SIMILITUD = 0.7

def crearLSA(corpus,pel_tfidf):
    print("Creación del modelo LSA: Latent Semantic Analysis")
    numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms = 50000)
    svd = np.linalg.svd(numpy_matrix, full_matrices=False, compute_uv = False)

    lsi = models.LsiModel(pel_tfidf, id2word=diccionario, num_topics=TOTAL_TOPICOS_LSA)

    indice = similarities.MatrixSimilarity(lsi[pel_tfidf])

    return (lsi,indice)

def crearCodigosPeliculas(peliculas):
    codigosPeliculas = []
    for i, elemento in enumerate(peliculas):
        pelicula = peliculas[elemento]
        codigosPeliculas.append(pelicula['codigo'])
    return codigosPeliculas

def crearModeloSimilitud(peliculas, pel_tfidf,lsi,indice,salida=None):
    codigosPeliculas = crearCodigosPeliculas(peliculas)
    print("Creando enlaces de similitud entre películas")
    if (salida != None):
        print("Generando salida en fichero ",salida)
        ficheroSalida = open(salida,"w")
        
    for i, doc in enumerate(pel_tfidf):
        print("============================")
        peliculaI = peliculas[codigosPeliculas[i]]
        print("Pelicula I = ",i,"  ",peliculaI['codigo'],"  ",peliculaI['titulo'])

        if (salida != None):
            ficheroSalida.write("============================")
            ficheroSalida.write("\n")
            ficheroSalida.write("Pelicula I = " + peliculaI['codigo'] + "  " + peliculaI['titulo'])
            ficheroSalida.write("\n")
            
        vec_lsi = lsi[doc]
        #print(vec_lsi)
        indice_similitud = indice[vec_lsi]
        similares = []
        for j, elemento in enumerate(peliculas):
            s = indice_similitud[j]
            if (s > UMBRAL_SIMILITUD) & (i != j):
                peliculaJ = peliculas[codigosPeliculas[j]]
                similares.append((codigosPeliculas[j], s))
                
                print("   Similitud: ",s,"   ==> Pelicula J = ",j,"  ",peliculaJ['codigo'],"  ",peliculaJ['titulo'])
                if (salida != None):
                    ficheroSalida.write("   Similitud: " + str(s) + "   ==> Pelicula J = " + peliculaJ['codigo'] + "  " + peliculaJ['titulo'])
                    ficheroSalida.write("\n")
                    
            similares = sorted(similares, key=lambda item: -item[1])

            peliculaI['similares'] = similares
            peliculaI['totalSimilares'] = len(similares)

    if (salida != None):
        ficheroSalida.close()
    
peliculas   = leerPeliculas(300)
palabras    = preprocesarPeliculas(peliculas)
textos      = crearColeccionTextos(peliculas)
diccionario = crearDiccionario(textos)
corpus      = crearCorpus(diccionario)
pel_tfidf   = crearTfIdf(corpus)
(lsi,indice)= crearLSA(corpus,pel_tfidf)
# crearModeloSimilitud(peliculas,pel_tfidf,lsi,indice,"peliculasSimilares.txt")
crearModeloSimilitud(peliculas,pel_tfidf,lsi,indice)
