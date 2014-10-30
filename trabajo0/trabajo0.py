''' filename : code-T0.py

Description : Codigo de los ejercicios del Trabajo 0.

Usage : python code-T0.py 

Written by : Francisco Javier Moreno Vega. (jmorenov@correo.ugr.es) '''

import cv2
import sys

def leeimagen(filename, flagColor):
	""" 1) Funcion que lee una imagen en nivel de grises (cv2.CV_LOAD_IMAGE_GRAYSCALE)
			o en color (cv2.CV_LOAD_IMAGE_COLOR). """
	return cv2.imread(filename, flagColor) # Lee la imagen con nombre filename y como tipo flagColor.

def pintaI(im):
	""" 2) Funcion que visualiza una imagen. """
	cv2.namedWindow('Imagen')        # Crea una ventana para visualizar la imagen.
	cv2.imshow('Imagen',im)          # Coloca la imagen en la ventana creada.
	cv2.waitKey(0)                   # Espera a la pulsacion de una tecla para continuar con la ejecucion del programa.
	cv2.destroyAllWindows()					 # Borra las ventana creadas.
	return

def pintaMI(vim):
	""" 3) Funcion que visualiza varias imagenes a la vez. """
	image = vim[0]										# Inicializa la imagen final con la primera imagen de la lista.
	for i in range(1, len(vim)):			# Recorre la lista de imagenes.
		image += vim[i]									# Suma las imagenes una a una para generar la imagen final, 
																		# 	si las imagenes fueran de tipos distintos (CV_LOAD_IMAGE_GRAYSCALE y CV_LOAD_IMAGE_COLOR por ejemplo)
																		#		la funcion daria fallo por tener distinto numero de canales el almacenamiento de las imagenes.
	pintaI(image) 										# Funcion ya creada para visualizar una imagen.
	return

def modificarI(im, pix, val):
	""" 4) Funcion que modifica el valor de una lista pix de coordenadas 
					de pixeles en la imagen im, siendo val el nuevo valor de los n pixeles.  """
	if(len(pix) != len(val)): 				# Comprueba que las dos listas tienen la misma longitud.
		print "FALLO: Longitud de las listas de coordenadas distintos."
	else:
		for i in range(0, len(pix)):    # Recorre las listas de coordenadas y valores.
			im[pix[i]] = val[i]						# Modifica las coordenadas de la imagen con los nuevos valores.

#################################################
##						 Programa principal							 ##
#################################################

images = ["imagenes/lena.jpg", "imagenes/lena2.jpg", "imagenes/opencv.jpg"]

im = leeimagen(images[0], cv2.CV_LOAD_IMAGE_COLOR)		# Ejercicio 1

vim = []
for img_file in images:
	im2 = leeimagen(img_file, cv2.CV_LOAD_IMAGE_COLOR) # Lee cada una de las imagenes pasadas como argumentos.
	if (im2 == None):                      							# Comprueba que existe la imagen y la ha podido leer.
		print "FALLO: Could not open or find the image"
	else:
		vim.append(im2) 																	# Almacena cada imagen en una lista.
#vim[0] = leeimagen(sys.argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)   # Esta instruccion haria que el programa fallara al llamar a la funcion pintaMI(vim)
																																# 	debido a que estariamos visualizando a la vez imagenes de distinto tipo de visualizacion.
	
pintaI(im)																					# Ejercicio 2	
pintaMI(vim)																				# Ejercicio 3

pix = [[0,0], [0,1], [0,2], [1,1], [2,3]]
val = [50, 50, 50, 50, 50]

modificarI(im, pix, val)														# Ejercicio 4

pintaI(im)
