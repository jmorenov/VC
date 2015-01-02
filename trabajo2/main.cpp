/*
 *
 * Visión por computador
 * Trabajo 2
 *
 * Created on: 17/11/2014
 *      Author: Javier Moreno Vega <jmorenov@correo.ugr.es>
 * Last modified on: 05/12/2014
 * 	Modified by: Javier Moreno Vega <jmorenov@correo.ugr.es>
 * 
 * File: main.cpp
 */

#include <iostream>

#include "funciones.h"

using namespace std;

void readImages(const vector<string> &imgs, vector<Mat> &images);
void ejercicio1(vector<Mat> images);
void ejercicio2(vector<Mat> images);
void ejercicio3(vector<Mat> images);
void ejercicio4(vector<Mat> images);

int main(int argc, char* argv[])
{
	cout << "Inicio Trabajo 2." << endl;
	vector<string> imgs;
	imgs.push_back("imagenes/yosemite_full/yosemite1.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite2.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite3.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite4.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite5.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite6.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite7.jpg");

	vector<Mat> images(imgs.size());
	readImages(imgs, images);
	ejercicio1(images);
	ejercicio2(images);
	ejercicio3(images);
	ejercicio4(images);

	cout << "Fin Trabajo 2." << endl;
	return 0;
}

/**
 * Lectura de las imágenes.
 */
void readImages(const vector<string> &imgs, vector<Mat> &images)
{
	cout << "Leyendo imágenes..." << endl;
	for (unsigned int i = 0; i < imgs.size(); i++)
		images[i] = imread(imgs[i]);
	cout << "Finalizada lectura de imágenes." << endl;
}

/**
 * Ejercicio 1: Detección de puntos Harris multiescala.
 */
void ejercicio1(vector<Mat> images)
{
	vector<HarrisPoint> pointharris;
	cout << "Ejercicio 1:" << endl;
	cout << "Detectando Puntos Harris..." << endl;
	for (unsigned int i = 0; i < images.size(); i++)
	{
		cout << "Detectando imagen " << i << "..." << endl;
		detectHarris(images[i], pointharris);
		cout << "Pintando puntos harris en imagen " << i << "..." << endl;
		drawHarrisPoints(images[i], pointharris);
		cout << "Pintando regiones en imagen " << i << "..." << endl;
		drawHarrisRegions(images[i], pointharris);
	}
}

/**
 * Ejercicio 2: Comparación de los detectores SIFT y SURF con el detector implementado MOPS.
 */
void ejercicio2(vector<Mat> images)
{
	vector<KeyPoint> keypoints;
	cout << "Ejercicio 2:" << endl;

	// SIFT
	cout << "Detectando keypoints SIFT..." << endl;
	for (unsigned int i = 0; i < images.size(); i++)
	{
		cout << "Detectando imagen " << i << "..." << endl;
		detectSIFT(images[i], keypoints);
		cout << "Pintando keypoints en imagen " << i << "..." << endl;
		drawKeyPoints(images[i], keypoints);
	}

	// SURF
	cout << "Detectando keypoints SURF..." << endl;
	for (unsigned int i = 0; i < images.size(); i++)
	{
		cout << "Detectando imagen " << i << "..." << endl;
		detectSURF(images[i], keypoints);
		cout << "Pintando keypoints en imagen " << i << "..." << endl;
		drawKeyPoints(images[i], keypoints);
	}
}

/**
 * Ejercicio 3: Poner en correspondencia dos imágenes.
 */
void ejercicio3(vector<Mat> images)
{
	cout << "Ejercicio 3:" << endl;
	vector<KeyPoint> keypoints1, keypoints2;
	vector<DMatch> matches;
	cout << "Detectando correspondencia en imágenes..." << endl;
	computeMatching(images[0], images[1], keypoints1, keypoints2, matches,
			SIFT_M);
	cout << "Pintando correspondencia en imágenes..." << endl;
	drawImageMatches(images[0], keypoints1, images[1], keypoints2, matches);
}

/**
 * Ejercicio 4: Estimar la homografía entre dos imágenes y crear un mosaico entre ambas imágenes.
 */
void ejercicio4(vector<Mat> images)
{
	cout << "Ejercicio 4:" << endl;
	cout << "Generando mosaico entre dos imágenes..." << endl;
	Mat mosaic = computeMosaic(images[0], images[1]);
	cout << "Pintando mosaico..." << endl;
	pintaI(mosaic);
}
