/**
 * Javier Moreno Vega <jmorenov@correo.ugr.es>
 *
 * Visión por computador.
 * Trabajo 2
 *
 * 05/12/2014
 */

#include "funciones.h"

int main(int argc, char* argv[])
{
	vector<string> imgs;
	imgs.push_back("imagenes/yosemite_full/yosemite1.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite2.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite3.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite4.jpg");
	//imgs.push_back("imagenes/yosemite_full/yosemite5.jpg");
	//imgs.push_back("imagenes/yosemite_full/yosemite6.jpg");
	//imgs.push_back("imagenes/yosemite_full/yosemite7.jpg");

	// Lectura de las imágenes.
	vector<Mat> m, images(imgs.size());
	for (unsigned int i = 0; i < imgs.size(); i++)
		images[i] = imread(imgs[i]);
	
	// Ejercicio 1: Detección de puntos Harris multiescala.
	vector<Mat> images_harris;
	for(unsigned int i = 0; i < imgs.size(); i++)
		images_harris.push_back(detectHarris(images[i]));
	pintaMI(images_harris);

	// Ejercicio 2: Comparación de los detectores SIFT y SURF con el detector implementado MOPS.
	vector<Mat> images_sift, images_surf;
	vector<KeyPoint> sift_keyp;
	for(unsigned int i=0; i<imgs.size(); i++)
		images_sift.push_back(detectSIFT(images[i], sift_keyp));

	vector<KeyPoint> surf_keyp;
	for(unsigned int i=0; i<imgs.size(); i++)
		images_surf.push_back(detectSURF(images[i], surf_keyp));

	// Ejercicio 3: Poner en correspondencia dos imágenes.
	vector<KeyPoint> keypoints1, keypoints2;
	vector<DMatch> matches;
	computeMatching(images[0], images[1], keypoints1, keypoints2, matches);
	Mat img_matches;
	drawMatches(images[0], keypoints1, images[1], keypoints2, matches, img_matches);
	pintaI(img_matches);

	// Ejercicio 4: Estimar la homografía entre dos imágenes y crear un mosaico entre ambas imágenes.
	Mat mosaic = computeMosaic(images[0], images[1]);
	pintaI(mosaic);
	
	return 0;
}
