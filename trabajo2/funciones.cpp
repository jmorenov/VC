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
 * File: funciones.cpp
 */

#include <cmath>
#include <iostream>
#include <math.h>
#include <algorithm>

#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>

#include "funciones.h"

#define BLOCKSIZE 7 // blockSize = [3,13]
#define KSIZE 5 // kSize = [3,9]

using namespace std;
using namespace cv;

/**
 * Funciones Trabajo 2.
 */

/**
 * Calcula una lista de Mat mediante el médoto de la pirámide Gaussiana de una imagen.
 */
vector<Mat> pyramidGaussianList(const Mat &img, int levels)
{
	vector<Mat> pyramid;
	Mat img_aux = img.clone();
	pyramid.push_back(img);
	for (int i = 1; i < levels; i++)
	{
		pyrDown(img_aux, img_aux);
		pyramid.push_back(img_aux);
	}
	return pyramid;
}


/**
 * Genera una lista de puntos de Harris de una lista de Mat.
 */
void listHarrisPoints(const Mat &img, const vector<Mat> &pyramid, vector<HarrisPoint> &result)
{
	vector<HarrisPoint> pHarris, aux;
	Mat MatrixHarris, Mc;
	for (unsigned int k = 0; k < pyramid.size(); k++)
	{
		MatrixHarris = Mat::zeros(img.size(), CV_32FC(6));
		Mc = Mat::zeros(img.size(), CV_32FC1);

		cornerEigenValsAndVecs(pyramid[k], MatrixHarris, BLOCKSIZE, KSIZE);

		for (int i = 0; i < MatrixHarris.rows; i++)
		{
			for (int j = 0; j < MatrixHarris.cols; j++)
			{
				float lambda_1 = MatrixHarris.at<Vec6f>(i, j)[0];
				float lambda_2 = MatrixHarris.at<Vec6f>(i, j)[1];
				Mc.at<float>(i,j) = lambda_1*lambda_2 - 0.04f*pow((lambda_1 + lambda_2 ), 2);
			}
		}
		adaptativeNonMaximalSupression(Mc, 5, k, aux);
		pHarris.insert(pHarris.end(), aux.begin(), aux.end());
	}

	sort(pHarris.begin(), pHarris.end());
	unsigned int n_points = min(1000, (int)pHarris.size());
	for (unsigned int i = 0; i < n_points; i++)
	{
		result.push_back(pHarris[pHarris.size() - i - 1]);
	}
}

/**
 * Falta por implementar.
 */
void adaptativeNonMaximalSupression(const Mat &puntosHarrys, int tamanoVentana, int nivel, vector<HarrisPoint> &resultado)
{
	for(int i=0; i<puntosHarrys.rows; i++)
	{
		for(int j=0; j<puntosHarrys.cols; j++)
		{
			resultado.push_back(HarrisPoint(Point(j,i), nivel, puntosHarrys.at<double>(i,i)));
		}
	}
}

/**
 * Comprueba si la imagen es en color.
 */
bool checkColor(Mat img)
{
	if (img.type() >= 8)
		return true;
	return false;
}

/**
 * Convierte una imagen a blanco y negro.
 */
void convertToGray(Mat &img)
{
	cvtColor(img, img, CV_RGB2GRAY);
}

/**
 * Comprueba si una imagen es en color y la convierte a blanco y negro.
 */
void convertToGrayIfColor(Mat &img)
{
	if(checkColor(img))
		convertToGray(img);
}

/**
 * Refina una lista de puntos de Harris de una lista de Mat.
 */
void refinePoints(vector<Mat> pyramid, vector<HarrisPoint> &pHarris)
{
	for (unsigned int i = 0; i < pyramid.size(); i++)
		refinePoints(pyramid[i], pHarris, i);
}

/**
 * Refina una lista de puntos de Harris de un Mat.
 */
void refinePoints(const Mat &img, vector<HarrisPoint> &pHarris, int level)
{
	vector<Point2f> pSubPix;
	for (unsigned int i = 0; i < pHarris.size(); i++)
		if (pHarris[i].level == level || level == -1)
			pSubPix.push_back(pHarris[i].p);

	cornerSubPix(img, pSubPix, Size(5, 5), Size(-1, -1),
			TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001));

	for (unsigned int i = 0; i < pHarris.size(); i++)
		if (pHarris[i].level == level || level == -1)
			pHarris[i].p = pSubPix[i];
}

/**
 * Calcula la orientación de una lista de puntos de Harris en una imagen.
 */
void calculateOrientation(const Mat &img, vector<HarrisPoint> &pHarris)
{
	Mat sobelX, sobelY;
	double deltaX = 0, deltaY = 0;
	Sobel(img, sobelX, CV_64F, 1, 0);
	Sobel(img, sobelY, CV_64F, 0, 1);
	for (unsigned int i = 0; i < pHarris.size(); i++)
	{
		deltaX = sobelX.at<double>(pHarris[i].p);
		deltaY = sobelY.at<double>(pHarris[i].p);
		if (deltaX == 0)
			pHarris[i].orientation = 0;
		else
			pHarris[i].orientation = atan((deltaY / deltaX) * (180.0 / M_PI));
	}
}

/**
 * Función para detectar los puntos de Harris en una imagen.
 */
void detectHarris(const Mat &img, vector<HarrisPoint> &pHarris)
{
	Mat img_gray = img.clone();
	convertToGrayIfColor(img_gray);
	vector<Mat> pyramid = pyramidGaussianList(img_gray, 4);
	listHarrisPoints(img_gray, pyramid, pHarris);
	refinePoints(img_gray, pHarris);
	calculateOrientation(img_gray, pHarris);
}

/**
 * Dibuja círculos en los puntos de Harris definidos.
 */
void drawCircles(Mat &img, vector<HarrisPoint> pHarris, int level)
{
	Scalar color = Scalar(185, 174, 255);
	for (unsigned int i = 0; i < pHarris.size(); i++)
	{
		if (level == -1 || pHarris[i].level == level)
			circle(img, pHarris[i].p, 5, color);
	}
}

/**
 * Pinta los puntos de Harris en una imagen.
 */
void drawHarrisPoints(const Mat &img, const vector<HarrisPoint> &pHarris)
{
	Mat img_original = img.clone();
	drawCircles(img_original, pHarris);
	pintaI(img_original);
}

/**
 * Pinta las regiones relevantes en una imagen.
 */
void drawHarrisRegions(const Mat &img, const vector<HarrisPoint> &pHarris)
{
	Mat img_original = img.clone();
	double radio;
	Scalar color;
	for (unsigned int i = 0; i < pHarris.size(); i++)
	{
		switch (pHarris[i].level)
		{
		case 0:
			color = Scalar(185, 174, 5);
			break;
		case 1:
			color = Scalar(45, 233, 128);
			break;
		case 2:
			color = Scalar(128, 45, 95);
			break;
		default: //case 3:
			color = Scalar(33, 99, 200);
			break;
		}
		radio = (pHarris[i].level + 1) * 6;
		RotatedRect rRect = RotatedRect(pHarris[i].p, Size(radio, radio),
				pHarris[i].orientation);
		Rect brect = rRect.boundingRect();
		rectangle(img_original, brect, color);
		line(img_original, pHarris[i].p,
				Point(pHarris[i].p.x + radio * cos(pHarris[i].orientation),
						pHarris[i].p.y + radio * sin(pHarris[i].orientation)),
				color, 2);
	}
	pintaI(img_original);
}

/**
 * Detector SIFT.
 */
void detectSIFT(const Mat &img, vector<KeyPoint> &keypoints)
{
	/*int nfeatures = 0;
	 int nOctaves = 4;
	 int nOctaveLayers = 3;
	 double contrastThreshold = 0.06;
	 double edgeThreshold=10;
	 double sigma = 1.6;
	 SIFT detector = SIFT(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);*/
	Mat img_original = img.clone();
	SIFT detector = SIFT(); // Parámetros por defecto.
	detector.operator()(img_original, Mat(), keypoints);
}

/**
 * Detector SURF.
 */
void detectSURF(const Mat &img, vector<KeyPoint> &keypoints)
{
	/*double hessianThreshold = 1;
	 int nOctaves = 4;
	 int nOctaveLayers = 2;
	 bool extended = true;
	 bool upright = false;
	 SURF detector = SURF(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);*/
	Mat img_original = img.clone();
	SURF detector = SURF(); // Parámetros por defecto.
	detector.operator()(img_original, Mat(), keypoints);
}

/**
 * Pinta una lista de KeyPoint en una imagen.
 */
void drawKeyPoints(const Mat &img, const vector<KeyPoint> &keypoints)
{
	Mat img_original = img.clone();
	drawKeypoints(img_original, keypoints, img_original);
	pintaI(img_original);
}

/**
 * Calcula la correspondencia entre dos im
 */
void computeMatching(const Mat &img1, const Mat &img2, vector<KeyPoint> &keypoints1,
		vector<KeyPoint> &keypoints2, vector<DMatch> &matches, METHOD method)
{
	Mat gray1 = img1.clone();
	Mat gray2 = img2.clone();
	convertToGrayIfColor(gray1);
	convertToGrayIfColor(gray2);
	
	Mat descriptors1, descriptors2;
	if (method == SIFT_M)
	{
		SiftDescriptorExtractor extractor;
		detectSIFT(gray1, keypoints1);
		detectSIFT(gray2, keypoints2);
		// computing descriptors
		extractor.compute(gray1, keypoints1, descriptors1);
		extractor.compute(gray2, keypoints2, descriptors2);
	}
	else if (method == SURF_M)
	{
		SurfDescriptorExtractor extractor;
		detectSURF(gray1, keypoints1);
		detectSURF(gray2, keypoints2);
		// computing descriptors
		extractor.compute(gray1, keypoints1, descriptors1);
		extractor.compute(gray2, keypoints2, descriptors2);
	}
	else if (method == SIFT_AUTO)
	{
		SiftDescriptorExtractor extractor;
		SiftFeatureDetector detector(400);
		// detecting keypoints
		if (keypoints1.empty() || keypoints2.empty())
		{
			detector.detect(gray1, keypoints1);
			detector.detect(gray2, keypoints2);
		}
		// computing descriptors
		extractor.compute(gray1, keypoints1, descriptors1);
		extractor.compute(gray2, keypoints2, descriptors2);
	}
	else //SURF_AUTO
	{
		SurfFeatureDetector detector(400);
		SurfDescriptorExtractor extractor;
		// detecting keypoints
		if (keypoints1.empty() || keypoints2.empty())
		{
			detector.detect(gray1, keypoints1);
			detector.detect(gray2, keypoints2);
		}
		// computing descriptors
		extractor.compute(gray1, keypoints1, descriptors1);
		extractor.compute(gray2, keypoints2, descriptors2);
	}

	// matching descriptors
	BruteForceMatcher<L2<float> > matcher;
	matcher.match(descriptors1, descriptors2, matches);
}

/**
 * Pinta en una imagen la correspondencia entre dos imágenes.
 */
void drawImageMatches(const Mat &img1, const vector<KeyPoint> &keypoints1, const Mat &img2, const vector<KeyPoint> &keypoints2, const vector<DMatch> &matches)
{
	Mat image;
	drawMatches(img1, keypoints1, img2, keypoints2, matches, image);
	pintaI(image);
}

/**
 * Genera un mosaico entre dos imágenes.
 */
Mat computeMosaic(const Mat &img1, const Mat &img2)
{
	vector<KeyPoint> keypoints1, keypoints2;
	vector<DMatch> matches;
	computeMatching(img1, img2, keypoints1, keypoints2, matches, SURF_AUTO);

	Mat img_mosaic;
	vector<Point2f> p1, p2;
	for (unsigned int i = 0; i < matches.size(); i++)
	{
		p1.push_back(keypoints2[matches[i].trainIdx].pt);
		p2.push_back(keypoints1[matches[i].queryIdx].pt);
	}

	Mat H = findHomography(p1, p2, CV_RANSAC);
	warpPerspective(img2, img_mosaic, H,
			Size(img1.cols + img2.cols, img1.rows));
	Mat half(img_mosaic, Rect(0, 0, img1.cols, img1.rows));
	img1.copyTo(half);

	cropBlackArea(img_mosaic);

	return img_mosaic;
}

/**
 * Genera una vista panorámica entre una lista de imágenes.
 */
Mat computePanorama(const vector<Mat> &imgs)
{
	assert(imgs.size() >= 2);

	Mat panorama = computeMosaic(imgs[0], imgs[1]);
	for (unsigned int i = 2; i < imgs.size(); i++)
	{
		panorama = computeMosaic(panorama, imgs[i]);
	}

	return panorama;
}

/**
 * Elimina de una imagen las zonas en negro, útil para la generación de mosaicos y panorámicas.
 */
void cropBlackArea(Mat &img)
{
	vector<Point> nonBlackList;
	nonBlackList.reserve(img.rows * img.cols);
	for (int j = 0; j < img.rows; ++j)
		for (int i = 0; i < img.cols; ++i)
		{
			if (img.at<Vec3b>(j, i) != Vec3b(0, 0, 0))
			{
				nonBlackList.push_back(Point(i, j));
			}
		}
	Rect bb = boundingRect(nonBlackList);

	img = img(bb);
}

/**
 * Funciones Trabajo 1.
 */

/**
 * Genera una ventana en la que pinta la imagen que se pasa en img.
 */
void pintaI(Mat &img)
{
	namedWindow("Imagen");
	imshow("Imagen", img);
	waitKey(0);
	destroyWindow("Imagen");
}

/**
 * Genera una ventana en la que pinta en forma de mosaico todas las imágenes que se pasan en m.
 */
void pintaMI(vector<Mat> &m)
{
	assert(m.size() != 0);
	Mat res, aux;
	Size s = Size(m[0].cols, m[0].rows);
	int n_img = m.size();
	int groups = 1;

	if (n_img > 3)
	{
		if (n_img % 2 == 0)
			groups = 2;
		else if (n_img % 3 == 0)
			groups = 3;
		else
		{
			groups = 2;
			n_img++;
			Mat blank(m[0].rows, m[0].cols, 16, Scalar(255, 255, 255));
			m.push_back(blank);
		}
	}

	vector<Mat> aux2(groups);

	for (int i = 0; i < groups; i++)
	{
		m[i * (n_img / groups)].convertTo(aux2[i], CV_8UC3);
		resize(aux2[i], aux2[i], s);
		for (int j = i * (n_img / groups) + 1;
				j < n_img / groups + n_img / groups * i; j++)
		{
			m[j].convertTo(aux, CV_8UC3);
			resize(aux, aux, s);
			hconcat(aux2[i], aux, aux2[i]);
		}
	}

	res = aux2[0];
	for (int i = 1; i < groups; i++)
	{
		vconcat(res, aux2[i], res);
	}
	pintaI(res);
}

/**
 * Calcula el valor para el vector máscara con los valores x y sigma.
 */
double gaussValue(int x, double sigma)
{
	return exp(-0.5 * ((x * x) / (sigma * sigma)));
}

/**
 * Calcula el vector máscara para un determinado tamaño y sigma.
 */
Mat gaussMask(int size, double sigma)
{
//Vector máscara a devolver.
	vector<double> vMask(size);
//Valor para normalizar la máscara.
	double normal = 0.0;

	for (int i = 0; i <= size / 2; i++)
	{
		//Calculo del vector simétrico.
		normal += vMask[(size / 2) - i] = vMask[(size / 2) + i] = gaussValue(i,
				sigma);
	}
	normal = 1.0 / (normal * 2 - gaussValue(0, sigma));
	Mat mask(vMask, true);
	mask *= normal;
	return mask;
}

/**
 * Calcula la convolución de una matriz signal de un canal con la máscara 1D mask y devuelve el resultado en d_ij.
 * Calculo basado en la explicación: http://www.songho.ca/dsp/convolution/convolution2d_separable.html
 */
void convolVectorMask1C(Mat &signal, Mat &mask, double &d_ij)
{
	vector<double> m(signal.cols);
	d_ij = 0.0;
	for (int i = 0; i < signal.cols; i++)
	{
		m[i] = 0.0;
		for (int j = 0; j < signal.rows; j++)
		{
			//Multiplica el valor de la fila i, la columna j por el valor de la máscara en la fila j.
			m[i] += signal.at<double>(j, i) * mask.at<double>(j, 0);
		}
		m[i] *= mask.at<double>(i, 0);
		d_ij += m[i];
	}
}

/**
 * Calcula la convolución de una matriz signal de tres canales con la máscara 1D mask y devuelve el resultado en v_ij.
 */
void convolVectorMask3C(Mat &signal, Mat &mask, Vec3d &v_ij)
{
	vector<Mat> channels;
	split(signal, channels);

//Recorre cada canal.
	for (int i = 0; i < signal.channels(); i++)
	{
		//Calcula la convolución del vector de cada canal.
		convolVectorMask1C(channels[i], mask, v_ij[i]);
	}
}

/**
 * Añade bordes con valor 0 del tamaño de la mitad de la máscara a la imagen src.
 */
Mat setEdges(Mat src, Mat & mask)
{
	Mat bh(src.rows, mask.rows / 2, src.type(), 0.0);
	hconcat(bh, src, src);
	hconcat(src, bh, src);
	Mat bv(mask.rows / 2, src.cols, src.type(), 0.0);
	vconcat(bv, src, src);
	vconcat(src, bv, src);
	return src;
}

/**
 * Calcula la convolución de la imagen src con la máscara mask.
 */
Mat filterGauss(Mat src, Mat &mask)
{
	src.convertTo(src, CV_64F);
	Mat aux;
	Mat b = setEdges(src, mask);
	for (int i = 0; i < src.cols; i++)
	{
		for (int j = 0; j < src.rows; j++)
		{
			aux = b(Rect(i, j, mask.rows, mask.rows));
			if (aux.channels() > 1)
				convolVectorMask3C(aux, mask, src.at<Vec3d>(j, i));
			else
				convolVectorMask1C(aux, mask, src.at<double>(j, i));
		}
	}

	return src;
}

/**
 * Calcula la convolución de la imagen src con un valor sigma.
 */
Mat filterGauss(Mat &src, double sigma = 1.0)
{
//Genera una máscara con un tamaño calculado respecto a sigma.
	Mat mask = gaussMask(4 * sigma + 1, sigma);
//Calcula la convolución con la máscara creada.
	Mat imGauss = filterGauss(src, mask);
	return imGauss;
}

/**
 * Calcula una imagen híbrida a partir de dos imágenes en alta y baja frecuencia.
 */
Mat imgHybrid(Mat m1, double sigma1, Mat m2, double sigma2)
{
// Mayor sigma --> Más eliminación de alta frecuencia --> Usado para bajas frecuencias.
	Mat alta, baja, h, filter1, filter2, aux;

// Igualo tamaño de ambas imágenes.
	resize(m2, m2, Size(m1.cols, m1.rows));

	filter1 = filterGauss(m1, sigma1);
	filter2 = filterGauss(m2, sigma2);
	filter2.convertTo(filter2, CV_64F);
	filter1.convertTo(filter1, CV_64F);

	m1.convertTo(m1, CV_64F);
	m2.convertTo(m2, CV_64F);

//Selecciona el sigma con mayor valor para calcular las bajas frecuencias en esa imagen.
	if (sigma1 >= sigma2)
	{
		alta = m2 - filter2;
		baja = filter1;
	}
	else
	{
		alta = m1 - filter1;
		baja = filter2;
	}
	h = baja + alta;

	hconcat(baja, alta, aux);
	hconcat(aux, h, aux);

	return aux;
}

/**
 * Calcula una pirámide Gaussiana de tantos niveles como se indique.
 */
Mat pyramidGaussian(Mat &src, int levels)
{
	Mat pyramid = src;
	Mat aux = pyramid;
	Mat z;
	for (int i = 0; i < levels; i++)
	{
		pyrDown(aux, aux);
		//Añade imagen del tamaño necesario para poder concatenarla verticalmente con la pirámide.
		z = Mat::zeros(aux.rows, src.cols - aux.cols, aux.type());
		hconcat(z, aux, z);
		vconcat(pyramid, z, pyramid);
	}
	return pyramid;
}

