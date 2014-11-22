/*
 * funciones.h
 *
 *  Created on: 17/11/2014
 *      Author: jmorenov
 */

#ifndef FUNCIONES_H_
#define FUNCIONES_H_

#include <queue>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

struct punto{
	int x, y, nivel;
	double valorH, orientacion;

	punto(int xa, int ya, int nivela, double valorHa){
		x = xa;
		y = ya;
		nivel = nivela;
		valorH = valorHa;
		orientacion = 0;
	}

	bool operator < (const punto & pt) const{
		return valorH < pt.valorH;
	}

	bool operator > (const punto & pt) const {
		return valorH > pt.valorH;
	}

};

double valorHarrys(double det1, double det2);

vector <Mat> gaussPirHarrys(Mat & imagen, int niveles);

vector <punto> puntosHarrys(vector <Mat> piramide);

vector <punto> selecMax(Mat puntosHarrys, int tamanoVentana, int nivel);

void pintaCirculos(vector<Mat> piramide, vector<punto> puntos);

void refinarPuntos(Mat imagen, vector<punto> & pts);

void calcularOrientacion(Mat imagen,  vector<punto> & pts);

void drawOrientacion(Mat imagen,  vector<punto> & pts);

void harrys(Mat imagen);

void sift(Mat& img1, vector<KeyPoint>& keypoints);

void surf(Mat& img1, vector<KeyPoint>& keypoints);

void computeMatching(Mat& img1, Mat& img2,vector<KeyPoint>& keypoints1,vector<KeyPoint>& keypoints2, vector<DMatch>& matches);

Mat mosaic(Mat& img1, Mat& img2,vector<KeyPoint>& keypoints1,vector<KeyPoint>& keypoints2, vector<DMatch>& matches);

Mat makePanorama(vector <Mat> imagenes);

/**
 * Funciones implementadas en el Trabajo 1.
 */

void pintaI(Mat &img);

void pintaMI(vector<Mat> &m);

double gaussValue(int x, double sigma);

Mat gaussMask(int size, double sigma);

void convolVectorMask1C(Mat &signal, Mat &mask, double &d_ij);

void convolVectorMask3C(Mat &signal, Mat &mask, Vec3d &v_ij);

Mat setEdges(Mat src, Mat & mask);

Mat filterGauss(Mat src, Mat &mask);

Mat filterGauss(Mat &src, double sigma);

Mat imgHybrid(Mat m1, double sigma1, Mat m2, double sigma2);

Mat pyramidGaussian(Mat &src, int levels);


#endif /* FUNCIONES_H_ */
