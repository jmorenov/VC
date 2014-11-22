/*
 * funciones.h
 *
 *  Created on: 17/11/2014
 *      Author: jmorenov
 */

#ifndef FUNCIONES_H_
#define FUNCIONES_H_

#include <queue>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <vector>

using namespace cv;
using namespace std;

struct punto
{
	int x, y, nivel;
	double valorH;
	bool operator < (const punto & pt) const
	{
		return valorH < pt.valorH;
	}
	bool operator > (const punto & pt) const
	{
		return valorH > pt.valorH;
	}
};

double valorHarrys(Vec6d pixel);

vector <Mat> gaussPirHarrys(Mat & imagen, int niveles);

vector <punto> puntosHarrys(Mat src);

Mat pintaCirculos(Mat imagen);

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
