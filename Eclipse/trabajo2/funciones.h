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

/**
 * CAMBIAR!!
 */
struct PointH
{
	Point p;
	int level;
	double orientation;
	double valorH;

	PointH(Point _p, int _level, double _valorH)
	{
		p.x = _p.x;
		p.y = _p.y;
		level = _level;
		orientation = 0.0;
		valorH = _valorH;
	}

	bool operator < (const PointH & pt) const{
			return valorH < pt.valorH;
		}

		bool operator > (const PointH & pt) const {
			return valorH > pt.valorH;
		}
};

double HarrisPixel(Vec3d p);

vector<Mat> pyramidGaussianList2(Mat img, int levels);

vector<Mat> pyramidGaussianList(Mat img, int levels);

vector<PointH> adaptativeNonMaximalSupression(Mat puntosHarrys,int tamanoVentana, int nivel);

vector<PointH> listPointHarris(Mat img, vector<Mat> &pyramid);

void drawCircles(Mat img, vector<PointH> pHarris, int level = -1);

void refinePoints(vector<Mat> pyramid, vector<PointH> &pHarris);

void refinePoints(Mat img, vector<PointH> &pHarris, int level = -1);

void calculateOrientation(Mat img, vector<PointH> &pHarris);

void calculateOrientation2(Mat img, vector<PointH> &pHarris);

void drawRegions(Mat img, vector<PointH> &pHarris);

void detectPointsHarris(Mat img);

void detectSIFT(Mat img, vector<KeyPoint> &keypoints);

void detectSURF(Mat img, vector<KeyPoint> &keypoints);

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
