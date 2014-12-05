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
 * File: funciones.h
 */

#ifndef FUNCIONES_H_
#define FUNCIONES_H_

#include <queue>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

class HarrisPoint
{
	public:
	Point p;
	int level;
	double orientation;
	double valorH;

	HarrisPoint(Point _p, int _level, double _valorH)
	{
		p.x = _p.x;
		p.y = _p.y;
		level = _level;
		orientation = 0.0;
		valorH = _valorH;
	}

	bool operator <(const HarrisPoint & pt) const
	{
		return valorH < pt.valorH;
	}

	bool operator >(const HarrisPoint & pt) const
	{
		return valorH > pt.valorH;
	}
};

enum METHOD
{
	SIFT_AUTO, SURF_AUTO, SIFT_M, SURF_M, HARRIS
};

double HarrisPixel(Vec3d p);

vector<Mat> pyramidGaussianList(const Mat &img, int levels);

void adaptativeNonMaximalSupression(const Mat &MatrixHarris, int entornoSize, int level, vector<HarrisPoint> &pHarris);

void listHarrisPoints(const Mat &img, const vector<Mat> &pyramid, vector<HarrisPoint> &result);

void drawCircles(Mat img, vector<HarrisPoint> pHarris, int level = -1);

void refinePoints(vector<Mat> pyramid, vector<HarrisPoint> &pHarris);

void refinePoints(Mat img, vector<HarrisPoint> &pHarris, int level = -1);

void calculateOrientation(Mat img, vector<HarrisPoint> &pHarris);

void calculateOrientation2(Mat img, vector<HarrisPoint> &pHarris);

void detectHarris(const Mat &img, vector<HarrisPoint> &pHarris);

void drawHarrisPoints(const Mat &img, const vector<HarrisPoint> &pHarris);

void drawHarrisRegions(const Mat &img, const vector<HarrisPoint> &pHarris);

void detectSIFT(const Mat &img, vector<KeyPoint> &keypoints);

void detectSURF(const Mat &img, vector<KeyPoint> &keypoints);

void drawKeyPoints(const Mat &img, const vector<KeyPoint> &keypoints);

void computeMatching(const Mat &img1, const Mat &img2,
		vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2,
		vector<DMatch> &matches, METHOD method);

void drawImageMatches(const Mat &img1, const vector<KeyPoint> &keypoints1,
		const Mat &img2, const vector<KeyPoint> &keypoints2,
		const vector<DMatch> &matches);

Mat computeMosaic(const Mat &img1, const Mat &img2);

Mat computePanorama(const vector<Mat> &imgs);

void cropBlackArea(Mat &img);

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
