/*
 *
 * Visión por computador
 * Trabajo 3
 *
 * Created on: 02/01/2015
 *      Author: Javier Moreno Vega <jmorenov@correo.ugr.es>
 * Last modified on: 02/01/2015
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

using namespace std;
using namespace cv;

/**
 * Funciones Trabajo 3.
 */
Mat generateP()
{

	Mat P = Mat(3, 4, CV_64FC1);
	Mat KR;
	do
	{
		randu(P, Scalar(0), Scalar(FLT_MAX));
		KR = P(Rect(0, 0, 3, 3));
		P /= P.at<double>(P.rows / 2, P.cols / 2);
	} while (determinant(KR) < 1);

	return P;
}

Mat calculateCoordCam(Mat &P)
{
	Mat M = P(Rect(0, 0, 3, 3));
	Mat P4 = P(Rect(3, 0, 1, 3));
	Mat C = (-M.inv()) * P4;
	return C;
}

void calculateKRT(Mat &P, Mat &K, Mat &R, Mat &T)
{
	decomposeProjectionMatrix(P, K, R, T);
	T = K.inv() * P(Rect(3, 0, 1, 3));
}

double verifyKRT(Mat &P, Mat &K, Mat &R, Mat &T)
{
	Mat _P;
	hconcat(R, T, _P);
	_P = K * _P;

	return calculateECM(P, _P);
}

double calculateECM(Mat &P, Mat &_P)
{
	double ECM = 0.0;
	for (int i = 0; i < _P.rows; i++)
		for (int j = 0; j < _P.cols; j++)
		{
			ECM += pow(_P.at<double>(i, j) - P.at<double>(i, j), 2.0);
		}
	return ECM;
}

Mat drawEpipolarLines(Mat &image1, vector<Point2f> &points1, Mat &F,
		int whichImage)
{
	vector<Vec3f> lines1;
	computeCorrespondEpilines(Mat(points1), whichImage, F, lines1);
	Mat lines = Mat(2, lines1.size(), CV_64FC1);

	int lines_c = 0;
	Point p1, p2;
	for (vector<cv::Vec3f>::const_iterator it = lines1.begin();
			it != lines1.end(); ++it)
	{
		if (lines_c < 200)
		{
			p1 = Point(0, -(*it)[2] / (*it)[1]);
			p2 = Point(image1.cols,
					-((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]);
			line(image1, p1, p2, Scalar(0, 0, 255));
			lines.at<Point>(0, lines_c) = p1;
			lines.at<Point>(1, lines_c) = p2;
			lines_c++;
		}
	}

	return lines;
}

Mat calculateF(Mat &image0, Mat &image1, vector<Point2f> &pts1,
		vector<Point2f> &pts2)
{
	vector<KeyPoint> keypoints1, keypoints2;
	vector<DMatch> matches;
	computeMatching(image0, image1, keypoints1, keypoints2, matches, SURF_AUTO);
	for (unsigned int i = 0; i < matches.size(); i++)
	{
		pts1.push_back(keypoints1[matches[i].queryIdx].pt);
		pts2.push_back(keypoints2[matches[i].trainIdx].pt);
	}
	Mat F = findFundamentalMat(pts1, pts2, CV_FM_RANSAC);
	return F;
}

// MAL
double verifyF(Mat &lines1, Mat &lines2, vector<Point2f> &points1,
		vector<Point2f> &points2)
{
	double error1 = 0.0;
	for (int i = 0; i < lines1.rows; i++)
	{
		error1 += distance_to_line(lines1.at<Point>(0, i),
				lines1.at<Point>(1, i), points1[i]);
	}
	error1 /= lines1.rows;

	double error2 = 0.0;
	for (int i = 0; i < lines2.rows; i++)
	{
		error2 += distance_to_line(lines2.at<Point>(0, i),
				lines2.at<Point>(1, i), points2[i]);
	}
	error2 /= lines2.rows;

	return (error1 + error2) / 2;
}

double distance_to_line(Point begin, Point end, Point x)
{
	//translate the begin to the origin
	end -= begin;
	x -= begin;

	//¿do you see the triangle?
	double area = x.cross(end);
	return area / norm(end);
}

void calculateMovement(Mat &image0, Mat &image1, Mat &K0, Mat &K1, Mat &radial0,
		Mat &radial1, Mat &R0, Mat &R1, Mat &t0, Mat &t1, Mat &output_R,
		Mat &output_t)
{
	vector<Point2f> pts1;
	vector<Point2f> pts2;
	Mat F = calculateF(image0, image1, pts1, pts2);

	Mat E = K1.t() * F * K0;
	Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1); // diag(1,1,0)
	SVD svd_(E);
	Mat newE = svd_.u * Mat(W) * svd_.vt;

	SVD svd(newE);
	vector<Mat> R(2), t(2);
	R[0] = svd.u * Mat(W) * svd.vt; // U W V^T
	R[1] = svd.u * Mat(W).t() * svd.vt; // U W^T V^T
	t[0] = svd.u.col(2); // +u3
	t[1] = -svd.u.col(2); // -u3

	// 4 soluciones, solo una es correcta:
	vector<Mat> P1(4);
	hconcat(R[0], t[0], P1[0]);
	hconcat(R[0], t[1], P1[1]);
	hconcat(R[1], t[0], P1[2]);
	hconcat(R[1], t[1], P1[3]);
	Mat P = Mat::eye(3, 4, CV_64FC1);

	vector<Mat> points_world(4);
	triangulatePoints(P, P1[0], pts1, pts2, points_world[0]);
	triangulatePoints(P, P1[1], pts1, pts2, points_world[1]);
	triangulatePoints(P, P1[2], pts1, pts2, points_world[2]);
	triangulatePoints(P, P1[3], pts1, pts2, points_world[3]);

	vector<int> points_front_cameras(4, 0);
	int max_points = 0, max_k = 0;
	Mat cam0 = calculateCoordCam(P), cam1;
	for (unsigned int k = 0; k < points_world.size(); k++)
	{
		cam1 = calculateCoordCam(P1[k]);
		for (int i = 0; i < points_world[k].rows; i++)
		{
			for (int j = 0; j < points_world[k].cols; j++)
			{
				if (Mat(points_world[k].at<Point3d>(i, j)).dot(
						cam1) >= 0.0
						&& Mat(points_world[k].at<Point3d>(i, j)).dot(
								cam0) >= 0.0)
				{
					points_front_cameras[k]++;
				}
			}
		}
		if (max_points < points_front_cameras[k])
		{
			max_points = points_front_cameras[k];
			max_k = k;
		}
	}

	if (max_k == 0 || max_k == 1)
		output_R = R[0].clone();
	else
		output_R = R[1].clone();

	if (max_k == 0 || max_k == 2)
		output_t = t[0].clone();
	else
		output_t = t[1].clone();
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
void computeMatching(const Mat &img1, const Mat &img2,
		vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2,
		vector<DMatch> &matches, METHOD method)
{
	Mat gray1 = img1.clone();
	Mat gray2 = img2.clone();
	convertToGrayIfColor(gray1);
	convertToGrayIfColor(gray2);

	Mat descriptors1, descriptors2;
	if (method == SIFT_AUTO)
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
	if (checkColor(img))
		convertToGray(img);
}

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

