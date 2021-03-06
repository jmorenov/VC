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
 * File: funciones.h
 */

#ifndef FUNCIONES_H_
#define FUNCIONES_H_

#include <queue>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

enum METHOD
{
	SIFT_AUTO, SURF_AUTO
};

Mat generateP();

Mat calculateCoordCam(Mat &P);

void calculateKRT(Mat &P, Mat &K, Mat &R, Mat &T);

double verifyKRT(Mat &P, Mat &K, Mat &R, Mat &T);

double calculateECM(Mat &P, Mat &_P);

Mat drawEpipolarLines(Mat &image1, vector<Point2f> &points1, Mat &F, int whichImage);

Mat calculateF(Mat &image0, Mat &image1, vector<Point2f> &pts1, vector<Point2f> &pts2, unsigned int n_points = 5000, bool sort_matches = false);

double verifyF(Mat &lines1, Mat &lines2, vector<Point2f> &points1, vector<Point2f> &points2);

double distance_to_line( Point begin, Point end, Point x);

Mat calculateE(Mat &image0, Mat &image1, vector<Point2f> &pts1, vector<Point2f> &pts2, Mat &K0,
		Mat &K1, unsigned int n_points);

void calculateMovement(Mat &image0, Mat &image1, Mat &K0, Mat &K1,
		Mat &output_R, Mat &output_t, Mat &output_P, Mat &output_P1, Mat &output_points_world);

void reconstruction3D(vector<Mat> &images, vector<Mat> &K, vector<Mat> &output_images);

void drawKeyPoints(const Mat &img, const vector<KeyPoint> &keypoints);

void computeMatching(const Mat &img1, const Mat &img2,
		vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2,
		vector<DMatch> &matches, METHOD method, bool sort_matches = false);

bool checkColor(Mat img);

void convertToGray(Mat &img);

void convertToGrayIfColor(Mat &img);

void cropBlackArea(Mat &img);

void pintaI(Mat &img);

void pintaMI(vector<Mat> &m);

#endif /* FUNCIONES_H_ */
