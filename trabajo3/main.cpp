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
 * File: main.cpp
 */

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "funciones.h"

using namespace std;

void readImages(const vector<string> &imgs, vector<Mat> &images);
bool readData(const string file, Mat &K, Mat &radial_distorsion, Mat &R,
		Mat &t);
void computeLine(string line_s, vector<double> &output);
void ejercicio1(vector<Mat> images);
void ejercicio2(vector<Mat> images);
void ejercicio3(vector<Mat> images);
void ejercicio4(vector<Mat> images);

int main(int argc, char* argv[])
{
	cout << "Inicio Trabajo 3." << endl;

	vector<string> imgs;
	imgs.push_back("imagenes/Vmort1.pgm");
	imgs.push_back("imagenes/Vmort2.pgm");
	vector<Mat> images(imgs.size());
	readImages(imgs, images);
	//ejercicio1(images);
	//ejercicio2(images);

	vector<string> imgs_reconstruccion;
	imgs_reconstruccion.push_back("imagenes/reconstruccion/rdimage.000.ppm");
	imgs_reconstruccion.push_back("imagenes/reconstruccion/rdimage.001.ppm");
	imgs_reconstruccion.push_back("imagenes/reconstruccion/rdimage.004.ppm");
	vector<Mat> images_reconstruccion(imgs_reconstruccion.size());
	readImages(imgs_reconstruccion, images_reconstruccion);
	//ejercicio3(images_reconstruccion);

	ejercicio4(images);

	cout << "Fin Trabajo 3." << endl;
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

bool readData(const string file_input, Mat &K, Mat &radial_distorsion, Mat &R,
		Mat &t)
{
	string line_s;
	vector<double> n;
	ifstream file(file_input.c_str());
	if (file.is_open())
	{
		// (3x3) camera matrix K
		K = Mat(3, 3, CV_64F);
		for (int k = 0; k < 3; k++)
		{
			getline(file, line_s);
			computeLine(line_s, n);
			K.at<double>(k, 0) = n[0];
			K.at<double>(k, 1) = n[1];
			K.at<double>(k, 2) = n[2];
		}

		// (3) radial distortion parameters
		radial_distorsion = Mat(1, 3, CV_64F);
		getline(file, line_s);
		computeLine(line_s, n);
		radial_distorsion.at<double>(0, 0) = n[0];
		radial_distorsion.at<double>(0, 1) = n[1];
		radial_distorsion.at<double>(0, 2) = n[2];

		// (3x3) rotation matrix R
		R = Mat(3, 3, CV_64F);
		for (int k = 0; k < 3; k++)
		{
			getline(file, line_s);
			computeLine(line_s, n);
			R.at<double>(k, 0) = n[0];
			R.at<double>(k, 1) = n[1];
			R.at<double>(k, 2) = n[2];
		}

		// (3) translation vector t
		t = Mat(1, 3, CV_64F);
		getline(file, line_s);
		computeLine(line_s, n);
		t.at<double>(0, 0) = n[0];
		t.at<double>(0, 1) = n[1];
		t.at<double>(0, 2) = n[2];

		file.close();
		return true;
	}
	return false;
}

void computeLine(string line_s, vector<double> &output)
{
	output.clear();
	for (unsigned int i = 0; i < line_s.size(); i++)
	{
		int j;
		for (j = i; line_s[j] != ' ' && line_s[j] != '\t' && line_s[j] != '\n';
				j++)
		{
		}
		output.push_back(atof(line_s.substr(i, j - i).c_str()));
		i = j;
	}
}

/**
 * Ejercicio 1
 */
void ejercicio1(vector<Mat> images)
{
	cout << "Ejercicio 1:" << endl;
	Mat P = generateP();
	Mat C = calculateCoordCam(P);
	Mat K, R, T;
	calculateKRT(P, K, R, T);
	double ECM = verifyKRT(P, K, R, T);
	cout << "Error: " << ECM << endl;
}

/**
 * Ejercicio 2
 */
void ejercicio2(vector<Mat> images)
{
	cout << "Ejercicio 2:" << endl;

	vector<Point2f> pts1, pts2;
	Mat F = calculateF(images[0], images[1], pts1, pts2);

	Mat lines1 = drawEpipolarLines(images[0], pts1, F, 1);
	Mat lines2 = drawEpipolarLines(images[1], pts2, F, 2);

	pintaMI(images);

	double error = verifyF(lines1, lines2, pts1, pts2);
	cout << "Error: " << error << endl;
}

/**
 * Ejercicio 3
 */
void ejercicio3(vector<Mat> images)
{
	cout << "Ejercicio 3:" << endl;
	vector<Mat> K(3), radial(3), R(3), t(3);
	readData("imagenes/reconstruccion/rdimage.000.ppm.camera", K[0], radial[0], R[0],
			t[0]);
	readData("imagenes/reconstruccion/rdimage.001.ppm.camera", K[1], radial[1], R[1],
			t[1]);
	/*readData("imagenes/reconstruccion/rdimage.004.ppm.camera", K[2], radial[2], R[2],
			t[2]);*/

	vector<Mat> movement_R(3), movement_t(3);
	cout << "Pareja de imágenes: 000 y 001:" << endl;
	calculateMovement(images[0], images[1], K[0], K[1], radial[0], radial[1], R[0], R[1], t[0], t[1], movement_R[0], movement_t[0]);
	cout<<"R: "<<"\n"<<movement_R[0]<<endl;
	cout<<"t: "<<"\n"<<movement_t[0]<<endl;

	cout << "Pareja de imágenes: 000 y 004:" << endl;
	calculateMovement(images[0], images[2], K[0], K[2], radial[0], radial[2], R[0], R[2], t[0], t[2], movement_R[1], movement_t[1]);
	cout<<"R: "<<"\n"<<movement_R[1]<<endl;
	cout<<"t: "<<"\n"<<movement_t[1]<<endl;

	cout << "Pareja de imágenes: 001 y 004:" << endl;
	calculateMovement(images[1], images[2], K[1], K[2], radial[1], radial[2], R[1], R[2], t[1], t[2], movement_R[2], movement_t[2]);
	cout<<"R: "<<"\n"<<movement_R[2]<<endl;
	cout<<"t: "<<"\n"<<movement_t[2]<<endl;
}

/**
 * Ejercicio 4
 */
void ejercicio4(vector<Mat> images)
{
	cout << "Ejercicio 4:" << endl;

}
