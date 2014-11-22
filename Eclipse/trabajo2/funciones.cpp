/*
 * funciones.cpp
 *
 *  Created on: 17/11/2014
 *      Author: jmorenov
 */


#include "funciones.h"
#include <algorithm>
#include <opencv2/nonfree/nonfree.hpp>
#include <cmath>
#include <iostream>

using namespace std;
using namespace cv;

double valorHarrys(Vec6d pixel) {
    double a = pixel[0], b = pixel[1], c;
    c = pow(a + b, 2);

    return ((a * b) - (0.04 * c));
}

vector <Mat> gaussPirHarrys(Mat &imagen, int niveles) {
    //Matriz auxiliar para hacer las concatenaciones con la imagen original
    Mat aux((imagen.rows / 2), (imagen.cols / 2), CV_64FC3, 0.0);
    Mat alisada;

    vector<Mat> salida;
    //Actulizamos la imagen
    imagen.copyTo(alisada);
    salida.push_back(imagen);
    //Para cada nivel
    for (int i = 1; i < niveles; i++) {
        alisada.convertTo(alisada, CV_8U);
        //Aplicamos la convolucion a la imagen
        alisada = filterGauss(alisada, 2.0);
        //Submuestreamos
        for (int j = 0; j < aux.rows; j++) {
            for (int k = 0; k < aux.cols; k++) {
                aux.at <Vec3d> (j, k) = alisada.at <Vec3d> (j * 2, k * 2);
            }
        }
        aux.copyTo(alisada);
        aux.convertTo(aux, CV_8UC3);

        salida.push_back(aux);
        //Reseteamos la matriz auxiliar
        aux = aux.zeros((alisada.rows / 2), (alisada.cols / 2), CV_64FC3);
    }

    return salida;
}

vector <punto> puntosHarrys(vector<Mat> piramide) {
    vector<punto> puntos, maximos, aux;

    Mat resultados, valoresH;
    for (unsigned int i = 0; i < piramide.size(); i++) {

        if (piramide[i].type() != 0) {
            piramide[i].convertTo(piramide[i], CV_8UC3);
            cvtColor(piramide[i], piramide[i], CV_RGB2GRAY);
        }

        cornerEigenValsAndVecs(piramide[i], resultados, 5, 3);
        valoresH = valoresH.zeros(resultados.rows, resultados.cols, CV_64FC1);

        for (int j = 0; j < resultados.rows; j++) {
            for (int k = 0; k < resultados.cols; k++) {
                valoresH.at<double>(j, k) = valorHarrys(resultados.at<Vec6d>(j, k));
            }
        }
        aux = selecMax(valoresH, 5, i);
        //cornerHarris(piramide[i], resultados, 5, 3, 0.04, BORDER_DEFAULT);
        //aux = selecMax(resultados, 5, i);
        puntos.insert(puntos.end(), aux.begin(), aux.end());

    }
    sort (puntos.begin(), puntos.end());


    for (unsigned int i = 0; (i < 1000) && (i < puntos.size()); i++) {
        maximos.push_back(puntos[puntos.size() - i - 1]);
        //cout <<  puntos[puntos.size()-i-1].valorH << endl;
    }


    return maximos;
}

vector <punto> selecMax(Mat puntosHarrys, int tamanoVentana, int nivel) {
    Mat binaria;
    binaria = binaria.zeros(puntosHarrys.rows, puntosHarrys.cols, CV_8UC1);
    Mat aux;
    vector <punto> resultado;
    double max;
    bool esMax;
    for (int i = 0; i < puntosHarrys.rows - tamanoVentana; i++) {
        for (int j = 0; j < puntosHarrys.cols - tamanoVentana; j++) {

            aux = puntosHarrys(Rect(j, i, tamanoVentana, tamanoVentana));
            max = aux.at<double>((tamanoVentana / 2) + 1, (tamanoVentana / 2) + 1);
            esMax = true;

            for (int k = 0; k < aux.rows; k++) {
                for (int l = 0; l < aux.cols; l++) {
                    if (aux.at<double>(k, l) > max) {
                        //cout << max << " " <<
                        esMax = false;
                    }
                }
            }
            if (esMax == true) {
                binaria.at<unsigned char> (i + (tamanoVentana / 2) + 1, j + (tamanoVentana / 2) + 1) = 255;
            }


        }
    }
    for (int i = 0; i < binaria.rows; i++) {
        for (int j = 0; j < binaria.cols; j++) {
            if (binaria.at<unsigned char>(i, j) != 0) {
                resultado.push_back(punto(j, i, nivel, puntosHarrys.at<double>(i, j)));
            }
        }
    }

    return resultado;

}

void pintaCirculos(vector<Mat> piramide, vector<punto> puntos) {

    vector<Mat> circulitos;

    Scalar color = Scalar(185, 174, 255);
    circulitos.resize(piramide.size());
    for (unsigned int i = 0; i < piramide.size(); i++) {
        piramide[i].copyTo(circulitos[i]);
        if (circulitos[i].type() != 16) {
            cvtColor(circulitos[i], circulitos[i], CV_GRAY2RGB);
        }
        if (piramide[i].type() != 0) {
            cvtColor(piramide[i], piramide[i], CV_RGB2GRAY);
        }

    }

    for (unsigned int i = 0; i < puntos.size(); i++) {
        circle(circulitos[puntos[i].nivel], Point(puntos[i].x, puntos[i].y), 5, color);
    }

    for (unsigned int i = 0; i < circulitos.size(); i++) {
        pintaI(circulitos[i]);
    }

}

void refinarPuntos(Mat imagen, vector<punto> &pts) {


    Size winSize = Size( 5, 5 );
    Size zeroZone = Size( -1, -1 );
    TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );

    vector<Point2f> corners;
    for (unsigned int i = 0; i < pts.size(); i++) {
        corners.push_back(Point2f(pts[i].x, pts[i].y));
    }
    imagen.convertTo(imagen, CV_8UC1);
    cornerSubPix(imagen, corners, winSize, zeroZone, criteria );

    for (unsigned int i = 0; i < pts.size(); i++) {
        pts[i].x = corners[i].x;
        pts[i].y = corners[i].y;
    }

}

void calcularOrientacion(Mat imagen,  vector<punto> &pts) {
    Mat sobelX, sobelY;
    double deltaX, deltaY, PI = 3.14159265;
    Sobel(imagen, sobelX, CV_64F, 1, 0);
    Sobel(imagen, sobelY, CV_64F, 0, 1);

    for (unsigned int i = 0; i < pts.size(); i++) {
        deltaX = sobelX.at <double> (pts[i].x, pts[i].y);
        deltaY = sobelY.at <double> (pts[i].x, pts[i].y);
        if (deltaX == 0) {
            pts[i].orientacion = 0;
        }
        else {
            pts[i].orientacion = atan((deltaY / deltaX) * (180 / PI));
        }


    }
}

void drawOrientacion(Mat imagen,  vector<punto> &pts) {
    Mat salida;
    imagen.copyTo(salida);
    double radio;
    cvtColor(salida, salida, CV_GRAY2RGB);

    Scalar color;
    for (unsigned int i = 0; i < pts.size(); i++) {
        if (pts[i].nivel == 0) {
            color = Scalar(185, 174, 5);
        }
        if (pts[i].nivel == 1) {
            color = Scalar(45, 233, 128);
        }
        if (pts[i].nivel == 2) {
            color = Scalar(128, 45, 95);
        }
        if (pts[i].nivel == 3) {
            color = Scalar(33, 99, 200);
        }
        radio = (pts[i].nivel + 1) * 6;
        circle(salida, Point(pts[i].x, pts[i].y), radio, color, 1);

        line(salida, Point(pts[i].x, pts[i].y), Point(pts[i].x + radio * cos(pts[i].orientacion), pts[i].y + radio * sin(pts[i].orientacion)), color, 2);

        color = Scalar(185, 174, 5);
    }

    pintaI(salida);

}

void harrys(Mat imagen) {
    vector<Mat> piramide = gaussPirHarrys(imagen, 4);
    vector<punto> puntos = puntosHarrys(piramide);

    pintaCirculos(piramide, puntos);

    refinarPuntos(imagen, puntos);

    calcularOrientacion(imagen, puntos);

    drawOrientacion(imagen, puntos);
}

void sift(Mat &img1, vector<KeyPoint> &keypoints) {
    SIFT detector = SIFT();
    Mat mask;
    detector.operator()(img1, mask, keypoints);

    Mat salida;
    drawKeypoints(img1, keypoints, salida);
    //pintaI(salida);
}

void surf(Mat &img1, vector<KeyPoint> &keypoints) {
    SURF detector = SURF();
    Mat mask;
    detector.operator()(img1, mask, keypoints);

    Mat salida;
    drawKeypoints(img1, keypoints, salida);
    pintaI(salida);
}


void computeMatching(Mat &img1, Mat &img2, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches ) {
    // computing descriptors
    SiftDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(img1, keypoints1, descriptors1);
    extractor.compute(img2, keypoints2, descriptors2);

    // matching descriptors
    bool crossCheck = 1;
    BFMatcher matcher (NORM_L2, crossCheck );
    matcher.match(descriptors1, descriptors2, matches);

}

Mat mosaic(Mat &img1, Mat &img2, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches) {

    Mat H0 (3, 3, CV_64FC1), mosaico;
    H0.at<double>(0, 0) = 1;
    H0.at<double>(0, 1) = 0;
    H0.at<double>(0, 2) = img1.cols / 4;
    H0.at<double>(1, 0) = 0;
    H0.at<double>(1, 1) = 1;
    H0.at<double>(1, 2) = img1.rows / 4;
    H0.at<double>(2, 0) = 0;
    H0.at<double>(2, 1) = 0;
    H0.at<double>(2, 2) = 1;


    warpPerspective(img1, mosaico, H0, Size(img1.rows * 2, img1.cols * 2));

    vector< Point2f > imagen;
    vector< Point2f > escena;

    for ( unsigned int i = 0; i < matches.size(); i++ ) {

        imagen.push_back( keypoints1[ matches[i].queryIdx ].pt );
        escena.push_back( keypoints2[ matches[i].trainIdx ].pt );
    }

    Mat H = findHomography(escena, imagen, CV_RANSAC );

    H = H * H0;

    warpPerspective(img2, mosaico, H, mosaico.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);

    return mosaico;
}

Mat makePanorama(vector <Mat> imagenes){

    Mat mosaico;

    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    vector<DMatch> matches;

    keypoints1.clear();
    sift(imagenes[0],keypoints1);
    keypoints2.clear();
    sift(imagenes[1],keypoints2);

    matches.clear();
    computeMatching(imagenes[0],imagenes[1],keypoints1,keypoints2, matches);

    mosaico = mosaic(imagenes[0],imagenes[1],  keypoints1, keypoints2, matches);

    for(unsigned int i = 2; i<imagenes.size(); i++){

        keypoints1.clear();
        sift(mosaico,keypoints1);

        keypoints2.clear();
        sift(imagenes[i],keypoints2);

        matches.clear();
        computeMatching(mosaico,imagenes[i],keypoints1,keypoints2, matches);

        mosaico = mosaic(mosaico,imagenes[i],  keypoints1, keypoints2, matches);

    }

    return mosaico;
}

/**
 * Genera una ventana en la que pinta la imagen que se pasa en img.
 */
void pintaI(Mat &img)
{
	namedWindow("Imagen");
	imshow("Imagen",img);
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
	Size s = Size(m[0].cols,m[0].rows);
	int n_img = m.size();
	int groups = 1;

	if(n_img > 3)
	{
		if(n_img % 2 == 0)
			groups = 2;
		else if(n_img % 3 == 0)
			groups = 3;
		else
		{
			groups = 2;
			n_img++;
			Mat blank (m[0].rows, m[0].cols, 16, Scalar(255,255,255));
			m.push_back(blank);
		}
	}

	vector<Mat> aux2(groups);

	for(int i = 0; i<groups; i++)
	{
		m[i*(n_img/groups)].convertTo(aux2[i], CV_8UC3);
		resize(aux2[i],aux2[i],s);
		for(int j = i*(n_img/groups) + 1; j<n_img/groups + n_img/groups*i; j++)
		{
			m[j].convertTo(aux, CV_8UC3);
			resize(aux,aux,s);
			hconcat(aux2[i], aux, aux2[i]);
		}
	}

	res = aux2[0];
	for(int i=1; i<groups; i++)
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
	return exp(-0.5*((x*x)/(sigma*sigma)));
}

/**
 * Calcula el vector máscara para un determinado tamaño y sigma.
 */
Mat gaussMask(int size, double sigma)
{
	//Vector máscara a devolver.
	vector <double> vMask (size);
	//Valor para normalizar la máscara.
	double normal = 0.0;

	for (int i = 0; i <= size/2; i++)
	{
		//Calculo del vector simétrico.
		normal += vMask[(size/2)-i] = vMask[(size/2)+i] = gaussValue(i ,sigma);
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
    for(int i=0; i<signal.cols; i++)
    {
        m[i] = 0.0;
        for(int j=0; j<signal.rows; j++)
        {
        	//Multiplica el valor de la fila i, la columna j por el valor de la máscara en la fila j.
            m[i] += signal.at<double>(j,i) * mask.at<double>(j,0);
        }
        m[i] *= mask.at<double>(i,0);
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
    for(int i=0; i<signal.channels(); i++)
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
	Mat bh(src.rows, mask.rows/2, src.type(), 0.0);
	hconcat(bh, src, src);
	hconcat(src, bh, src);
	Mat bv(mask.rows/2, src.cols, src.type(), 0.0);
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
			if(aux.channels() > 1)
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
	Mat mask = gaussMask(4*sigma + 1, sigma);
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

    filter1 =  filterGauss(m1, sigma1);
    filter2 = filterGauss(m2, sigma2);
    filter2.convertTo(filter2, CV_64F);
    filter1.convertTo(filter1, CV_64F);

    m1.convertTo(m1, CV_64F);
    m2.convertTo(m2, CV_64F);

    //Selecciona el sigma con mayor valor para calcular las bajas frecuencias en esa imagen.
    if(sigma1 >= sigma2)
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
    for(int i=0; i<levels; i++)
    {
        pyrDown(aux, aux);
        //Añade imagen del tamaño necesario para poder concatenarla verticalmente con la pirámide.
        z = Mat::zeros(aux.rows, src.cols-aux.cols, aux.type());
        hconcat(z, aux, z);
        vconcat(pyramid, z, pyramid);
    }
    return pyramid;
}

