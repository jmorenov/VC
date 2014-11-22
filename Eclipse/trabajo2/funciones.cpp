/*
 * funciones.cpp
 *
 *  Created on: 17/11/2014
 *      Author: jmorenov
 */


#include "funciones.h"

double valorHarrys(Vec6d pixel)
{
	double a = pixel[0], b = pixel[1], c;
	c=pow(a+b,2);
	return ((a*b)-(0.04*c));
}

vector <Mat> gaussPirHarrys(Mat & imagen, int niveles)
{
	//Matriz auxiliar para hacer las concatenaciones con la imagen original
	Mat aux((imagen.rows/2), (imagen.cols/2), CV_64FC3, 0.0);
	Mat alisada;

	vector<Mat> salida;
	//Actulizamos la imagen
	imagen.copyTo(alisada);
	salida.push_back(imagen);
	//Para cada nivel
	for(int i = 1; i<niveles; i++)
	{
		alisada.convertTo(alisada, CV_8U);
		//Aplicamos la convolucion a la imagen
		alisada = filterGauss(alisada, 1.0);
		//Submuestreamos
		for(int j = 0; j < aux.rows; j++)
		{
			for (int k = 0; k < aux.cols; k++)
			{
				aux.at <Vec3d> (j,k) = alisada.at <Vec3d> (j*2, k*2);
			}
		}
		aux.copyTo(alisada);
		aux.convertTo(aux, CV_8UC3);

		salida.push_back(aux);
		//Reseteamos la matriz auxiliar
		aux = aux.zeros((alisada.rows/2), (alisada.cols/2), CV_64FC3);
	}

	return salida;
}

vector <punto> puntosHarrys(Mat src)
{
	vector<punto> puntos;
	vector<Mat> piramide;
	piramide = gaussPirHarrys(src,4);
	Mat resultados;
	Vec6d aux;
	punto pt;
	priority_queue <punto> pq;
	//priority_queue <punto, vector<punto>, greater<punto> > pq;
	for(unsigned int i = 0; i < piramide.size(); i++)
	{
		resultados.zeros(piramide[0].rows, piramide[0].cols, CV_64FC(6));

		if(piramide[i].type() != 0)
		{
			cvtColor(piramide[i], piramide[i], CV_RGB2GRAY);
		}
		cornerEigenValsAndVecs(piramide[i], resultados, 7, 5);
		for(int j = 0; j<resultados.rows; j++)
		{
			for(int k = 0; k<resultados.cols; k++)
			{
				aux = resultados.at<Vec6d>(j,k);
				if(valorHarrys(aux)>0)
				{
					pt.x = j;
					pt.y = k;
					pt.nivel = i+1;
					pt.valorH = valorHarrys(aux);
					pq.push(pt);
				}
			}
		}
	}
	for(int i = 0; i < 1000; i++)
	{
		puntos.push_back(pq.top());
		pq.pop();
	}
	return puntos;
}

Mat pintaCirculos(Mat imagen)
{
	Mat salida;
	vector<punto> puntos = puntosHarrys(imagen);
	imagen.copyTo(salida);
	Scalar color = Scalar(185, 174, 255);
	salida.convertTo(salida, CV_8UC1);
	//salida.convertTo(salida, CV_GRAY2RGB);
	for(unsigned int i = 0; i < puntos.size(); i++)
	{
		circle(salida, Point(((puntos[i].x)*(puntos[i].nivel)),((puntos[i].y)*(puntos[i].nivel))), 10, color);
	}
	return salida;
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

