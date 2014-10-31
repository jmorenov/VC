/**
 * trabajo1.cpp
 * Falla en línea 298, función hconcat
 */

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>

using namespace cv;
using namespace std;

void pintaI(Mat im)
{
	namedWindow("Imagen",1);
	imshow("Imagen",im);
	waitKey(0);
	destroyWindow("Imagen");
}

void pintaMI(vector<Mat> im)
{
	if(im.size() == 1)
	{
		pintaI(im[0]);
	}
	else
	{
		Mat collage, aux;
			
		if(im.size()==2)
		{		
			collage = im[0];
			if(collage.type() != 16)
			{
				cvtColor(collage, collage, CV_GRAY2BGR, 3);
			}

			if(im[1].type() == 0)
			{
				cvtColor(im[1], aux, CV_GRAY2BGR, 3);
				hconcat(collage, aux, collage);
			}
			else if(im[1].type() == 22)
			{
				im[1].convertTo(aux, CV_8U);
				hconcat(collage, aux, collage);
			}			
			else
			{
				hconcat(collage, im[1], collage);
			}
			pintaI(collage);
		}
		else
		{
			int raizE = sqrt(im.size()), nfilasColl, ncolumnasColl;
			float raizF = sqrt(im.size());
			if((raizF-raizE) > 0.0)
			{
				nfilasColl = (im.size()/(raizE+1)) + 1;
				ncolumnasColl = (raizE + 1);
			}
			else
			{
				ncolumnasColl = nfilasColl = raizE;
			}
			vector<Mat> filasCollage  (nfilasColl);

			for (unsigned int i = 0; i < filasCollage.size(); i++)
			{
				filasCollage[i] = im[((nfilasColl*i))];
				if(filasCollage[i].type() == 0)
				{
					cvtColor(filasCollage[i], filasCollage[i], CV_GRAY2BGR, 3);
				}
				if(filasCollage[i].type() == 22)
				{
					filasCollage[i].convertTo(filasCollage[i], CV_8U);
				}
			}
			unsigned int index = 1, ncolumna = 0;
			while (index < im.size())
			{
				if((index%ncolumnasColl) != 0)
				{
					if (im[index].type() == 0)
					{
						cvtColor(im[index], aux, CV_GRAY2BGR, 3);
						hconcat(filasCollage[ncolumna], aux, filasCollage[ncolumna]);
					}
					else if(im[index].type() == 22)
					{
						im[index].convertTo(aux, CV_8U);
						hconcat(filasCollage[ncolumna], aux, filasCollage[ncolumna]);
					}
					else
					{
						hconcat(filasCollage[ncolumna], im[index], filasCollage[ncolumna]);
					}
					index++;
				}
				else
				{
					ncolumna++;
					index++;
				}
			}
			if(filasCollage[0].cols > filasCollage[(filasCollage.size() -1)].cols)
			{
				Mat negra (im[0].rows, im[0].cols, 16, Scalar(255,255,255));
				while(filasCollage[0].cols > filasCollage[(filasCollage.size() -1)].cols)
				{
					hconcat(filasCollage[(filasCollage.size() -1)], negra, filasCollage[(filasCollage.size() -1)]);
				}
			}
			collage = filasCollage[0];
			for (unsigned int i = 1; i < filasCollage.size(); i++)
			{
				vconcat(collage, filasCollage[i], collage);
			}
			pintaI(collage);
		}
	}
}

double gauss(int x, double sigma)
{
	double a = pow(x,2), b = pow(sigma,2), c;
	c = -0.5*(a/b);
	return exp(c);
}

Mat get_GaussMask(int size, double sigma)
{
	vector <double> gaussVec (size);
	double suma;
	suma = gaussVec[(size/2)] = gauss(0, sigma);

	for (int i = 1; i <= ((size/2)); i++)
	{
		gaussVec[((size/2)-i)] = gaussVec[((size/2)+i)] = gauss(i ,sigma);
		suma+= 2*gauss(i ,sigma);
	}

	for (unsigned int i = 0; i < gaussVec.size(); i++)
	{
		gaussVec[i]/=suma;
	}

	Mat kernel(gaussVec, true);
	//kernel = kernel*kernel.t();
	return kernel;
}

Mat makeBorders(Mat & src, Mat & mask, int modo)
{
	Mat aux (1,1,src.type());

	if(modo == 0)
	{
		Mat borders(src.rows,(mask.rows/2), src.type(), 0.0);			
		hconcat(borders, src, aux);
		hconcat(aux, borders, aux);
		borders = borders.zeros((mask.rows/2), aux.cols, src.type());
		vconcat(borders, aux, aux);
		vconcat(aux, borders, aux); 
	}
	else
	{
		Mat borders(src.rows,(mask.rows/2), src.type(),2.0);
		
		for (int i = 0; i < borders.rows; i++)
		{
			for (int j = 0; j < borders.cols; j++)
			{
				borders.at <Vec3b> (i,j) = src.at <Vec3b> (i, ((borders.cols-1)-j));
			}
		}
		hconcat(borders, src, aux);

		for (int i = 0; i < borders.rows; i++)
		{
			for (int j = 0; j < borders.cols; j++)
			{
				borders.at <Vec3b> (i,j) = src.at <Vec3b> (i, ((src.cols-1)-j));
			}
		}
		hconcat(aux, borders, aux); 
		borders = borders.zeros((mask.rows/2), aux.cols, src.type());

		for (int i = 0; i < borders.rows; i++)
		{
			for (int j = 0; j < borders.cols; j++)
			{
				borders.at <Vec3b> ((i),j) = aux.at <Vec3b> (((borders.rows-1)-i), j);
			}
		}
		vconcat(borders, aux, aux);

		for (int i = 0; i < borders.rows; i++)
		{
			for (int j = 0; j < borders.cols; j++)
			{
				borders.at <Vec3b> (i,j) = aux.at <Vec3b> (((aux.rows-1)-i), j);
			}
		}
		vconcat(aux, borders, aux);	
	}
	return aux;
}

Mat aplicarMask(Mat & src, Mat & mask, int modo)
{
	if(src.type() == 0)
	{
		cvtColor(src, src, CV_GRAY2BGR, 3);
	}

	Mat bordes = makeBorders(src, mask, modo);
	bordes.convertTo(bordes, CV_64F);
	Mat modificada (src.rows, src.cols, CV_64FC3);
	Mat aux;
	vector<Mat> canales;
	Vec3d suma;

	for (int i = 0; i < (modificada.cols); i++)
	{
		for (int j = 0; j < (modificada.rows); j++)
		{			
			aux = bordes(Rect(i+(mask.rows/2),j, 1, mask.rows));
			split(aux, canales);
			suma[0] = canales[0].dot(mask);
			suma[1] = canales[1].dot(mask);
			suma[2] = canales[2].dot(mask);
			modificada.at <Vec3d> (j,i) = suma;
		}
	}
	modificada.convertTo(modificada, CV_8U);
	bordes = makeBorders(modificada, mask, modo);
	modificada.convertTo(modificada, CV_64F);
	bordes.convertTo(bordes, CV_64F);

	for (int i = 0; i < (modificada.cols); i++)
	{
		for (int j = 0; j < (modificada.rows); j++)
		{
			aux = bordes(Rect(i,j+(mask.rows/2), mask.rows, 1));
			split(aux, canales);
			suma[0] = canales[0].dot(mask.t());
			suma[1] = canales[1].dot(mask.t());
			suma[2] = canales[2].dot(mask.t());
			modificada.at <Vec3d> (j,i) = suma;
		}
	}
	return modificada;
}

Mat my_imGaussConvol(Mat& src, double sigma, int modo)
{
	int size = (6*sigma) + 1;
	Mat kernel = get_GaussMask(size, sigma);
	Mat convol = aplicarMask(src, kernel, modo);
	return convol;
}

Mat genImagenesHibrid(Mat & im1, double sigma1, Mat & im2, double sigma2)
{
	Mat Hfrec, Lfrec, hibrida, aux;	
	aux = my_imGaussConvol(im1, sigma1, 1);
	aux.convertTo(aux, CV_8U);
	Hfrec = im1 - aux ;
	//Hfrec *= 2;

	Lfrec = my_imGaussConvol(im2, sigma2, 1);
	Lfrec.convertTo(Lfrec, CV_8U);

	if(Hfrec.size > Lfrec.size)
	{
		Hfrec.copyTo(hibrida);
		aux = hibrida(Rect(0,0, Lfrec.cols, Lfrec.rows));
		aux += Lfrec;
	}
	else
	{
		if(Lfrec.size > Hfrec.size)
		{
			Lfrec.copyTo(hibrida);
			aux = hibrida(Rect(0,0, Hfrec.cols, Hfrec.rows));
			aux += Hfrec;
		}
		else
		{
			hibrida = Hfrec + Lfrec;
		}
	}
	hconcat(Lfrec, Hfrec, aux);
	hconcat(aux, hibrida, aux);

	return aux;
}

Mat gaussPir(Mat & imagen, int niveles)
{
	Mat aux(imagen.rows, (imagen.cols/2), CV_64FC3, 0.0);
	Mat marco, alisada, salida;
	imagen.copyTo(alisada);
	imagen.copyTo(salida);
	if(imagen.type() == 0)
	{
		cvtColor(salida, salida, CV_GRAY2BGR, 3);
	}

	for(int i = 0; i<niveles; i++)
	{	
		marco = aux(Rect(0,(imagen.rows -(alisada.rows/2)), (alisada.cols/2), (alisada.rows/2)));
		alisada = my_imGaussConvol(alisada, 2.0, 1);
		for(int j = 0; j < marco.rows; j++)
		{
			for (int k = 0; k < marco.cols; k++)
			{
				marco.at <Vec3d> (j,k) = alisada.at <Vec3d> (j*2, k*2);
			}
		}
		marco.copyTo(alisada);
		aux.convertTo(aux, CV_8UC3);
		hconcat(salida, aux, salida);
		aux = aux.zeros(imagen.rows, (alisada.cols/2), CV_64FC3);
	}

	return salida;
}

double gaussDerPrimera(int x, double sigma)
{
	double a = gauss(x, sigma), b = pow(sigma,2);
	a *= x;
	a /= b;

	return -a;
}

Mat get_GaussMaskDerUno(int size, double sigma)
{
	vector <double> gaussVec (size);
	double suma;
	suma = gaussVec[(size/2)] = gaussDerPrimera(0, sigma);

	for (int i = 1; i <= ((size/2)); i++)
	{
		gaussVec[((size/2)-i)] = gaussVec[((size/2)+i)] = gaussDerPrimera(i ,sigma);
		suma+= 2*gaussDerPrimera(i ,sigma);
	}
	for (unsigned int i = 0; i < gaussVec.size(); i++)
	{
		gaussVec[i]/=suma;
	}

	Mat kernel(gaussVec, true);
	//kernel = kernel*kernel.t();
	return kernel;
}

double gaussDerSegunda(int x, double sigma)
{
	double a = gaussDerPrimera(x,sigma), b = pow(sigma, 4);
	a *= pow(x,2);
	a /= b;

	return a - gaussDerPrimera(x, sigma);
}

Mat get_GaussMaskDerDos(int size, double sigma)
{
	vector <double> gaussVec (size);
	double suma;
	suma = gaussVec[(size/2)] = gaussDerSegunda(0, sigma);

	for (int i = 1; i <= ((size/2)); i++)
	{
		gaussVec[((size/2)-i)] = gaussVec[((size/2)+i)] = gaussDerSegunda(i ,sigma);
		suma+= 2*gaussDerSegunda(i ,sigma);
	}
	for (unsigned int i = 0; i < gaussVec.size(); i++)
	{
		gaussVec[i]/=suma;
	}

	Mat kernel(gaussVec, true);
	//kernel = kernel*kernel.t();
	return kernel;
}


Mat extraerCont(Mat src, double umbral1, double umbral2)
{
	Mat canny(src.rows, src.cols, src.type());
	Canny(src, canny, umbral1, umbral2);
	pintaI(canny);
	vector<vector<Point> > contornos;
	vector<Vec4i> jerarquia;
	findContours(canny, contornos, jerarquia, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat salida;
	src.copyTo(salida);
	Scalar color = Scalar(185, 174, 255);
		
	for(unsigned int i = 0; i< contornos.size(); i++ )
	{
		drawContours(salida, contornos, i, color, 2, 8, jerarquia, 0, Point());
	}

	return salida;
}


int main(int argc, char* argv[])
{
		/*vector<Mat> m;
		for(int i=1; i<argc; i++)
		{
			m.push_back(imread(argv[i], CV_8UC3));
			resize(m[i-1], m[i-1], Size(375, 340));
		}		
		pintaMI(m);
		return 0;*/

	if(argc < 2)
	{
		Mat im, contornos;
		im = imread("imagenes/plane.bmp");
		contornos = extraerCont(im, 150.0, 70.0);
		pintaI(contornos);
		return 0;
	}
	else
	{
		vector<Mat> imagenesBN, imagenesCL;
		if (argc == 2)
		{			
			imagenesBN.push_back(imread(argv[1],0));
			resize(imagenesBN[0], imagenesBN[0], Size(375, 340));
			imagenesBN.push_back(my_imGaussConvol(imagenesBN[0], 1.0, 0));
			imagenesBN.push_back(my_imGaussConvol(imagenesBN[0], 1.0, 1));
			imagenesBN.push_back(my_imGaussConvol(imagenesBN[0], 2.0, 0));
			/*imagenesBN.push_back(my_imGaussConvol(imagenesBN[0], 2.0, 1));
			imagenesBN.push_back(my_imGaussConvol(imagenesBN[0], 10.0, 0));
			imagenesBN.push_back(my_imGaussConvol(imagenesBN[0], 10.0, 1));*/
			pintaMI(imagenesBN);
			imagenesCL.push_back(imread(argv[1],1));
			resize(imagenesCL[0], imagenesCL[0], Size(375, 340));			
			imagenesCL.push_back(my_imGaussConvol(imagenesCL[0], 1.0, 0));
			imagenesCL.push_back(my_imGaussConvol(imagenesCL[0], 1.0, 1));
			imagenesCL.push_back(my_imGaussConvol(imagenesCL[0], 2.0, 0));
			/*imagenesCL.push_back(my_imGaussConvol(imagenesCL[0], 2.0, 1));
			imagenesCL.push_back(my_imGaussConvol(imagenesCL[0], 10.0, 0));
			imagenesCL.push_back(my_imGaussConvol(imagenesCL[0], 10.0, 1));*/
			pintaMI(imagenesCL); 
			return 0;
		}
		else
		{
			if (argc == 3)
			{
				imagenesBN.push_back(imread(argv[1],0));
				resize(imagenesBN[0], imagenesBN[0], Size(375, 340));	
				pintaI(gaussPir(imagenesBN[0], atoi(argv[2])));
				imagenesCL.push_back(imread(argv[1],1));
				resize(imagenesCL[0], imagenesCL[0], Size(375, 340));	
				pintaI(gaussPir(imagenesCL[0], atoi(argv[2])));
				return 0;
			}
			else
			{
				imagenesBN.push_back(imread(argv[1],0));
				imagenesBN.push_back(imread(argv[3],0));
				resize(imagenesBN[0], imagenesBN[0], Size(375, 340));	
				resize(imagenesBN[1], imagenesBN[1], Size(375, 340));	
				pintaI(genImagenesHibrid(imagenesBN[0], atof(argv[2]), imagenesBN[1], atof(argv[4])));
				imagenesCL.push_back(imread(argv[1],1));
				imagenesCL.push_back(imread(argv[3],1));
				resize(imagenesCL[0], imagenesCL[0], Size(375, 340));	
				resize(imagenesCL[1], imagenesCL[1], Size(375, 340));	
				pintaI(genImagenesHibrid(imagenesCL[0], atof(argv[2]), imagenesCL[1], atof(argv[4])));
				return 0;
			}
		}
	}
}
