#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>

using namespace cv;
using namespace std;

void pintaI(Mat &img)
{
	namedWindow("Imagen");
	imshow("Imagen",img);
	waitKey(0);
	destroyWindow("Imagen");
}

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

double gaussValue(int x, double sigma)
{
	return exp(-0.5*((x*x)/(sigma*sigma)));
}

/**
 * EDITAR
 */
Mat gaussMask(int size, double sigma)
{
	vector <double> vMask (size);
	double normal = 0.0;
	for (int i = 0; i <= ((size/2)); i++)
	{
		normal += vMask[((size/2)-i)] = vMask[((size/2)+i)] = gaussValue(i ,sigma);
	}
	normal = 1.0 / (normal * 2 - gaussValue(0, sigma));
	Mat mask(vMask, true);
	mask *= normal;
	return mask;
}

double convolSignal1DMask(Mat &signal, Mat mask)
{
	/*Vec3d v;
	if(signal.type() == 0)
	{
		//v = signal.mul(mask);
		Mat a = signal*mask;
		//return a[0,0];
		return a[a.rows/2, a.cols/2];
		a.at
		//return (Mat(signal * mask))[signal.rows/2, mask.cols/2];
	}
	else
	{
		vector<Mat> channels;
		split(signal, channels);
		for(unsigned int i=0; i<channels.size(); i++)
			v[i] = channels[i].dot(mask);
		//merge(v, signal);
		return 0;
	}*/
}

Mat makeBorders(Mat & src, Mat & mask, int modo){
	//Creamos una matriz auxliar para trabajar comodamente
	Mat aux (1,1,src.type());
	//Si modo es igual a cero crearemos los bordes a partir de una constante: 0 tambien, es decir bordes negros
	if(modo == 0){
		//Creamos los bordes laterales con el alto de la imagen original y el ancho de la mitad del tamaño de la mascara
		Mat borders(src.rows,(mask.rows/2), src.type(), 0.0);

		//Pegamos los bordes nuevos por la derecha y por la izquierda
		hconcat(borders, src, aux);
		hconcat(aux, borders, aux);

		//Redimensionamos la matriz de bordes para poder pegarla arriba y abajo
		//Tomaremos el ancho de la imagen original + los bordes pegados anteriormente y el alto de la mitad del tamaño de la mascara
		borders = borders.zeros((mask.rows/2), aux.cols, src.type());

		//Pegamos los bordes por arriba y por abajo
		vconcat(borders, aux, aux);
		vconcat(aux, borders, aux);

	}
	else{ //En caso de que modo sea distinto de cero crearemos los bordes reflejando los pixels de la imagen original
		Mat borders(src.rows,(mask.rows/2), src.type(),2.0);

		//Partiendo la parte izquierda de la matriz tomamos los valores pertinentes de la imagen
		for (int i = 0; i < borders.rows; i++){
			for (int j = 0; j < borders.cols; j++){
				borders.at <Vec3b> (i,j) = src.at <Vec3b> (i, ((borders.cols-1)-j));
			}
		}
		//Pegamos el borde por la izquierda
		hconcat(borders, src, aux);

		//Partiendo la parte derecha de la matriz tomamos los valores pertinentes de la imagen
		for (int i = 0; i < borders.rows; i++){
			for (int j = 0; j < borders.cols; j++){
				borders.at <Vec3b> (i,j) = src.at <Vec3b> (i, ((src.cols-1)-j));
			}
		}
		//PEgamos el borde por la derecha
		hconcat(aux, borders, aux);

		//Redimensionamos la matriz de bordes para que se ajuste por arriba
		borders = borders.zeros((mask.rows/2), aux.cols, src.type());

		//Partiendo la parte superior de la matriz tomamos los valores pertinentes de la imagen
		for (int i = 0; i < borders.rows; i++){
			for (int j = 0; j < borders.cols; j++){
				borders.at <Vec3b> ((i),j) = aux.at <Vec3b> (((borders.rows-1)-i), j);
			}
		}
		//Pegamos el borde por arriba
		vconcat(borders, aux, aux);

		//Partiendo la parte inferior de la matriz tomamos los valores pertinentes de la imagen
		for (int i = 0; i < borders.rows; i++){
			for (int j = 0; j < borders.cols; j++){
				borders.at <Vec3b> (i,j) = aux.at <Vec3b> (((aux.rows-1)-i), j);
			}
		}
		//Pegamos el borde por abajo
		vconcat(aux, borders, aux);
	}

	return aux;
}

/*Mat filterGauss(Mat &im, Mat &mask, int contorno = 0)
{
	Mat imGauss = Mat(im.rows, im.cols, CV_64FC3);


	return imGauss;
}*/

Mat filterGauss(Mat &src, Mat &mask, int modo)
{
	/*if(src.type() == 0)
	 {
	 cvtColor(src, src, CV_GRAY2BGR, 3);
	 }*/
	Mat bordes = makeBorders(src, mask, modo);

	bordes.convertTo(bordes, CV_64F);

	Mat modificada(src.rows, src.cols, CV_64FC3);
	Mat aux;

	for (int i = 0; i < (modificada.cols); i++)
	{
		for (int j = 0; j < (modificada.rows); j++)
		{
			aux = bordes(Rect(i + (mask.rows / 2), j, 1, mask.rows));
			//modificada[j,i] = convolSignal1DMask(aux, mask);
		}
	}

	//Debido a un bug que corrompia los bordes de la imagen segun se aplicaba de forma sucesiva convoluciones sobre ella
	//Se ha procedido a solucionar el tema haciendo estas conversiones. Dichas conversiones no hacen absolutamente nada
	//Puesto que hacen un cambio y automaticamente lo revierten. Tras varias horas de trabajo solo en este bug esta fue la
	//unica solucion encontrada.
	/*modificada.convertTo(modificada, CV_8U);
	bordes = makeBorders(modificada, mask, modo);
	modificada.convertTo(modificada, CV_64F);
	bordes.convertTo(bordes, CV_64F);*/

	//Comenzamos a aplicar la mascara por columnas
	for (int i = 0; i < (modificada.cols); i++)
	{
		for (int j = 0; j < (modificada.rows); j++)
		{
			aux = bordes(Rect(i, j + (mask.rows / 2), mask.rows, 1));
			modificada.at<Vec3d>(j, i) = convolSignal1DMask(aux, mask.t());
		}
	}

	return modificada;
}

Mat filterGauss(Mat &im, double sigma = 1.0, int contorno = 0)
{
	Mat mask = gaussMask(4*sigma + 1, sigma);
	Mat imGauss = filterGauss(im, mask, contorno);
	return imGauss;
}

int main(int argc, char* argv[])
{
	vector<Mat> m;
	/*for(int i=1; i<argc; i++)
		//m.push_back(imread(argv[i], CV_LOAD_IMAGE_COLOR));
		//m.push_back(imread(argv[i], CV_GRAY2BGR));
		//m.push_back(imread(argv[i], CV_8UC3));
		m.push_back(imread(argv[i]));*/
	Mat im = imread(argv[1], CV_GRAY2BGR);
	m.push_back(filterGauss(im, 1.0, 0));
	m.push_back(filterGauss(im, 1.0, 1));
		m.push_back(filterGauss(im, 10.0, 0));
			m.push_back(filterGauss(im, 10.0, 1));

	pintaMI(m);
	return 0;
}
