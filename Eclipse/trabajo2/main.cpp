/**
 * Javier Moreno Vega <jmorenov@correo.ugr.es>
 *
 * Visión por computador.
 * Trabajo 2
 *
 * 17/11/2014
 */

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>

using namespace cv;
using namespace std;

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

int main(int argc, char* argv[])
{
    vector<string> imgs;
    imgs.push_back("imagenes/bird.bmp");
    imgs.push_back("imagenes/lena.jpg");
    imgs.push_back("imagenes/cat.bmp");
    imgs.push_back("imagenes/plane.bmp");
    vector<Mat> m, imgs_m(imgs.size());
    for(unsigned int i = 0; i<imgs.size(); i++)
    	imgs_m[i] = imread(imgs[i]);

    //pintaMI(imgs_m);

    //Ejercicio 1:
    cout<<"Ejercicio 1: Convolucion 2D de una imagen."<<endl;
    for(unsigned int i=0; i<imgs_m.size(); i++)
    {
        m.push_back(filterGauss(imgs_m[i], 1.0));
        m.push_back(filterGauss(imgs_m[i], 5.0));
    }
    pintaMI(m);
    m.clear();

    //Ejercicio 2:
    vector<Mat> imgs_hybrid(imgs_m.size()-1);
    cout<<"Ejercicio 2: Imagenes híbridas de parejas de imágenes."<<endl;
    for(unsigned int i=0; i<imgs_m.size()-1; i++)
    {
    	imgs_hybrid[i] = imgHybrid(imgs_m[i], 2.0, imgs_m[i+1], 10.0);
        m.push_back(imgs_hybrid[i]);
        pintaMI(m);
        m.clear();
    }

    //Ejercicio 3:
    cout<<"Ejercicio 3: Pirámide Gaussiana."<<endl;
    Mat pyramid;
    for(unsigned int i=0; i<imgs_hybrid.size(); i++)
    {
    	pyramid = pyramidGaussian(imgs_hybrid[i], 5);
        pintaI(pyramid);
    }
    return 0;
}
