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

Mat gaussMask(int size, double sigma)
{
	vector <double> vMask (size);
	double normal = 0.0;
	for (int i = 0; i <= size/2; i++)
	{
		normal += vMask[(size/2)-i] = vMask[(size/2)+i] = gaussValue(i ,sigma);
	}
	normal = 1.0 / (normal * 2 - gaussValue(0, sigma));
	Mat mask(vMask, true);
	mask *= normal;
	return mask;
}

Vec3d convolVectorMask(Mat &signal, Mat &mask)
{
	Vec3d v;
	vector<Mat> channels;
	split(signal, channels);
	Mat m(1, channels[0].cols, channels[0].type());
	Mat c;
	for(int i=0; i<signal.channels(); i++)
	{
		c = channels[i];
		for(int j=0; j<c.cols; j++)
		{
			m.col(j) = c.col(j).dot(mask);
		}
		v[i] = m.dot(mask.t());
	}

	/*for (int i = 0; i < channels[0].cols; i++)
	{
		for(int j=0; j < channels[0].rows; j++)
		{
			v[0] += channels[0].at<double>(i,j) * mask.at<double>(i, 0);
		}
	}
	v[i] = m.dot(mask.t());*/

	return v;
}

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
			src.at<Vec3d>(j, i) = convolVectorMask(aux, mask);
		}
	}

	return src;
}

Mat filterGauss(Mat &src, double sigma = 1.0)
{
	//cvtColor(src, src, CV_GRAY2BGR, 3);
	Mat mask = gaussMask(4*sigma + 1, sigma);
	Mat imGauss = filterGauss(src, mask);
	return imGauss;
}

Mat imgHybrid(Mat &m1, double sigma1, Mat &m2, double sigma2)
{
	Mat alta, baja, h, aux;

	// Igualo tamaño y tipo de ambas imágenes.
	resize(m2, m2, Size(m1.cols, m1.rows));
	m2.convertTo(m2, m1.type());

	baja = filterGauss(m2, sigma2);
	alta =  filterGauss(m1, sigma1);
	m1.convertTo(m1, alta.type());
	alta = 1 - alta;

	h = baja + alta;

	hconcat(baja, alta, aux);
	hconcat(aux, h, aux);
	return aux;
}

int main(int argc, char* argv[])
{
	vector<Mat> m;
	/*for(int i=1; i<argc; i++)
		//m.push_back(imread(argv[i], CV_LOAD_IMAGE_COLOR));
		//m.push_back(imread(argv[i], CV_GRAY2BGR));
		//m.push_back(imread(argv[i], CV_8UC3));
		m.push_back(imread(argv[i]));*/
	Mat m1 = imread(argv[1]);
	Mat m2 = imread(argv[2]);
	m.push_back(imgHybrid(m1, 1.0, m2, 5.0));
	/*m.push_back(im);
	m.push_back(filterGauss(im, 1.0));
	m.push_back(filterGauss(im, 1.0));
	m.push_back(filterGauss(im, 5.0));
	m.push_back(filterGauss(im, 5.0));
	m.push_back(filterGauss(im, 10.0));
	m.push_back(filterGauss(im, 10.0));*/

	pintaMI(m);
	return 0;
}
