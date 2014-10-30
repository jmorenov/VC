#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;


void pintaI(Mat im);

void pintaMI(vector <Mat> im);

double gauss(int x, double sigma);

Mat get_GaussMask(int size, double sigma);

Mat makeBorders(Mat & src, Mat & mask, int modo);

Mat aplicarMask(Mat & src, Mat & mask, int modo);

Mat my_imGaussConvol(Mat & src, double sigma, int modo);

Mat genImagenesHibrid(Mat & im1, double sigma1, Mat & im2, double sigma2);

Mat gaussPir(Mat & imagen, int niveles);

double gaussDerPrimera(int x, double sigma);

Mat get_GaussMaskDerUno(int size, double sigma);

double gaussDerSegunda(int x, double sigma);

Mat get_GaussMaskDerDos(int size, double sigma);

Mat extraerCont(Mat src, double umbral1, double umbral2);