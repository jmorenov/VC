#include "trabajo1.h"
#include <opencv2/opencv.hpp>
#include <cstdlib>

using namespace std;
using namespace cv;


int main(int argc, char* argv[]){

	if(argc < 2){
		Mat im, contornos;
		im = imread("imagenes/plane.bmp");

		contornos = extraerCont(im, 150.0, 70.0);

		pintaI(contornos);

		return 0;
	}
	else{
		vector<Mat> imagenesBN, imagenesCL;

		if (argc == 2){
			
			imagenesBN.push_back(imread(argv[1],0));

			imagenesBN.push_back(my_imGaussConvol(imagenesBN[0], 1.0, 0));
			imagenesBN.push_back(my_imGaussConvol(imagenesBN[0], 1.0, 1));
			/*imagenesBN.push_back(my_imGaussConvol(imagenesBN[0], 2.0, 0));
			imagenesBN.push_back(my_imGaussConvol(imagenesBN[0], 2.0, 1));
			imagenesBN.push_back(my_imGaussConvol(imagenesBN[0], 10.0, 0));
			imagenesBN.push_back(my_imGaussConvol(imagenesBN[0], 10.0, 1));*/
			pintaMI(imagenesBN);



			imagenesCL.push_back(imread(argv[1],1));
	
			imagenesCL.push_back(my_imGaussConvol(imagenesCL[0], 1.0, 0));
			imagenesCL.push_back(my_imGaussConvol(imagenesCL[0], 1.0, 1));
			/*imagenesCL.push_back(my_imGaussConvol(imagenesCL[0], 2.0, 0));
			imagenesCL.push_back(my_imGaussConvol(imagenesCL[0], 2.0, 1));
			imagenesCL.push_back(my_imGaussConvol(imagenesCL[0], 10.0, 0));
			imagenesCL.push_back(my_imGaussConvol(imagenesCL[0], 10.0, 1));*/
			pintaMI(imagenesCL); 

			return 0;

		}
		else{
			if (argc == 3){
				imagenesBN.push_back(imread(argv[1],0));
				pintaI(gaussPir(imagenesBN[0], atoi(argv[2])));

				imagenesCL.push_back(imread(argv[1],1));
				pintaI(gaussPir(imagenesCL[0], atoi(argv[2])));

				return 0;
			}
			else{
				imagenesBN.push_back(imread(argv[1],0));
				imagenesBN.push_back(imread(argv[3],0));
				pintaI(genImagenesHibrid(imagenesBN[0], atof(argv[2]), imagenesBN[1], atof(argv[4])));

				imagenesCL.push_back(imread(argv[1],1));
				imagenesCL.push_back(imread(argv[3],1));
				pintaI(genImagenesHibrid(imagenesCL[0], atof(argv[2]), imagenesCL[1], atof(argv[4])));

				return 0;

			}

		}
		
	}
	
	
}
