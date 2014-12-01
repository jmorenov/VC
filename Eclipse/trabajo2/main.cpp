/**
 * Javier Moreno Vega <jmorenov@correo.ugr.es>
 *
 * Visión por computador.
 * Trabajo 2
 *
 * 17/11/2014
 */

#include "funciones.h"


int main(int argc, char* argv[])
{
	vector<string> imgs;
	/*imgs.push_back("yosemite/Yosemite1.jpg");
	 imgs.push_back("yosemite/Yosemite2.jpg");*/
	imgs.push_back("imagenes/yosemite_full/yosemite1.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite2.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite3.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite4.jpg");

	vector<Mat> m, imgs_m(imgs.size());
	for (unsigned int i = 0; i < imgs.size(); i++)
		imgs_m[i] = imread(imgs[i], CV_LOAD_IMAGE_GRAYSCALE);

	detectPointsHarris(imgs_m[0]);

	return 0;
}
