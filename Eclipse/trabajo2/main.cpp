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
	imgs.push_back("imagenes/yosemite_full/yosemite1.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite2.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite3.jpg");
	imgs.push_back("imagenes/yosemite_full/yosemite4.jpg");
	//imgs.push_back("imagenes/yosemite_full/yosemite5.jpg");
	//imgs.push_back("imagenes/yosemite_full/yosemite6.jpg");
	//imgs.push_back("imagenes/yosemite_full/yosemite7.jpg");

	vector<Mat> m, imgs_m(imgs.size());
	for (unsigned int i = 0; i < imgs.size(); i++)
		//imgs_m[i] = imread(imgs[i], CV_LOAD_IMAGE_GRAYSCALE);
		imgs_m[i] = imread(imgs[i]);

	/*for(unsigned int i = 0; i < imgs.size(); i++)
		detectHarris(imgs_m[i]);*/

	/*vector<KeyPoint> sift_keyp;
	for(unsigned int i=0; i<imgs.size(); i++)
		detectSIFT(imgs_m[i], sift_keyp);

	vector<KeyPoint> surf_keyp;
	for(unsigned int i=0; i<imgs.size(); i++)
		detectSURF(imgs_m[i], surf_keyp);*/

	/*vector<KeyPoint> keypoints1, keypoints2;
	vector<DMatch> matches;
	computeMatchingSIFT(imgs_m[0], imgs_m[1], keypoints1, keypoints2, matches);

	Mat img_matches;
	drawMatches(imgs_m[0], keypoints1, imgs_m[1], keypoints2, matches, img_matches);
	pintaI(img_matches);*/

	Mat panorama = computePanorama(imgs_m);
	pintaI(panorama);
	return 0;
}
