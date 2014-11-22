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

	pintaI(imgs_m[0]);
	harrys (imgs_m[0]);
	vector<KeyPoint> keypoints1;
	surf(imgs_m[0], keypoints1);
	vector<KeyPoint> keypoints2;
	surf(imgs_m[1], keypoints2);

	vector<DMatch> matches;
	computeMatching(imgs_m[0], imgs_m[1], keypoints1, keypoints2, matches);

	Mat img_matches;
	drawMatches(imgs_m[0], keypoints1, imgs_m[1], keypoints2, matches, img_matches);
	pintaI(img_matches);

	keypoints1.clear();
	sift(imgs_m[0], keypoints1);
	keypoints2.clear();
	sift(imgs_m[1], keypoints2);

	matches.clear();
	computeMatching(imgs_m[0], imgs_m[1], keypoints1, keypoints2, matches);

	drawMatches(imgs_m[0], keypoints1, imgs_m[1], keypoints2, matches,
			img_matches);
	pintaI(img_matches);

	Mat mosaico;
	mosaico = makePanorama(imgs_m);
	pintaI (mosaico);

	return 0;
}
