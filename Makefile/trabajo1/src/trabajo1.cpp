#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>

using namespace cv;
using namespace std;

void pintaI(Mat img)
{
	namedWindow("Imagen",1);
	imshow("Imagen",img);
	waitKey(0);
	destroyWindow("Imagen");
}

void tile(const vector<Mat> &src, Mat &dst, int grid_x, int grid_y)
{
    // patch size
    int width  = dst.cols/grid_x;
    int height = dst.rows/grid_y;

    // iterate through grid
    int k = 0;
    for(int i = 0; i < grid_y; i++)
		{
        for(int j = 0; j < grid_x; j++)
				{
            Mat s = src[k];
						//cvtColor(s, s, CV_GRAY2BGR, 3);            
						resize(s,s,Size(width,height));
            s.copyTo(dst(Rect(j*width,i*height,width,height)));
						//s.copyTo(dst(Rect(0,0,width, height)));
						//s.copyTo(dst(Rect()));	
					k++;	
    	}
    }
}

void pintaMI(vector<Mat> im)
{
	int gridx=3, gridy=1;
	Mat res = Mat(600,600, CV_8UC3);
  tile(im,res,gridx,gridy);
  pintaI(res);
	/*Mat collage, aux;
	collage = im[0];
	cvtColor(collage, collage, CV_GRAY2BGR, 3);
	aux = im[1];
	//cvtColor(aux, aux, CV_GRAY2BGR, 3);
	cout<<collage.type()<<" "<<aux.type()<<endl;

	hconcat(collage, aux, collage);
	pintaI(collage);	*/
}

int main(int argc, char* argv[])
{
	vector<Mat> m;
	for(int i=1; i<argc; i++)
		m.push_back(imread(argv[i], 1));
	//m.push_back(imread(argv[2], 1));
	pintaMI(m);
	return 0;
}
