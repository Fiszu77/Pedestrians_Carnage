#include<opencv2/opencv.hpp>
#include<iostream>
#include "opencv2/opencv.hpp"
#include <sstream>
#include <windows.h>
#include <thread>
#include "Stopwatch.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

//TODO detect when threads finished processing rois, leave moving things


void Morphology_Operations(int, void*);
void efficientGrayscale(Mat toGray, Mat &out)
{
	const int lutlength = 768;
	uchar LUT[lutlength] = {};
	float pointthree = 1 / 3;
	for (int i = 0; i < lutlength; ++i)
	{
		LUT[i] = (uchar)(i/3);
	}

	if (toGray.depth() != CV_8U)
		toGray.convertTo(toGray, CV_8UC3);
	out = Mat(toGray.rows, toGray.cols, CV_8UC1);

	int nRows = out.rows;
	int nCols = out.cols;

	if (toGray.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}
	int i, j;
	uchar* po;
	uchar* pg;
	for (i = 0; i < nRows; ++i)
	{
		pg = toGray.ptr<uchar>(i);
		po = out.ptr<uchar>(i);
		for (j = 0; j < nCols; ++j)
		{
			po[j] = LUT[(pg[(j - 1) * 3 + 1] + pg[(j - 1) * 3 + 2] + pg[(j - 1) * 3 + 3])];
		}
	}

}

void simpleGrayscale(Mat& toGray, Mat& out)
{
	//Mat out(toGray.cols, toGray.rows, CV_8UC1);
	if(toGray.depth() != CV_8U)
	toGray.convertTo(toGray, CV_8UC3);
	if (out.depth() != CV_8UC1)
	out.convertTo(toGray, CV_8UC1);
	//Mat output = toGray.clone();
	for (int i = 0; i < toGray.cols; ++i) {
		for (int j = 0; j < toGray.rows; ++j) {
			Vec3b intensity = toGray.at<Vec3b>(Point(i, j));
			//cout << (intensity.val[0] + intensity.val[1] + intensity.val[2]) / 3 << endl;
			out.at<uchar>(Point(i, j)) = (uchar)((intensity.val[0] + intensity.val[1] + intensity.val[2]) / 3);
		}
	}
}

const int density = 30;

class MultiArr {
public:
	uchar** toProcess;
	int dimensions;
	int density;

	MultiArr(int dens, int dim)
	{
		dimensions = dim;
		density = dens;
		toProcess = new uchar * [dens];

		for (int i = 0; i < dens; ++i)
		{
			toProcess[i] = new uchar[dim];
		}
		for (int i = 0; i < dens; ++i)
		{
			for (int j = 0; j < dim; ++j)
			{
				toProcess[i][j] = (uchar)255;
			}
		}
	}

	~MultiArr()
	{
		for (int i = 0; i < density; ++i)
		{
			delete[] toProcess[i];
		}
		delete[] toProcess;
	}
};
class MultiArrMat {
public:
	Mat** toProcess;
	int dimensions;
	int density;
	MultiArrMat(int dens, int dim)
	{
		dimensions = dim;
		density = dens;
		toProcess = new Mat * [dens];

		for (int i = 0; i < dens; ++i)
		{
			toProcess[i] = new Mat[dim];
		}
		/*for (int i = 0; i < dens; ++i)
		{
			for (int j = 0; j < dim; ++j)
			{
				toProcess[i][j] = sth
			}
		}*/
	}

	~MultiArrMat()
	{
		for (int i = 0; i < density; ++i)
		{
			delete[] toProcess[i];
		}
		delete[] toProcess;
	}
};


void pCarnage(vector<Mat>& framesILikeInColour, Mat& medianeC, const int LUTB[255], const int LUTG[255], const int LUTR[255], int id = 0)
{
	Stopwatch timer;
	int nRows = framesILikeInColour[0].rows;
	int nCols = framesILikeInColour[0].cols;
	const int dim = nCols * nRows * 3;
	//cout << "rows"<<framesILikeInColour[0].rows << endl;
	MultiArr all(density, dim);
	timer.startTimer();
	int hDens = (int)(density * 0.5f);
	
	//int rowsToM = nCols;

	uchar* pf;
	uchar* pm;

	
	for (int k = 0; k < density; k++)
	{
		if (framesILikeInColour[k].isContinuous())
		{
			nCols *= nRows;
			nRows = 1;
		}

		for (int i = 0; i < nRows; i++)
		{
			pf = framesILikeInColour[k].ptr<uchar>(i);
			pm = medianeC.ptr<uchar>(i);
			for (int j = 0; j < nCols; j++) {
				for (int l = 0; l < density; l++)
				{
					if (LUTR[all.toProcess[l][i * framesILikeInColour[0].rows + j * 3 + 2]] +
						LUTG[all.toProcess[l][i * framesILikeInColour[0].rows + j * 3 + 1]] +
						LUTB[all.toProcess[l][i * framesILikeInColour[0].rows + j * 3]] > LUTR[pf[j * 3 + 2]] + LUTG[pf[j * 3 + 1]] + LUTB[pf[j * 3]])
					{
						for (int m = density - 1; m > l; m--)
						{
							all.toProcess[m][i * framesILikeInColour[0].rows + j * 3] = all.toProcess[m - 1][i * framesILikeInColour[k].rows + j * 3];
							all.toProcess[m][i * framesILikeInColour[0].rows + j * 3 + 1] = all.toProcess[m - 1][i * framesILikeInColour[k].rows + j * 3 + 1];
							all.toProcess[m][i * framesILikeInColour[0].rows + j * 3 + 2] = all.toProcess[m - 1][i * framesILikeInColour[k].rows + j * 3 + 2];
						}
						all.toProcess[l][i * framesILikeInColour[0].rows + j * 3] = pf[j * 3];
						all.toProcess[l][i * framesILikeInColour[0].rows + j * 3 + 1] = pf[j * 3 + 1];
						all.toProcess[l][i * framesILikeInColour[0].rows + j * 3 + 2] = pf[j * 3 + 2];
						break;
					}
				}
				if (k == density - 1)
				{
					pm[3 * id * nCols + j * 3] = all.toProcess[hDens][i * framesILikeInColour[0].rows + j * 3];
					pm[3 * id * nCols + j * 3 + 1] = all.toProcess[hDens][i * framesILikeInColour[0].rows + j * 3 + 1];
					pm[3 * id * nCols + j * 3 + 2] = all.toProcess[hDens][i * framesILikeInColour[0].rows + j * 3 + 2];
				}

				/*if (j == 99)
				{
					cout << "real num: " <<(int) pf[j] << endl;
					for (int g = 0; g < density; g++)
					{
						cout <<(int) all.toProcess[g][i * framesILike[k].rows + j] << endl;
					}
				}*/
			}
		}
	}
	timer.getTimeWithMessage();
}

void pCarnageArr(Mat framesILikeInColour[density], Mat& medianeC, const int LUTB[255], const int LUTG[255], const int LUTR[255],int defRoiH, int id = 0)
{
	Stopwatch timer;
	int nRows = framesILikeInColour[0].rows;
	int nCols = framesILikeInColour[0].cols;
	int properIndexCols = defRoiH * framesILikeInColour[0].cols;
	const int dim = nCols * nRows * 3;
	//cout << "rows"<<framesILikeInColour[0].rows << endl;
	MultiArr all(density, dim);
	timer.startTimer();
	int hDens = (int)(density * 0.5f);

	//int rowsToM = nCols;

	uchar* pf;
	uchar* pm;


	for (int k = 0; k < density; k++)
	{
		if (framesILikeInColour[k].isContinuous())
		{
			nCols *= nRows;
			nRows = 1;
		}

		for (int i = 0; i < nRows; i++)
		{
			pf = framesILikeInColour[k].ptr<uchar>(i);
			pm = medianeC.ptr<uchar>(i);
			for (int j = 0; j < nCols; j++) {
				for (int l = 0; l < density; l++)
				{
					if (LUTR[all.toProcess[l][i * framesILikeInColour[0].rows + j * 3 + 2]] +
						LUTG[all.toProcess[l][i * framesILikeInColour[0].rows + j * 3 + 1]] +
						LUTB[all.toProcess[l][i * framesILikeInColour[0].rows + j * 3]] > LUTR[pf[j * 3 + 2]] + LUTG[pf[j * 3 + 1]] + LUTB[pf[j * 3]])
					{
						for (int m = density - 1; m > l; m--)
						{
							all.toProcess[m][i * framesILikeInColour[0].rows + j * 3] = all.toProcess[m - 1][i * framesILikeInColour[k].rows + j * 3];
							all.toProcess[m][i * framesILikeInColour[0].rows + j * 3 + 1] = all.toProcess[m - 1][i * framesILikeInColour[k].rows + j * 3 + 1];
							all.toProcess[m][i * framesILikeInColour[0].rows + j * 3 + 2] = all.toProcess[m - 1][i * framesILikeInColour[k].rows + j * 3 + 2];
						}
						all.toProcess[l][i * framesILikeInColour[0].rows + j * 3] = pf[j * 3];
						all.toProcess[l][i * framesILikeInColour[0].rows + j * 3 + 1] = pf[j * 3 + 1];
						all.toProcess[l][i * framesILikeInColour[0].rows + j * 3 + 2] = pf[j * 3 + 2];
						break;
					}
				}
				if (k == density - 1)
				{		
						pm[3 * id * properIndexCols + j * 3] = all.toProcess[hDens][i * framesILikeInColour[0].rows + j * 3];
						pm[3 * id * properIndexCols + j * 3 + 1] = all.toProcess[hDens][i * framesILikeInColour[0].rows + j * 3 + 1];
						pm[3 * id * properIndexCols + j * 3 + 2] = all.toProcess[hDens][i * framesILikeInColour[0].rows + j * 3 + 2];
				}
			}
		}
	}
	timer.getTimeWithMessage();
}

int morph_elem = 0;
int morph_size = 0;
int morph_add_operator = 5;
int const max_add_operator = 5;
int morph_operator = 0;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 21;
Mat dst2;

int main(int argc, char* argv[]) {

	int LUTB[255] = {};
	int LUTG[255] = {};
	int LUTR[255] = {};

	const unsigned int nthreads = thread::hardware_concurrency()+1;
	vector<thread> t(nthreads - 1);
	Stopwatch timer;
	//timer.startTimer();
	VideoCapture cap("traffic.mp4");

	float bw = 0.6, gw = 1, rw = 1;
	for (int i = 0; i < 255; ++i)
	{
		LUTB[i] = (int)(i * bw);
	}
	for (int i = 0; i < 255; ++i)
	{
		LUTG[i] = (int)(i * gw);
	}
	for (int i = 0; i < 255; ++i)
	{
		LUTR[i] = (int)(i * rw);
	}

	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	double fps = cap.get(CAP_PROP_FPS);
	int FPS_INT = (int)fps;
	Mat mediane,medianeC;
	int frameCount = cap.get(CAP_PROP_FRAME_COUNT);
	Mat framesILike[density] = {};
	Mat framesILikeInColour[density] = {};
	Mat roiToCheck;
	int iterator = (int)frameCount / density;
	int currIt = 0;
	int i = 0;
	

		//cap.read(framesILike[0]);
		//Mat grayed(framesILike[0].rows, framesILike[0].cols, CV_8UC1);


		//timer.startTimer();
		//simpleGrayscale(framesILike[0], grayed);
		//timer.getTimeWithMessage();


		//cap.read(framesILike[0]);
		//Mat grayed2(framesILike[0].rows, framesILike[0].cols, CV_8UC1);

		//timer.startTimer();
		//efficientGrayscale(framesILike[0], framesILike[0]);
		//timer.getTimeWithMessage();

		//
		//cap.read(framesILike[0]);
		////Mat grayed2(framesILike[0].rows, framesILike[0].cols, CV_8UC1);

		//timer.startTimer();
		//cvtColor(framesILike[0], framesILike[0], COLOR_BGR2GRAY);
		//timer.getTimeWithMessage();
		//while (1) {
		//	imshow("Colour", framesILike[0]);
		//	//imshow("My method", grayed2);
		//	char c = (char)waitKey(1000 / fps);
		//		if (c == 27)
		//			break;
		//}
		
		for (int i = 0; i < density; i++)
		{
			cap.set(1, currIt);
			cap.read(framesILikeInColour[i]);
			if (framesILikeInColour[i].depth() != CV_8U)
				framesILikeInColour[i].convertTo(framesILikeInColour[i], CV_8UC3);
			//framesILikeInColour[i].convertTo(framesILikeInColour[i], CV_8U);
			//efficientGrayscale(framesILikeInColour[i], framesILike[i]);
			currIt += iterator;
		}
		medianeC = Mat(framesILikeInColour[0].rows, framesILikeInColour[0].cols, CV_8UC3);
		/*timer.startTimer();
		array<uchar, density> toProcess;
		timer.startTimer();
		for (int i = 0; i < framesILike[0].cols; i++) {
			for (int j = 0; j < framesILike[0].rows; j++) {
				for (int k = 0; k < density; k++)
				{
					toProcess[k] = framesILike[k].at<uchar>(Point(i, j));
				}
				sort(toProcess.begin(), toProcess.end());
				mediane.at<uchar>(Point(i, j)) = toProcess[(int)density / 2];
			}
		}
		timer.getTimeWithMessage();*/
		//timer.getTimeWithMessage();
	//	timer.startTimer();
		//vector<vector<Mat>> rois(nthreads - 1,vector<Mat>(density));
		//Mat roisArr[nth][density];
		MultiArrMat roisArr(nthreads - 1, density);
		Mat dst(framesILikeInColour[0].rows, framesILikeInColour[0].cols, CV_8UC1);
		dst2 = Mat(framesILikeInColour[0].rows, framesILikeInColour[0].cols, CV_8UC3);
		Mat med(framesILikeInColour[0].rows, framesILikeInColour[0].cols, CV_8UC1);
		Rect roi;
		int roiH = (int)(framesILikeInColour[0].rows / (nthreads - 1));
		int extRoiH = 0;
		int d;
		for (d = 0; d < nthreads-1 ; d++)
		{
			if (d == nthreads - 2)
			{
				extRoiH = framesILikeInColour[0].rows - roiH * (d); //smth is wrong here
				roi = Rect(0, d * roiH, framesILikeInColour[0].cols, extRoiH);
			}
			else
				roi = Rect(0,d*roiH,framesILikeInColour[0].cols,roiH);
			for (int k = 0; k < density; k++)
			{
				roisArr.toProcess[d][k] = framesILikeInColour[k](roi);
			}
			//cout << framesILikeInColour[0].rows << " " << framesILikeRoi[0].rows << endl;
			t[d] = thread(pCarnageArr, roisArr.toProcess[d], ref(medianeC),LUTB, LUTG, LUTR,roiH, d);
		}
		//pCarnageArr(framesILikeInColour, ref(medianeC), LUTB, LUTG, LUTR, d);
		//t[0] = thread(pCarnage, framesILikeInColour, ref(medianeC), i);
		//timer.getTimeWithMessage();
		//
		


		


		namedWindow("Morphology");

		// Create Trackbar to select Morphology operation
		createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", "Morphology", &morph_operator, max_operator, Morphology_Operations);

		createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat - 5:No Operator", "Morphology", &morph_add_operator, max_add_operator, Morphology_Operations);
		// Create Trackbar to select kernel type
		createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", "Morphology",
			&morph_elem, max_elem,
			Morphology_Operations);

		 //Create Trackbar to choose kernel size
		createTrackbar("Kernel size:\n 2n +1", "Morphology",
			&morph_size, max_kernel_size,
			Morphology_Operations);

		Ptr<BackgroundSubtractor> pBackSub;
		pBackSub = createBackgroundSubtractorMOG2();
		Mat fgMask;

		cap.set(1, 0);
		while (1) {
			
			imshow("Mediane", medianeC);
			efficientGrayscale(medianeC, med);
			//imshow("med", med);
			Mat frame;
			 //Capture frame-by-frame
			cap >> frame;
			if (frame.empty()) {
				cap.set(1, 0);
				cap >> frame;
			}
			
				//bitwise_xor(medianeC, frame, frame);
			efficientGrayscale(frame, dst);
			absdiff(med,dst,dst2);
			//pBackSub->apply(frame, dst2);
			threshold(dst2, dst2,35,255,THRESH_BINARY);
			Morphology_Operations(0, 0);

				 //If the frame is empty, break immediately
				
					// Display the resulting frame
			imshow("dst", dst2);
			imshow("Source", frame);
			
			
			

			// Press  ESC on keyboard to exit
			char c = (char)waitKey(1000 / fps);
			if (c == 27)
				break;
		}
		// When everything done, release the video capture object
		cap.release();

	

	// Closes all the frames
		for (int i = 0; i < nthreads - 1;++i)
		{
			t[i].join();
		}
		//t[0].join();
		cv::destroyAllWindows();
		
		return 0;

		
}
void Morphology_Operations(int, void*)
{
	// Since MORPH_X : 2,3,4,5 and 6
	int operation = morph_operator + 2;
	int operation2 = morph_add_operator + 2;

	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	Mat dst3(dst2.rows, dst2.cols, CV_8UC3);
	/// Apply the specified morphology operation
	morphologyEx(dst2, dst3, operation, element);
	if(morph_add_operator!=5)
		morphologyEx(dst3, dst3, operation2, element);
	imshow("Morphology", dst3);
}