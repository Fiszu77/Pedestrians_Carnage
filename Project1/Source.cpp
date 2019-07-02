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
#include <mutex>

using namespace std;
using namespace cv;

//TODO detect when threads finished processing rois, leave moving things


void efficientGrayscale(Mat toGray, Mat& out);

void simpleGrayscale(Mat& toGray, Mat& out);
const int density = 15;

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

int LUTB[256] = {};
int LUTG[256] = {};
int LUTR[256] = {};

void pCarnage(vector<Mat>& framesILikeInColour, Mat& medianeC, int id = 0);
void pCarnageArr(Mat framesILikeInColour[density], Mat& medianeC, int defRoiH, int id = 0);
void goThreads(Mat framesILikeInColour[density], Mat& medianeC, int roiH, int nthreads, MultiArrMat& roisArr);

int morph_elem = 0;
int morph_size = 0;
int morph_add_operator = 5;
int morph_operator = 0;
int const max_add_operator = 5;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 21;
Mat dst2;
vector<thread> t;

void Morphology_Operations(int, void*);

mutex mtx;

int tDone = -1;

int main(int argc, char* argv[]) {
	morph_elem = 0;
	morph_size = 0;
	morph_add_operator = 5;
	morph_operator = 0;

	const unsigned int nthreads = thread::hardware_concurrency();
	t = vector<thread>(nthreads - 1);
	Stopwatch timer;
	//timer.startTimer();
	VideoCapture cap("zverev.mp4");

	float bw = 0.6, gw = 1, rw = 1;
	for (int i = 0; i <= 255; ++i)
	{
		LUTB[i] = (int)(i * bw);
	}
	for (int i = 0; i <= 255; ++i)
	{
		LUTG[i] = (int)(i * gw);
	}
	for (int i = 0; i <= 255; ++i)
	{
		LUTR[i] = (int)(i * rw);
	}

	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	double fps = cap.get(CAP_PROP_FPS);
	int FPS_INT = (int)fps;
	Mat mediane, medianeC;
	int frameCount = cap.get(CAP_PROP_FRAME_COUNT);
	Mat framesILike[density] = {};
	Mat framesILikeInColour[density] = {};
	Mat framesILikeInColour2[density] = {};
	Mat roiToCheck;
	int iterator = (int)frameCount / density;
	int currIt = 0;
	int i = 0;

	cap.set(1, 0);
	cap.read(framesILikeInColour[0]);
	if (framesILikeInColour[0].depth() != CV_8U)
		framesILikeInColour[0].convertTo(framesILikeInColour[i], CV_8UC3);
	cap.set(1, 0);
	/*for (int i = 0; i < density; i++)
	{
		cap.set(1, currIt);
		cap.read(framesILikeInColour[i]);
		if (framesILikeInColour[i].depth() != CV_8U)
			framesILikeInColour[i].convertTo(framesILikeInColour[i], CV_8UC3);
		currIt += iterator;
	}*/
	medianeC = Mat(framesILikeInColour[0].rows, framesILikeInColour[0].cols, CV_8UC3);

	MultiArrMat roisArr(nthreads - 1, density);
	Mat dst(framesILikeInColour[0].rows, framesILikeInColour[0].cols, CV_8UC1);
	dst2 = Mat(framesILikeInColour[0].rows, framesILikeInColour[0].cols, CV_8UC3);
	Mat med(framesILikeInColour[0].rows, framesILikeInColour[0].cols, CV_8UC1);
	int roiH = (int)(framesILikeInColour[0].rows / (nthreads - 1));
	//goThreads(framesILikeInColour, medianeC, roiH, nthreads, roisArr);

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
	int FILiterator = 0;
	int interval = 10;
	int iter = 0;
	bool first = false, oneortwo = false;
	Mat frame;
	while (1) {
		
		efficientGrayscale(medianeC, med);
		imshow("Mediane", medianeC);
		//imshow("med", med);
		//Capture frame-by-frame
		if (iter % interval)
		{
			if (oneortwo) {
				cap >> framesILikeInColour[FILiterator];
				if (framesILikeInColour[i].depth() != CV_8U)
					framesILikeInColour[i].convertTo(framesILikeInColour[i], CV_8UC3);
				
			}
			else {
				cap >> framesILikeInColour2[FILiterator];
				if (framesILikeInColour2[i].depth() != CV_8U)
					framesILikeInColour2[i].convertTo(framesILikeInColour2[i], CV_8UC3);
				
			}
			FILiterator++;
			cap.set(1, iter);
		}
		iter++;
		cap >> frame;
		/*if (frame.empty()) { // trzeba naprawic
			cap.set(1, 0);
			cap >> frame;
		}*/
		if (FILiterator == density - 1)
		{
			cout <<"FILit:"<< FILiterator << endl;
			if (first) {
				while (tDone != nthreads - 2)
				{

				}
			}
			if (oneortwo) {
				goThreads(framesILikeInColour, medianeC, roiH, nthreads, roisArr);
				oneortwo = !oneortwo;
			}
			else
			{
				goThreads(framesILikeInColour2, medianeC, roiH, nthreads, roisArr);
				oneortwo = !oneortwo;
			}
			FILiterator = 0;
			first = true;
		}
			

		//bitwise_xor(medianeC, frame, frame);
		//efficientGrayscale(frame, dst);
		//absdiff(med, dst, dst2);
		//pBackSub->apply(frame, dst2);
		//threshold(dst2, dst2, 25, 255, THRESH_BINARY);
		//Morphology_Operations(0, 0);

		// Display the resulting frame
		//imshow("dst", dst2);
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
	if (morph_add_operator != 5)
		morphologyEx(dst3, dst3, operation2, element);
	imshow("Morphology", dst3);
}

void simpleGrayscale(Mat& toGray, Mat& out)
{
	//Mat out(toGray.cols, toGray.rows, CV_8UC1);
	if (toGray.depth() != CV_8U)
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

void efficientGrayscale(Mat toGray, Mat& out)
{
	const int lutlength = 768;
	uchar LUT[lutlength] = {};
	float pointthree = 1 / 3;
	for (int i = 0; i < lutlength; ++i)
	{
		LUT[i] = (uchar)(i / 3);
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

void pCarnageArr(Mat framesILikeInColour[density], Mat& medianeC, int defRoiH, int id)
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
	mtx.lock();
	tDone++;
	cout <<"tDOne: "<< tDone <<endl;
	timer.getTimeWithMessage();
	mtx.unlock();
	
}

void pCarnage(vector<Mat>& framesILikeInColour, Mat& medianeC, int id)
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

void goThreads(Mat framesILikeInColour[density], Mat& medianeC, int roiH, const int nthreads, MultiArrMat& roisArr)
{
	Rect roi;
	int extRoiH = 0;
	for (int d = 0; d < nthreads - 1; d++)
	{
		if (d == nthreads - 2)
		{
			extRoiH = framesILikeInColour[0].rows - roiH * (d);
			roi = Rect(0, d * roiH, framesILikeInColour[0].cols, extRoiH);
		}
		else
			roi = Rect(0, d * roiH, framesILikeInColour[0].cols, roiH);
		for (int k = 0; k < density; k++)
		{
			roisArr.toProcess[d][k] = framesILikeInColour[k](roi);
		}
		t[d] = thread(pCarnageArr, roisArr.toProcess[d], ref(medianeC), roiH, d);
	}
}
