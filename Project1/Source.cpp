#include<opencv2/opencv.hpp>
#include<iostream>
#include "opencv2/opencv.hpp"
#include <sstream>
#include <windows.h>
#include "Stopwatch.h"

using namespace std;
using namespace cv;



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
			for (int j = 0; j< dim; ++j)
			{
				toProcess[i][j] =(uchar) 255;
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


int main(int argc, char* argv[]) {

	Stopwatch timer;
	timer.startTimer();
	VideoCapture cap("traffic.mp4");

	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	double fps = cap.get(CAP_PROP_FPS);
	int FPS_INT = (int)fps;
	Mat mediane,medianeC;
	int frameCount = cap.get(CAP_PROP_FRAME_COUNT);
	const int density = 30;
	Mat framesILike[density] = {};
	Mat framesILikeInColour[density] = {};
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
			framesILikeInColour[i].convertTo(framesILikeInColour[i], CV_8U);
			efficientGrayscale(framesILikeInColour[i], framesILike[i]);
			currIt += iterator;
		}
		mediane = Mat(framesILike[0].rows, framesILike[0].cols, CV_8UC1);
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


		const int dim = framesILikeInColour[0].cols * framesILikeInColour[0].rows * 3;
		
		MultiArr all(density, dim);

		timer.getTimeWithMessage();
		timer.startTimer();
		int nRows = framesILikeInColour[0].rows;
		int nCols = framesILikeInColour[0].cols;
		uchar* pf;
		uchar* pm;
		int LUTB[255] = {};
		int LUTG[255] = {};
		int LUTR[255] = {};
		float bw = 1.0, gw = 1.0, rw = 1.0;
		for (int i = 0; i < 255; ++i)
		{
			LUTB[i] = (int)(i*bw);
		}
		for (int i = 0; i < 255; ++i)
		{
			LUTG[i] = (int)(i * gw);
		}
		for (int i = 0; i < 255; ++i)
		{
			LUTR[i] = (int)(i * rw);
		}
		for (int k = 0; k < density; k++)
		{
			if (framesILikeInColour[k].isContinuous())
			{
				nCols *= nRows;
				nRows = 1;
			}
			for (int i = 0; i < nRows; i++) {
				pf = framesILikeInColour[k].ptr<uchar>(i);
				pm = medianeC.ptr<uchar>(i);
				for (int j = 0; j < nCols; j++) {
					for (int l = 0; l < density; l++)
					{	
						if (LUTR[all.toProcess[l][i * framesILikeInColour[k].rows + j * 3 + 2]] +
							LUTG[all.toProcess[l][i * framesILikeInColour[k].rows + j * 3 + 1]] +
							LUTB[all.toProcess[l][i * framesILikeInColour[k].rows + j * 3 ]] > LUTR[pf[j * 3 +2]] + LUTG[pf[j * 3 + 1]] + LUTB[pf[j * 3]])
						{
							for (int m = density - 1; m > l; m--)
							{
								all.toProcess[m][i * framesILikeInColour[k].rows + j * 3] = all.toProcess[m - 1][i * framesILikeInColour[k].rows + j * 3];
								all.toProcess[m][i * framesILikeInColour[k].rows + j * 3 + 1] = all.toProcess[m - 1][i * framesILikeInColour[k].rows + j * 3 + 1];
								all.toProcess[m][i * framesILikeInColour[k].rows + j * 3 + 2] = all.toProcess[m - 1][i * framesILikeInColour[k].rows + j * 3 + 2];
							}
							all.toProcess[l][i * framesILikeInColour[k].rows + j * 3] = pf[j * 3];
							all.toProcess[l][i * framesILikeInColour[k].rows + j * 3 + 1] = pf[j * 3 + 1];
							all.toProcess[l][i * framesILikeInColour[k].rows + j * 3 + 2] = pf[j * 3 + 2];
							break;
						}
					}
					if (k == density - 1)
					{
						pm[j * 3] = all.toProcess[(int)(density * 0.5f)][i * framesILikeInColour[k].rows + j * 3];
						pm[j * 3 + 1] = all.toProcess[(int)(density * 0.5f)][i * framesILikeInColour[k].rows + j * 3 + 1];
						pm[j * 3 + 2] = all.toProcess[(int)(density * 0.5f)][i * framesILikeInColour[k].rows + j * 3 + 2];

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
		cap.set(1, 0);
		while (1) {

			imshow("After", medianeC);

			Mat frame;
			// Capture frame-by-frame
			cap >> frame;

			// If the frame is empty, break immediately
			if (frame.empty())
				break;

			// Display the resulting frame
			imshow("Frame", frame);

			// Press  ESC on keyboard to exit
			char c = (char)waitKey(1000 / fps);
			if (c == 27)
				break;
		}

		// When everything done, release the video capture object
		cap.release();

	

	// Closes all the frames
	cv::destroyAllWindows();

	return 0;
}





//LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
//LARGE_INTEGER Frequency;

//cap.read(framesILike[0]);
//Mat grayed(framesILike[0].rows, framesILike[0].cols, CV_8UC1);

//QueryPerformanceFrequency(&Frequency);
//QueryPerformanceCounter(&StartingTime);

//simpleGrayscale(framesILike[0], grayed);

//QueryPerformanceCounter(&EndingTime);
//ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

//ElapsedMicroseconds.QuadPart *= 1000000;
//ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

//cout << ElapsedMicroseconds.QuadPart << endl;

//QueryPerformanceFrequency(&Frequency);
//QueryPerformanceCounter(&StartingTime);

//cvtColor(framesILike[i], framesILike[i], COLOR_BGR2GRAY);

//QueryPerformanceCounter(&EndingTime);
//ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

//ElapsedMicroseconds.QuadPart *= 1000000;
//ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

//cout << ElapsedMicroseconds.QuadPart << endl;
