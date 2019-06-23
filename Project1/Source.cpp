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
			po[j] = (uchar)((pg[(j - 1) * 3 + 1] + pg[(j - 1) * 3 + 2] + pg[(j - 1) * 3 + 3]) / 3);
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
	VideoCapture cap("ny.mp4");

	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	double fps = cap.get(CAP_PROP_FPS);
	int FPS_INT = (int)fps;
	Mat mediane;
	int frameCount = cap.get(CAP_PROP_FRAME_COUNT);
	const int density = 30;
	Mat framesILike[density] = {};
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
			cap.read(framesILike[i]);
			if (framesILike[i].depth() != CV_8U)
				framesILike[i].convertTo(framesILike[i], CV_8UC3);
			framesILike[i].convertTo(framesILike[i], CV_8U);
			efficientGrayscale(framesILike[i], framesILike[i]);
			currIt += iterator;
		}
		mediane = Mat(framesILike[0].rows, framesILike[0].cols, CV_8UC1);

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


		const int dim = framesILike[0].cols * framesILike[0].rows;
		
		MultiArr all(density, dim);

		timer.getTimeWithMessage();
		timer.startTimer();
		int nRows = framesILike[0].rows;
		int nCols = framesILike[0].cols;
		uchar* pf;
		uchar* pm;
		for (int k = 0; k < density; k++)
		{
			if (framesILike[k].isContinuous())
			{
				nCols *= nRows;
				nRows = 1;
			}
			for (int i = 0; i < nRows; i++) {
				pf = framesILike[k].ptr<uchar>(i);
				pm = mediane.ptr<uchar>(i);
				for (int j = 0; j < nCols; j++) {
					
					for (int l = 0; l < density; l++)
					{
						
						if (all.toProcess[l][i * framesILike[k].rows + j] > pf[j])
						{
							for (int m = density - 1; m > l;m--)
							{
								all.toProcess[m][i * framesILike[k].rows + j] = all.toProcess[m - 1][i * framesILike[k].rows + j];
							}
							all.toProcess[l][i * framesILike[k].rows + j] = pf[j];
							break;
						}
					}
					if (k == density - 1)
					{
						pm[j] = all.toProcess[(int)(density * 0.5f)][i * framesILike[k].rows + j];
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

			imshow("After", mediane);

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
	destroyAllWindows();

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
