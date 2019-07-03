#include <windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/opencv.hpp"
#include "Stopwatch.h"

using namespace std;
using namespace cv;

Stopwatch::Stopwatch() {
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;

}

void Stopwatch::startTimer() {
	//cout << "start" << endl;
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
}

void Stopwatch::getTimeWithMessage()
{
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

	cout << "action took: " <<ElapsedMicroseconds.QuadPart <<" microseconds"<< endl;
}

double Stopwatch::getTime()
{
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

	return ElapsedMicroseconds.QuadPart;
}
